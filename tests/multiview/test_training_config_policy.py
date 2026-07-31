from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

from data_toolkit.pipeline.training_config_policy import (
    NODE17_PATH_REPLACEMENTS,
    STAGE_POLICIES,
    StageTrainingPolicy,
    create_node17_runtime_configs,
    node17_runtime_config_evidence,
    rebase_json_paths,
    validate_finetuning_configs,
)
from tests.multiview.test_configs import CONFIGS


CURRENT_TRAINING_POLICY = {
    "num_workers": 2,
    "i_print": 10,
    "i_log": 10,
    "i_sample": 1000,
    "i_save": 1000,
    "max_checkpoints": 3,
    "max_steps": 20_000,
}


def _copied_configs(tmp_path: Path) -> dict[str, Path]:
    outputs = {}
    for stage, source in CONFIGS.items():
        output = tmp_path / "configs" / source.name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, output)
        outputs[stage] = output
    return outputs


def _mutate_config(
    configs: dict[str, Path], stage: str, field: str, value: object
) -> None:
    selected = configs[stage]
    config = json.loads(selected.read_text())
    config["trainer"]["args"][field] = value
    selected.write_text(json.dumps(config))


def test_stage_policies_define_exact_six_gpu_batches():
    assert STAGE_POLICIES == {
        "ss64": StageTrainingPolicy(8, 4, 48),
        "shape512": StageTrainingPolicy(8, 4, 48),
        "shape1024": StageTrainingPolicy(2, 1, 12),
        "pbr1024": StageTrainingPolicy(2, 1, 12),
    }


def test_finetuning_configs_enforce_current_typed_policy():
    parsed = validate_finetuning_configs(CONFIGS)

    assert tuple(parsed) == tuple(CONFIGS)
    for stage, config in parsed.items():
        args = config["trainer"]["args"]
        policy = STAGE_POLICIES[stage]
        assert {
            name: args[name] for name in CURRENT_TRAINING_POLICY
        } == CURRENT_TRAINING_POLICY
        assert args["multiview_stage"] == stage
        assert args["batch_size_per_gpu"] == policy.batch_size_per_gpu
        assert args["batch_split"] == policy.batch_split
        assert (
            policy.batch_size_per_gpu * 6
            == policy.six_gpu_global_batch
        )
        assert all(
            type(args[name]) is int
            for name in (
                *CURRENT_TRAINING_POLICY,
                "batch_size_per_gpu",
                "batch_split",
            )
        )
        if stage == "ss64":
            assert args["snapshot_dataset_on_start"] is False
        else:
            assert "snapshot_dataset_on_start" not in args


@pytest.mark.parametrize(
    "field",
    (
        "batch_size_per_gpu",
        "batch_split",
        "max_steps",
        "num_workers",
        "i_print",
        "i_log",
        "i_sample",
        "i_save",
        "max_checkpoints",
    ),
)
def test_finetuning_configs_reject_bool_as_approved_integer(tmp_path, field):
    configs = _copied_configs(tmp_path)
    _mutate_config(configs, "ss64", field, True)

    with pytest.raises(ValueError, match="source config semantics"):
        validate_finetuning_configs(configs)


@pytest.mark.parametrize(
    ("field", "unapproved"),
    (
        ("batch_size_per_gpu", 7),
        ("batch_split", 3),
        ("max_steps", 19_999),
        ("num_workers", 1),
        ("i_print", 11),
        ("i_log", 11),
        ("i_sample", 999),
        ("i_save", 999),
        ("max_checkpoints", 4),
        ("multiview_stage", "shape512"),
    ),
)
def test_finetuning_configs_reject_each_unapproved_policy_value(
    tmp_path, field, unapproved
):
    configs = _copied_configs(tmp_path)
    _mutate_config(configs, "ss64", field, unapproved)

    with pytest.raises(ValueError, match="source config semantics"):
        validate_finetuning_configs(configs)


@pytest.mark.parametrize(
    ("stage", "mutation"),
    (
        ("ss64", "missing"),
        ("ss64", "true"),
        ("shape512", "present"),
        ("shape1024", "present"),
        ("pbr1024", "present"),
    ),
)
def test_finetuning_configs_enforce_stage_specific_snapshot_key(
    tmp_path, stage, mutation
):
    configs = _copied_configs(tmp_path)
    selected = configs[stage]
    config = json.loads(selected.read_text())
    args = config["trainer"]["args"]
    if mutation == "missing":
        args.pop("snapshot_dataset_on_start")
    else:
        args["snapshot_dataset_on_start"] = mutation == "true"
    selected.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="source config semantics"):
        validate_finetuning_configs(configs)


def test_finetuning_configs_require_exact_stage_order():
    reordered = dict(reversed(tuple(CONFIGS.items())))

    with pytest.raises(ValueError, match="ordered exactly"):
        validate_finetuning_configs(reordered)


def test_rebase_json_paths_changes_only_complete_path_prefix_values():
    original = {
        "exact": "/file2/youngwoo/pixal3d",
        "nested": [
            "/file2/youngwoo/pixal3d/train/data.json",
            {
                "checkpoint": (
                    "/file3/youngwoo/pixal3d/train/checkpoint.pt"
                )
            },
        ],
        "/file3/youngwoo/pixal3d/key": "keys are not values",
        "substring": "prefix-/file2/youngwoo/pixal3d/train",
        "sibling": "/file2/youngwoo/pixal3d-other/train",
        "number": 2,
        "boolean": False,
        "none": None,
    }

    assert rebase_json_paths(original, NODE17_PATH_REPLACEMENTS) == {
        "exact": "/root/data2/pixal3d",
        "nested": [
            "/root/data2/pixal3d/train/data.json",
            {"checkpoint": "/root/data3/pixal3d/train/checkpoint.pt"},
        ],
        "/file3/youngwoo/pixal3d/key": "keys are not values",
        "substring": "prefix-/file2/youngwoo/pixal3d/train",
        "sibling": "/file2/youngwoo/pixal3d-other/train",
        "number": 2,
        "boolean": False,
        "none": None,
    }
    assert original["exact"] == "/file2/youngwoo/pixal3d"


def test_node17_runtime_configs_change_only_machine_path_prefixes(tmp_path):
    outputs = create_node17_runtime_configs(
        CONFIGS, tmp_path / "runtime"
    )

    assert tuple(outputs) == tuple(CONFIGS)
    for stage, output in outputs.items():
        source = json.loads(CONFIGS[stage].read_text())
        runtime = json.loads(output.read_text())
        assert output.name == f"{CONFIGS[stage].stem}.node17.json"
        assert runtime["trainer"]["args"]["num_workers"] == source[
            "trainer"
        ]["args"]["num_workers"]
        assert "/file2/youngwoo/pixal3d" not in output.read_text()
        assert "/file3/youngwoo/pixal3d" not in output.read_text()
        restored = rebase_json_paths(
            runtime,
            (
                ("/root/data2/pixal3d", "/file2/youngwoo/pixal3d"),
                ("/root/data3/pixal3d", "/file3/youngwoo/pixal3d"),
            ),
        )
        assert restored == source
        assert output.read_bytes() == (
            json.dumps(runtime, indent=2, sort_keys=True) + "\n"
        ).encode()


def test_node17_runtime_configs_reuse_identical_complete_set(tmp_path):
    output_root = tmp_path / "runtime"
    outputs = create_node17_runtime_configs(CONFIGS, output_root)
    before = {stage: path.read_bytes() for stage, path in outputs.items()}

    reused = create_node17_runtime_configs(CONFIGS, output_root)

    assert reused == outputs
    assert {
        stage: path.read_bytes() for stage, path in reused.items()
    } == before


def test_node17_runtime_configs_create_in_existing_empty_root(tmp_path):
    output_root = tmp_path / "runtime"
    output_root.mkdir()

    outputs = create_node17_runtime_configs(CONFIGS, output_root)

    assert all(path.is_file() for path in outputs.values())


def test_node17_runtime_configs_reject_partial_existing_set(tmp_path):
    output_root = tmp_path / "runtime"
    outputs = create_node17_runtime_configs(CONFIGS, output_root)
    retained = outputs["ss64"]
    retained_bytes = retained.read_bytes()
    outputs["shape512"].unlink()

    with pytest.raises(ValueError, match="partial runtime config output"):
        create_node17_runtime_configs(CONFIGS, output_root)

    assert retained.read_bytes() == retained_bytes
    assert not outputs["shape512"].exists()


def test_node17_runtime_configs_reject_different_existing_bytes(tmp_path):
    output_root = tmp_path / "runtime"
    outputs = create_node17_runtime_configs(CONFIGS, output_root)
    selected = outputs["shape1024"]
    selected.write_text(
        json.dumps(json.loads(selected.read_text()), separators=(",", ":"))
    )
    before = selected.read_bytes()

    with pytest.raises(
        FileExistsError,
        match="existing runtime config has different bytes",
    ):
        create_node17_runtime_configs(CONFIGS, output_root)

    assert selected.read_bytes() == before


def test_node17_runtime_evidence_proves_reverse_rebase_and_policy(tmp_path):
    outputs = create_node17_runtime_configs(
        CONFIGS, tmp_path / "runtime"
    )

    evidence = node17_runtime_config_evidence(outputs, CONFIGS)

    for stage, record in evidence.items():
        policy = STAGE_POLICIES[stage]
        assert record == {
            "path": str(outputs[stage]),
            "sha256": sha256(outputs[stage].read_bytes()).hexdigest(),
            "source_config": {
                "path": str(CONFIGS[stage]),
                "sha256": sha256(CONFIGS[stage].read_bytes()).hexdigest(),
            },
            "batch_size_per_gpu": policy.batch_size_per_gpu,
            "batch_split": policy.batch_split,
            "six_gpu_global_batch": policy.six_gpu_global_batch,
            "max_steps": 20_000,
            "save_interval": 1000,
            "retained_checkpoints": 3,
            "snapshot_interval": 1000,
            "startup_dataset_snapshot": stage != "ss64",
            "num_workers_per_rank": 2,
        }


def test_node17_runtime_evidence_rejects_non_path_drift(tmp_path):
    outputs = create_node17_runtime_configs(
        CONFIGS, tmp_path / "runtime"
    )
    selected = outputs["pbr1024"]
    runtime = json.loads(selected.read_text())
    runtime["trainer"]["args"]["num_workers"] = 1
    selected.write_text(json.dumps(runtime))

    with pytest.raises(ValueError, match="exact source path rebase"):
        node17_runtime_config_evidence(outputs, CONFIGS)


def test_node17_runtime_evidence_rejects_unchanged_source_paths(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    unchanged = {}
    for stage, source in CONFIGS.items():
        runtime = runtime_root / f"{source.stem}.node17.json"
        runtime.write_bytes(source.read_bytes())
        unchanged[stage] = runtime

    with pytest.raises(ValueError, match="exact source path rebase"):
        node17_runtime_config_evidence(unchanged, CONFIGS)
