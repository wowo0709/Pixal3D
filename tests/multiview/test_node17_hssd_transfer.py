from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from data_toolkit.pipeline import node17_hssd_transfer as transfer
from data_toolkit.pipeline.training_manifest import STAGES
from data_toolkit.pipeline.training_preflight import StagePreflight


NODE16_HSSD = Path(
    "/home/youngwoo/data/pixal3d/train/production/hssd"
)
DATA2_ROOT = Path("/root/data2/pixal3d")


def _paths(tmp_path: Path) -> transfer.Node17HssdTransferPaths:
    production_root = tmp_path / "train/production"
    production_root.mkdir(parents=True)
    return transfer.Node17HssdTransferPaths(
        source_host="youngwoo@n16.unist.info",
        source_port=55555,
        source_root=NODE16_HSSD,
        data2_root=DATA2_ROOT,
        production_root=production_root,
        staging_root=production_root / ".hssd-node16-transfer",
    )


def _materialization(
    stage: str,
    stage_root: Path,
    *,
    shared_root: Path = Path("/file2/youngwoo/pixal3d"),
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "source": "HSSD",
        "stage": stage,
        "stage_root": str(stage_root),
        "asset_count": 2,
        "stage_scope": ["asset-a", "asset-b"],
        "stage_scope_sha256": sha256(b"asset-a\nasset-b").hexdigest(),
        "source_indexes": [
            {
                "path": str(
                    shared_root
                    / "prepared/index/HSSD/HSSD-00000.json"
                ),
                "sha256": "1" * 64,
            }
        ],
        "counts": {
            "stages": {stage: 2},
            "training_exclusions": {stage: 0},
        },
        "packs": [
            {
                "path": str(shared_root / f"prepared/packs/{stage}.zip"),
                "sha256": "2" * 64,
                "tool_commit": "pack-commit",
            }
        ],
        "tool_commits": ["materializer-commit"],
    }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _stage_result(
    stage: str, root: Path, document: dict[str, object]
) -> StagePreflight:
    return StagePreflight(
        stage=stage,
        root=root,
        asset_count=2,
        asset_scope_sha256=document["stage_scope_sha256"],
        anchors_checked=4,
        validation_counts={"assets": 2, "renders": 16},
        materialization_bytes=_canonical_bytes(document),
        source="HSSD",
    )


def _write_staged_materializations(
    paths: transfer.Node17HssdTransferPaths,
    root_prefix: Path,
) -> dict[str, StagePreflight]:
    results = {}
    for stage in STAGES:
        active = paths.staging_root / stage / "active"
        active.mkdir(parents=True)
        document = _materialization(
            stage, root_prefix / stage / "active"
        )
        raw = _canonical_bytes(document)
        (active / "materialization.json").write_bytes(raw)
        (active / "payload.bin").write_bytes(stage.encode())
        results[stage] = _stage_result(stage, active, document)
    return results


def _write_existing_canonical_tree(
    paths: transfer.Node17HssdTransferPaths,
) -> None:
    for stage in STAGES:
        active = paths.canonical_root / stage / "active"
        active.mkdir(parents=True)
        document = _materialization(
            stage,
            paths.canonical_root / stage / "active",
            shared_root=DATA2_ROOT,
        )
        (active / "materialization.json").write_bytes(
            _canonical_bytes(document)
        )
        (active / "payload.bin").write_bytes(stage.encode())
    publication = paths.canonical_root / "publication"
    publication.mkdir()
    (publication / "report.json").write_text("{}")
    (publication / "handoff.json").write_text("{}")
    (paths.canonical_root / "training_data.json").write_text("{}")


def _validated_existing_chain(
    paths: transfer.Node17HssdTransferPaths,
    *,
    root: Path | None = None,
):
    selected = paths.canonical_root if root is None else root
    return SimpleNamespace(
        source="HSSD",
        path=paths.canonical_root / "training_data.json",
        sha256="a" * 64,
        report_path=selected / "publication/report.json",
        handoff_path=selected / "publication/handoff.json",
        stages={
            stage: SimpleNamespace(
                total_count=2,
                data_dir=transfer.training_preflight.stage_data_dir(
                    "HSSD", stage, selected / stage / "active"
                ),
            )
            for stage in STAGES
        },
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("source_host", "root@n16.unist.info", "source host"),
        ("source_port", 22, "source port"),
        (
            "source_root",
            Path("/home/youngwoo/data/pixal3d/train/production/other"),
            "source root",
        ),
        ("data2_root", Path("/root/other"), "data2 root"),
        ("production_root", Path("relative-production"), "absolute"),
        ("staging_root", Path("relative-staging"), "absolute"),
    ),
)
def test_plan_rejects_wrong_source_identity_or_noncanonical_roots(
    tmp_path, field, value, message
):
    paths = replace(_paths(tmp_path), **{field: value})

    with pytest.raises(ValueError, match=message):
        transfer.plan_hssd_transfer(paths)


def test_plan_rejects_wrong_staging_identity(tmp_path):
    paths = _paths(tmp_path)
    paths = replace(
        paths, staging_root=paths.production_root / ".other-transfer"
    )

    with pytest.raises(ValueError, match="staging root"):
        transfer.plan_hssd_transfer(paths)


def test_plan_builds_stage_only_resumable_and_checksum_commands(tmp_path):
    paths = _paths(tmp_path)

    plan = transfer.plan_hssd_transfer(paths)
    transfer_command = plan["transfer_command"]
    verification_command = plan["verification_command"]

    assert transfer_command[:4] == [
        "rsync",
        "-a",
        "--partial",
        "--info=progress2",
    ]
    for stage in STAGES:
        assert f"--include=/{stage}/***" in transfer_command
    assert "--exclude=*" in transfer_command
    assert "publication" not in " ".join(transfer_command)
    assert "training_data.json" not in " ".join(transfer_command)
    assert transfer_command[-2] == (
        "youngwoo@n16.unist.info:"
        "/home/youngwoo/data/pixal3d/train/production/hssd/"
    )
    assert transfer_command[-1] == str(paths.staging_root) + "/"
    for option in (
        "--checksum",
        "--dry-run",
        "--itemize-changes",
        "--delete",
    ):
        assert option in verification_command
    assert "--info=progress2" not in verification_command
    assert not paths.staging_root.exists()


def test_plan_rejects_invalid_existing_canonical_chain(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    paths.canonical_root.mkdir()
    monkeypatch.setattr(
        transfer,
        "validate_source_training_data",
        lambda *_args: (_ for _ in ()).throw(ValueError("bad chain")),
    )

    with pytest.raises(ValueError, match="canonical HSSD"):
        transfer.plan_hssd_transfer(paths)


def test_valid_existing_canonical_chain_is_reused_with_reconstructed_source_evidence(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    _write_existing_canonical_tree(paths)
    validated = _validated_existing_chain(paths)
    monkeypatch.setattr(
        transfer,
        "validate_source_training_data",
        lambda *_args: validated,
    )

    result = transfer.transfer_and_publish_hssd(
        paths,
        {
            stage: tmp_path / f"{stage}.node17.json"
            for stage in STAGES
        },
        command_runner=lambda _command: pytest.fail(
            "valid canonical reuse must not contact Node16"
        ),
    )

    assert result.stage_counts == dict.fromkeys(STAGES, 2)
    assert all(
        result.original_materialization_sha256[stage]
        != result.canonical_materialization_sha256[stage]
        for stage in STAGES
    )
    assert result.source_inventory.file_count == (
        result.target_inventory.file_count
    )
    assert result.source_inventory.logical_bytes != (
        result.target_inventory.logical_bytes
    )
    assert set(result.elapsed_seconds) == {
        "inventory",
        "transfer",
        "verification",
        "evidence_rebase",
        "strict_preflight",
        "promotion",
        "total",
    }


def test_existing_canonical_reuse_rejects_externally_bound_chain(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    _write_existing_canonical_tree(paths)
    external = tmp_path / "external/hssd"
    monkeypatch.setattr(
        transfer,
        "validate_source_training_data",
        lambda *_args: _validated_existing_chain(
            paths, root=external
        ),
    )

    with pytest.raises(ValueError, match="canonical HSSD"):
        transfer.plan_hssd_transfer(paths)


def test_existing_canonical_reuse_rejects_noncanonical_materialization_bytes(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    _write_existing_canonical_tree(paths)
    selected = (
        paths.canonical_root / "ss64/active/materialization.json"
    )
    selected.write_text(json.dumps(json.loads(selected.read_text())))
    monkeypatch.setattr(
        transfer,
        "validate_source_training_data",
        lambda *_args: _validated_existing_chain(paths),
    )

    with pytest.raises(ValueError, match="canonical serialization"):
        transfer.transfer_and_publish_hssd(
            paths,
            {
                stage: tmp_path / f"{stage}.node17.json"
                for stage in STAGES
            },
            command_runner=lambda _command: pytest.fail(
                "invalid canonical reuse must not contact Node16"
            ),
        )


def test_existing_canonical_chain_rejects_extra_top_level_path(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    paths.canonical_root.mkdir()
    (paths.canonical_root / "unexpected.txt").write_text("unsafe")
    monkeypatch.setattr(
        transfer,
        "validate_source_training_data",
        lambda *_args: _validated_existing_chain(paths),
    )

    with pytest.raises(ValueError, match="canonical HSSD"):
        transfer.plan_hssd_transfer(paths)


def test_disk_admission_uses_source_size_plus_larger_margin(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    inventory = transfer.TreeInventory(file_count=4, logical_bytes=100)
    required = 100 + 10 * 1024**3
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(
            required * 2, required + 1, required - 1
        ),
    )

    with pytest.raises(ValueError, match="insufficient local free space"):
        transfer._admit_free_space(paths, inventory)

    assert not paths.staging_root.exists()


def test_safe_staging_is_resumable_but_symlink_is_rejected(tmp_path):
    paths = _paths(tmp_path)
    paths.staging_root.mkdir()
    partial = paths.staging_root / "ss64"
    partial.mkdir()
    (partial / "partial.bin").write_bytes(b"partial")

    transfer.plan_hssd_transfer(paths)

    (paths.staging_root / "unsafe").symlink_to(partial)
    with pytest.raises(ValueError, match="unsafe staging"):
        transfer.plan_hssd_transfer(paths)


def test_execution_reuses_existing_safe_staging_for_resume(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    paths.staging_root.mkdir()
    inventory = transfer.TreeInventory(1, 1)
    monkeypatch.setattr(
        transfer,
        "_remote_inventory",
        lambda _paths, _runner: inventory,
    )
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(
            20 * 1024**3, 1, 20 * 1024**3 - 1
        ),
    )

    class ResumeReached(RuntimeError):
        pass

    def runner(_command):
        raise ResumeReached("rsync reached existing staging")

    with pytest.raises(ResumeReached, match="rsync reached"):
        transfer.transfer_and_publish_hssd(
            paths,
            {
                stage: tmp_path / f"{stage}.node17.json"
                for stage in STAGES
            },
            command_runner=runner,
        )

    assert paths.staging_root.is_dir()


def test_verification_rejects_any_itemized_rsync_output(tmp_path):
    paths = _paths(tmp_path)
    command = transfer.plan_hssd_transfer(paths)["verification_command"]

    def runner(_command):
        return subprocess.CompletedProcess(
            _command, 0, stdout=">fcs....... ss64/active/payload.bin\n"
        )

    with pytest.raises(ValueError, match="checksum verification"):
        transfer._run_verification(command, runner)


def test_verification_command_omits_progress_and_accepts_empty_output(tmp_path):
    paths = _paths(tmp_path)
    command = transfer.plan_hssd_transfer(paths)["verification_command"]

    def runner(_command):
        return subprocess.CompletedProcess(
            _command,
            0,
            stdout="",
        )

    assert "--info=progress2" not in command
    transfer._run_verification(command, runner)


def test_materialization_rebase_changes_only_exact_path_prefix_values(
    tmp_path,
):
    paths = _paths(tmp_path)
    original = _materialization(
        "ss64", NODE16_HSSD / "ss64/active"
    )
    original["substring"] = (
        "prefix-"
        "/home/youngwoo/data/pixal3d/train/production/hssd/ss64"
    )
    original["sibling"] = (
        "/home/youngwoo/data/pixal3d/train/production/hssd-other"
    )

    rebased = transfer._rebase_json_paths(
        original,
        (
            (str(NODE16_HSSD), str(paths.staging_root)),
            ("/file2/youngwoo/pixal3d", str(DATA2_ROOT)),
        ),
    )

    assert rebased["stage_root"] == str(
        paths.staging_root / "ss64/active"
    )
    assert rebased["source_indexes"][0]["path"] == str(
        DATA2_ROOT / "prepared/index/HSSD/HSSD-00000.json"
    )
    assert rebased["packs"][0]["path"] == str(
        DATA2_ROOT / "prepared/packs/ss64.zip"
    )
    assert rebased["substring"] == original["substring"]
    assert rebased["sibling"] == original["sibling"]
    assert original["stage_root"] == str(NODE16_HSSD / "ss64/active")


def test_preflight_result_remap_preserves_all_validated_evidence(tmp_path):
    paths = _paths(tmp_path)
    document = _materialization(
        "ss64", paths.staging_root / "ss64/active",
        shared_root=DATA2_ROOT,
    )
    result = _stage_result(
        "ss64", paths.staging_root / "ss64/active", document
    )

    promoted = transfer._promote_stage_preflight(paths, result)
    promoted_document = json.loads(promoted.materialization_bytes)

    assert promoted.root == paths.canonical_root / "ss64/active"
    assert promoted_document["stage_root"] == str(promoted.root)
    assert promoted.asset_count == result.asset_count
    assert promoted.asset_scope_sha256 == result.asset_scope_sha256
    assert promoted.anchors_checked == result.anchors_checked
    assert promoted.validation_counts == result.validation_counts
    for field in (
        "counts",
        "stage_scope",
        "stage_scope_sha256",
        "packs",
        "tool_commits",
        "source_indexes",
    ):
        assert promoted_document[field] == document[field]


@pytest.mark.parametrize("failure", ("publisher", "validator"))
def test_promotion_rolls_canonical_back_to_staging_on_chain_error(
    tmp_path, monkeypatch, failure
):
    paths = _paths(tmp_path)
    staged_results = _write_staged_materializations(
        paths, paths.staging_root
    )
    promoted_results = {
        stage: transfer._promote_stage_preflight(paths, result)
        for stage, result in staged_results.items()
    }

    def publisher(_spec, _results, report, handoff, training_data):
        if failure == "publisher":
            raise RuntimeError("publisher failed")
        report.parent.mkdir()
        report.write_text("{}")
        handoff.write_text("{}")
        training_data.write_text("{}")
        return report, handoff, training_data

    def validator(_source, _training_data):
        if failure == "validator":
            raise RuntimeError("validator failed")
        return SimpleNamespace(sha256="f" * 64)

    monkeypatch.setattr(
        transfer.training_preflight, "publish_source_handoff", publisher
    )
    monkeypatch.setattr(
        transfer, "validate_source_training_data", validator
    )

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        transfer._promote_and_publish(
            paths, SimpleNamespace(source="HSSD"), promoted_results
        )

    assert not paths.canonical_root.exists()
    assert paths.staging_root.exists()


def test_successful_promotion_has_exact_publication_topology(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    staged_results = _write_staged_materializations(
        paths, paths.staging_root
    )
    promoted_results = {
        stage: transfer._promote_stage_preflight(paths, result)
        for stage, result in staged_results.items()
    }

    def publisher(_spec, _results, report, handoff, training_data):
        report.parent.mkdir()
        report.write_text('{"report": true}')
        handoff.write_text('{"handoff": true}')
        training_data.write_text('{"training": true}')
        return report, handoff, training_data

    monkeypatch.setattr(
        transfer.training_preflight, "publish_source_handoff", publisher
    )
    monkeypatch.setattr(
        transfer,
        "validate_source_training_data",
        lambda _source, training_data: SimpleNamespace(
            path=training_data,
            sha256=sha256(training_data.read_bytes()).hexdigest(),
        ),
    )

    validated = transfer._promote_and_publish(
        paths, SimpleNamespace(source="HSSD"), promoted_results
    )

    assert validated.path == paths.canonical_root / "training_data.json"
    assert {path.name for path in paths.canonical_root.iterdir()} == {
        *STAGES,
        "publication",
        "training_data.json",
    }
    assert {
        path.name
        for path in (paths.canonical_root / "publication").iterdir()
    } == {"report.json", "handoff.json"}
    assert not paths.staging_root.exists()


def test_transfer_runs_without_network_and_returns_rebased_evidence(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(
            100 * 1024**3, 1, 100 * 1024**3 - 1
        ),
    )
    source_inventory = transfer.TreeInventory(
        8,
        sum(
            len(
                _canonical_bytes(
                    _materialization(
                        stage, NODE16_HSSD / stage / "active"
                    )
                )
            )
            + len(stage.encode())
            for stage in STAGES
        ),
    )
    calls = []

    def runner(command):
        command = list(command)
        calls.append(command)
        if command[0] == "ssh":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "file_count": source_inventory.file_count,
                        "logical_bytes": source_inventory.logical_bytes,
                    }
                ),
            )
        if "--dry-run" not in command:
            _write_staged_materializations(paths, NODE16_HSSD)
            return subprocess.CompletedProcess(command, 0, stdout="")
        return subprocess.CompletedProcess(command, 0, stdout="")

    def preflight(_spec, stage, root, _config):
        document = json.loads((root / "materialization.json").read_bytes())
        return _stage_result(stage, root, document)

    def publisher(_spec, _results, report, handoff, training_data):
        report.parent.mkdir()
        report.write_text("{}")
        handoff.write_text("{}")
        training_data.write_text("{}")
        return report, handoff, training_data

    monkeypatch.setattr(
        transfer.training_preflight, "preflight_stage", preflight
    )
    monkeypatch.setattr(
        transfer.training_preflight, "publish_source_handoff", publisher
    )
    monkeypatch.setattr(
        transfer,
        "validate_source_training_data",
        lambda _source, training_data: SimpleNamespace(
            path=training_data,
            sha256=sha256(training_data.read_bytes()).hexdigest(),
        ),
    )

    runtime_configs = {
        stage: tmp_path / f"{stage}.node17.json" for stage in STAGES
    }
    result = transfer.transfer_and_publish_hssd(
        paths, runtime_configs, command_runner=runner
    )

    assert result.source_inventory == result.target_inventory
    assert set(result.original_materialization_sha256) == set(STAGES)
    assert set(result.canonical_materialization_sha256) == set(STAGES)
    assert all(
        result.original_materialization_sha256[stage]
        != result.canonical_materialization_sha256[stage]
        for stage in STAGES
    )
    assert result.training_data == (
        paths.canonical_root / "training_data.json"
    )
    assert result.stage_counts == dict.fromkeys(STAGES, 2)
    assert set(result.elapsed_seconds) == {
        "inventory",
        "transfer",
        "verification",
        "evidence_rebase",
        "strict_preflight",
        "promotion",
        "total",
    }
    assert not paths.staging_root.exists()
    assert paths.canonical_root.exists()
    assert all(command[0] in {"ssh", "rsync"} for command in calls)
