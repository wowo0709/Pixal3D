import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from data_toolkit.pipeline.training_manifest import (
    STAGES,
    publish_combined_training_data,
    resolve_training_data,
)
from scripts.preflight_multisource_training import (
    CONFIGS,
    preflight_multisource_stage,
)
from tests.multiview.test_training_manifest import _write_source


class _TrainingFixture:
    def __init__(self, training_data: Path):
        self.training_data = training_data


def _write_metadata(
    root: Path, fields: list[str], assets: tuple[str, ...]
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["sha256", *fields, "aesthetic_score"]
        )
        writer.writeheader()
        for asset in assets:
            writer.writerow(
                {
                    "sha256": asset,
                    **{field: "True" for field in fields},
                    "aesthetic_score": "5.0",
                }
            )


def _write_render(root: Path, asset: str) -> None:
    render = root / "renders_cond" / asset
    render.mkdir(parents=True)
    frames = []
    for index in range(8):
        Image.new("RGBA", (4, 4), (index, 0, 0, 255)).save(
            render / f"{index:03d}.png"
        )
        transform = np.eye(4, dtype=np.float32)
        transform[2, 3] = 2.0
        frames.append(
            {
                "file_path": f"{index:03d}.png",
                "camera_angle_x": 0.7,
                "transform_matrix": transform.tolist(),
            }
        )
    (render / "transforms.json").write_text(json.dumps({"frames": frames}))


def _write_latent(root: Path, asset: str, kind: str) -> None:
    target = root / asset
    target.mkdir(parents=True, exist_ok=True)
    for anchor in (0, 1):
        path = target / f"view{anchor:02d}.npz"
        if kind == "ss":
            np.savez(
                path, z=np.ones((8, 16, 16, 16), dtype=np.float32)
            )
        else:
            np.savez(
                path,
                coords=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
                feats=np.ones((2, 32), dtype=np.float32),
            )
        (target / f"view{anchor:02d}_scale.json").write_text(
            json.dumps({"total_scale": 1.0})
        )


def _populate_stage(root: Path, stage: str, assets: tuple[str, ...]) -> None:
    _write_metadata(root, [], assets)
    _write_metadata(root / "renders_cond", ["cond_rendered"], assets)
    for asset in assets:
        _write_render(root, asset)
    if stage == "ss64":
        latent_roots = (
            (
                root
                / "ss_latents/ss_enc_conv3d_16l8_fp16_64_view",
                "ss",
                [
                    "ss_latent_view_scale00_encoded",
                    "ss_latent_view_scale01_encoded",
                ],
            ),
        )
    elif stage == "shape512":
        latent_roots = (
            (
                root
                / "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
                "shape",
                [
                    "shape_latent_view00_encoded",
                    "shape_latent_view01_encoded",
                ],
            ),
        )
    elif stage == "shape1024":
        latent_roots = (
            (
                root
                / "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
                "shape",
                [
                    "shape_latent_view00_encoded",
                    "shape_latent_view01_encoded",
                ],
            ),
        )
    else:
        latent_roots = (
            (
                root
                / "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
                "shape",
                [
                    "shape_latent_view00_encoded",
                    "shape_latent_view01_encoded",
                ],
            ),
            (
                root
                / "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix",
                "pbr",
                [
                    "pbr_latent_view00_encoded",
                    "pbr_latent_view01_encoded",
                ],
            ),
        )
    for latent_root, kind, fields in latent_roots:
        _write_metadata(latent_root, fields, assets)
        for asset in assets:
            _write_latent(latent_root, asset, kind)


@pytest.fixture
def two_source_fixture(tmp_path):
    source_paths = {
        "ABO": _write_source(tmp_path, "ABO", schema_version=1, count=1),
        "3D-FUTURE": _write_source(
            tmp_path, "3D-FUTURE", schema_version=2, count=2
        ),
    }
    for training_data in source_paths.values():
        source_manifest = json.loads(training_data.read_text())
        for stage in STAGES:
            root = Path(source_manifest["stages"][stage]["root"])
            materialization = json.loads(
                (root / "materialization.json").read_text()
            )
            _populate_stage(
                root, stage, tuple(materialization["stage_scope"])
            )
    combined = tmp_path / "combined" / "training_data.json"
    publish_combined_training_data(source_paths, combined)
    return _TrainingFixture(combined)


@pytest.mark.parametrize(
    "stage", ("ss64", "shape512", "shape1024", "pbr1024")
)
def test_combined_preflight_matches_disjoint_union(
    two_source_fixture, stage, monkeypatch
):
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: pytest.fail("combined preflight must not enumerate CUDA"),
    )
    result = preflight_multisource_stage(
        two_source_fixture.training_data,
        stage,
        CONFIGS[stage],
    )
    assert result["source_counts"] == {"ABO": 1, "3D-FUTURE": 2}
    assert result["total_count"] == 3
    assert result["sampling"] == "proportional-unweighted-concatenation"
    assert result["collated_sources"] == ["ABO", "3D-FUTURE"]


@pytest.fixture
def three_source_fixture(tmp_path):
    source_paths = {
        source: _write_source(
            tmp_path, source, schema_version=schema_version, count=2
        )
        for source, schema_version in (
            ("ABO", 1),
            ("3D-FUTURE", 2),
            ("HSSD", 2),
        )
    }
    for training_data in source_paths.values():
        source_manifest = json.loads(training_data.read_text())
        root = Path(source_manifest["stages"]["ss64"]["root"])
        materialization = json.loads(
            (root / "materialization.json").read_text()
        )
        _populate_stage(
            root, "ss64", tuple(materialization["stage_scope"])
        )
    combined = tmp_path / "combined" / "training_data.json"
    publish_combined_training_data(source_paths, combined)
    return _TrainingFixture(combined)


def test_three_source_preflight_loads_boundaries_and_collates(
    three_source_fixture, monkeypatch
):
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: pytest.fail("combined preflight must not enumerate CUDA"),
    )
    result = preflight_multisource_stage(
        three_source_fixture.training_data,
        "ss64",
        CONFIGS["ss64"],
    )
    assert result["collated_sources"] == [
        "ABO",
        "3D-FUTURE",
        "HSSD",
    ]
    assert result["boundary_instances_checked"] == 6


@pytest.fixture
def one_source_hssd_fixture(tmp_path):
    training_data = _write_source(
        tmp_path, "HSSD", schema_version=2, count=2
    )
    source_manifest = json.loads(training_data.read_text())
    root = Path(source_manifest["stages"]["ss64"]["root"])
    materialization = json.loads(
        (root / "materialization.json").read_text()
    )
    _populate_stage(
        root, "ss64", tuple(materialization["stage_scope"])
    )
    return training_data


def test_one_source_hssd_preflight_loads_boundaries_and_collates(
    one_source_hssd_fixture, monkeypatch
):
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: pytest.fail("combined preflight must not enumerate CUDA"),
    )
    result = preflight_multisource_stage(
        one_source_hssd_fixture,
        "ss64",
        CONFIGS["ss64"],
    )
    assert result["collated_sources"] == ["HSSD"]
    assert result["boundary_instances_checked"] == 2


def _fake_dataset(
    resolved, *, instances=None, load_error=None, collate_error=None
):
    class FakeDataset:
        def __init__(self):
            self.instances = (
                instances
                if instances is not None
                else [
                    (
                        resolved.data_dir[source],
                        asset,
                        source,
                    )
                    for source, scope in resolved.source_scopes.items()
                    for asset in scope
                ]
            )

        def get_instance(self, _root, asset):
            if load_error is not None:
                raise load_error
            return {"asset": asset}

        def collate_fn(self, batch):
            if collate_error is not None:
                raise collate_error
            return {"assets": [sample["asset"] for sample in batch]}

    return FakeDataset()


def test_preflight_rejects_available_cuda_before_manifest_resolution(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CPU-only preflight"):
        preflight_multisource_stage(
            tmp_path / "missing-training-data.json",
            "ss64",
            CONFIGS["ss64"],
        )


def test_preflight_rejects_cuda_initialized_during_loader_validation(
    two_source_fixture, monkeypatch
):
    import scripts.preflight_multisource_training as preflight

    resolved = resolve_training_data(
        two_source_fixture.training_data, "ss64"
    )
    initialized = iter((False, True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "is_initialized", lambda: next(initialized)
    )
    monkeypatch.setattr(
        preflight,
        "_construct_configured_dataset",
        lambda _resolved, _config: _fake_dataset(resolved),
    )
    with pytest.raises(RuntimeError, match="CPU-only preflight"):
        preflight_multisource_stage(
            two_source_fixture.training_data,
            "ss64",
            CONFIGS["ss64"],
        )


def test_preflight_rejects_unknown_resolved_source(
    two_source_fixture, monkeypatch
):
    import scripts.preflight_multisource_training as preflight

    resolved = resolve_training_data(
        two_source_fixture.training_data, "ss64"
    )
    unknown = replace(
        resolved,
        data_dir={"OTHER": resolved.data_dir["ABO"]},
        source_counts={"OTHER": 1},
        total_count=1,
        source_scopes={"OTHER": resolved.source_scopes["ABO"][:1]},
    )
    monkeypatch.setattr(
        preflight,
        "resolve_training_data",
        lambda _training_data, _stage: unknown,
    )
    with pytest.raises(ValueError, match="unknown resolved source"):
        preflight_multisource_stage(
            two_source_fixture.training_data,
            "ss64",
            CONFIGS["ss64"],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("omitted", "omitted"),
        ("unexpected_source", "unexpected source"),
        ("duplicate", "duplicate source/SHA"),
        ("permuted", "canonical order"),
    ],
)
def test_combined_preflight_rejects_nonexact_dataset_instances(
    two_source_fixture, monkeypatch, mutation, message
):
    import scripts.preflight_multisource_training as preflight

    resolved = resolve_training_data(
        two_source_fixture.training_data, "ss64"
    )
    exact = [
        (resolved.data_dir[source], asset, source)
        for source, scope in resolved.source_scopes.items()
        for asset in scope
    ]
    if mutation == "omitted":
        instances = exact[:-1]
    elif mutation == "unexpected_source":
        instances = [*exact[:-1], (exact[-1][0], exact[-1][1], "OTHER")]
    elif mutation == "permuted":
        instances = list(reversed(exact))
    else:
        instances = [*exact, exact[0]]
    monkeypatch.setattr(
        preflight,
        "_construct_configured_dataset",
        lambda _resolved, _config: _fake_dataset(
            resolved, instances=instances
        ),
    )
    with pytest.raises(ValueError, match=message):
        preflight_multisource_stage(
            two_source_fixture.training_data, "ss64", CONFIGS["ss64"]
        )


def test_combined_preflight_rejects_real_loader_filtering(
    two_source_fixture,
):
    resolved = resolve_training_data(
        two_source_fixture.training_data, "ss64"
    )
    render_metadata = (
        Path(resolved.data_dir["3D-FUTURE"]["render_cond"])
        / "metadata.csv"
    )
    rows = list(csv.DictReader(render_metadata.open()))
    rows[-1]["cond_rendered"] = ""
    with render_metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="loader filtering"):
        preflight_multisource_stage(
            two_source_fixture.training_data, "ss64", CONFIGS["ss64"]
        )


def test_combined_preflight_rejects_cross_source_collate_failure(
    two_source_fixture, monkeypatch
):
    import scripts.preflight_multisource_training as preflight

    resolved = resolve_training_data(
        two_source_fixture.training_data, "shape512"
    )
    monkeypatch.setattr(
        preflight,
        "_construct_configured_dataset",
        lambda _resolved, _config: _fake_dataset(
            resolved, collate_error=RuntimeError("synthetic collate failure")
        ),
    )
    with pytest.raises(RuntimeError, match="cross-source collate"):
        preflight_multisource_stage(
            two_source_fixture.training_data,
            "shape512",
            CONFIGS["shape512"],
        )


def test_combined_preflight_wraps_plain_loader_error_with_anchor_context(
    two_source_fixture, monkeypatch
):
    import scripts.preflight_multisource_training as preflight

    resolved = resolve_training_data(
        two_source_fixture.training_data, "ss64"
    )
    first_asset = resolved.source_scopes["ABO"][0]
    monkeypatch.setattr(
        preflight,
        "_construct_configured_dataset",
        lambda _resolved, _config: _fake_dataset(
            resolved, load_error=LookupError("plain loader failure")
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            rf"source=ABO stage=ss64 asset={first_asset} "
            r"anchor=view00"
        ),
    ) as captured:
        preflight_multisource_stage(
            two_source_fixture.training_data,
            "ss64",
            CONFIGS["ss64"],
        )

    assert isinstance(captured.value.__cause__, LookupError)
