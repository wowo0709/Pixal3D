import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data_toolkit.pipeline import training_eligibility
from data_toolkit.pipeline.training_eligibility import policy_evidence
from scripts import preflight_multiview_production as preflight
from scripts.preflight_multiview_production import validate_stage_structure


ASSET = "a" * 64


def component_root(stage: str, root: Path, component: str) -> Path:
    names = {
        "ss": "ss_latents/ss_enc_conv3d_16l8_fp16_64_view",
        "shape512": "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
        "shape1024": "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
        "pbr": "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix",
    }
    return root / names[component]


def write_metadata(root: Path, fieldnames: list[str], asset: str = ASSET) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sha256", *fieldnames])
        writer.writeheader()
        writer.writerow({"sha256": asset, **{field: "True" for field in fieldnames}})


def write_render(root: Path, asset: str = ASSET) -> None:
    render = root / "renders_cond" / asset
    render.mkdir(parents=True)
    frames = []
    for index in range(8):
        Image.new("RGBA", (512, 512), (index, 0, 0, 255)).save(render / f"{index:03d}.png")
        transform = np.eye(4, dtype=np.float32)
        transform[2, 3] = 2.0
        frames.append({
            "file_path": f"{index:03d}.png",
            "camera_angle_x": 0.7,
            "transform_matrix": transform.tolist(),
        })
    (render / "transforms.json").write_text(json.dumps({"frames": frames}))


def write_scale(path: Path, value: object = 1.0) -> None:
    path.write_text(json.dumps({"total_scale": value}))


def write_latent(root: Path, asset: str, kind: str, anchor: int) -> None:
    target = root / asset
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"view{anchor:02d}.npz"
    if kind == "ss":
        np.savez(path, z=np.ones((8, 16, 16, 16), dtype=np.float32))
    else:
        np.savez(
            path,
            coords=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
            feats=np.ones((2, 32), dtype=np.float32),
        )
    write_scale(target / f"view{anchor:02d}_scale.json")


def make_stage(tmp_path: Path, stage: str = "ss64") -> Path:
    root = tmp_path / stage / "active"
    write_metadata(root / "renders_cond", ["cond_rendered"])
    write_render(root)
    if stage == "ss64":
        latent_roots = [(component_root(stage, root, "ss"), "ss", ["ss_latent_view_scale00_encoded", "ss_latent_view_scale01_encoded"])]
    elif stage in ("shape512", "shape1024"):
        latent_roots = [(component_root(stage, root, stage), "shape", ["shape_latent_view00_encoded", "shape_latent_view01_encoded"])]
    else:
        latent_roots = [
            (component_root(stage, root, "shape1024"), "shape", ["shape_latent_view00_encoded", "shape_latent_view01_encoded"]),
            (component_root(stage, root, "pbr"), "pbr", ["pbr_latent_view00_encoded", "pbr_latent_view01_encoded"]),
        ]
    for latent_root, kind, fields in latent_roots:
        write_metadata(latent_root, fields)
        for anchor in (0, 1):
            write_latent(latent_root, ASSET, kind, anchor)
    return root


def assert_context(error: pytest.ExceptionInfo[ValueError], stage: str, anchor: str | None = None) -> None:
    text = str(error.value)
    assert "source=ABO" in text
    assert f"stage={stage}" in text
    assert f"asset={ASSET}" in text
    if anchor is not None:
        assert f"anchor={anchor}" in text


@pytest.mark.parametrize("stage, expected", [
    ("ss64", {"assets": 1, "renders": 8, "latents": 2, "scales": 2}),
    ("shape512", {"assets": 1, "renders": 8, "latents": 2, "scales": 2}),
    ("shape1024", {"assets": 1, "renders": 8, "latents": 2, "scales": 2}),
    ("pbr1024", {"assets": 1, "renders": 8, "latents": 4, "scales": 4}),
])
def test_structure_accepts_complete_literal_stage_fixture(tmp_path, stage, expected):
    assert validate_stage_structure(stage, make_stage(tmp_path, stage), [ASSET]) == expected


@pytest.mark.parametrize("stage, field", [
    ("ss64", "cond_rendered"),
    ("ss64", "ss_latent_view_scale00_encoded"),
    ("shape512", "shape_latent_view01_encoded"),
    ("pbr1024", "pbr_latent_view00_encoded"),
])
def test_structure_rejects_metadata_missing_required_column(tmp_path, stage, field):
    root = make_stage(tmp_path, stage)
    if field == "cond_rendered":
        metadata = root / "renders_cond" / "metadata.csv"
    elif field.startswith("ss_"):
        metadata = component_root(stage, root, "ss") / "metadata.csv"
    elif field.startswith("shape_"):
        metadata = component_root(stage, root, "shape512") / "metadata.csv"
    else:
        metadata = component_root(stage, root, "pbr") / "metadata.csv"
    rows = list(csv.DictReader(metadata.open()))
    fields = [name for name in rows[0] if name not in (field, None)]
    with metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{name: row[name] for name in fields} for row in rows])
    with pytest.raises(ValueError) as caught:
        validate_stage_structure(stage, root, [ASSET])
    assert_context(caught, stage)


def test_structure_requires_literal_true_metadata_flags(tmp_path):
    root = make_stage(tmp_path)
    path = root / "renders_cond" / "metadata.csv"
    path.write_text(f"sha256,cond_rendered\n{ASSET},False\n")
    with pytest.raises(ValueError) as caught:
        validate_stage_structure("ss64", root, [ASSET])
    assert_context(caught, "ss64")


def test_structure_rejects_component_scope_mismatch(tmp_path):
    root = make_stage(tmp_path)
    path = component_root("ss64", root, "ss") / "metadata.csv"
    path.write_text("sha256,ss_latent_view_scale00_encoded,ss_latent_view_scale01_encoded\n" + "b" * 64 + ",True,True\n")
    with pytest.raises(ValueError) as caught:
        validate_stage_structure("ss64", root, [ASSET])
    assert_context(caught, "ss64")


def test_structure_rejects_unlisted_component_asset_directory(tmp_path):
    root = make_stage(tmp_path)
    (root / "renders_cond" / ("b" * 64)).mkdir()
    with pytest.raises(ValueError) as caught:
        validate_stage_structure("ss64", root, [ASSET])
    assert "source=ABO" in str(caught.value)
    assert "stage=ss64" in str(caught.value)
    assert "asset=" + "b" * 64 in str(caught.value)


@pytest.mark.parametrize("mutation", ["missing", "extra", "rgb", "small", "unsafe", "angle", "distance", "shape", "nonfinite", "singular"])
def test_structure_rejects_invalid_render_or_camera_contract(tmp_path, mutation):
    root = make_stage(tmp_path)
    render = root / "renders_cond" / ASSET
    manifest = json.loads((render / "transforms.json").read_text())
    if mutation == "missing":
        (render / "007.png").unlink()
    elif mutation == "extra":
        Image.new("RGBA", (512, 512)).save(render / "008.png")
    elif mutation == "rgb":
        Image.new("RGB", (512, 512)).save(render / "000.png")
    elif mutation == "small":
        Image.new("RGBA", (4, 4)).save(render / "000.png")
    elif mutation == "unsafe":
        manifest["frames"][0]["file_path"] = "../escape.png"
    elif mutation == "angle":
        manifest["frames"][0]["camera_angle_x"] = 0.0
    elif mutation == "distance":
        manifest["frames"][0]["transform_matrix"][2][3] = 0.0
    elif mutation == "shape":
        manifest["frames"][0]["transform_matrix"] = [[1.0]]
    elif mutation == "nonfinite":
        manifest["frames"][0]["transform_matrix"][0][0] = float("nan")
    else:
        manifest["frames"][0]["transform_matrix"] = np.zeros((4, 4)).tolist()
    (render / "transforms.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError) as caught:
        validate_stage_structure("ss64", root, [ASSET])
    assert_context(caught, "ss64")


def test_structure_rejects_in_directory_render_png_symlink(tmp_path):
    root = make_stage(tmp_path)
    render = root / "renders_cond" / ASSET
    (render / "000.png").unlink()
    (render / "000.png").symlink_to("001.png")
    with pytest.raises(ValueError) as caught:
        validate_stage_structure("ss64", root, [ASSET])
    assert_context(caught, "ss64")


@pytest.mark.parametrize("mutation", ["missing_npz", "missing_scale", "scale_missing", "scale_nan", "scale_zero", "scale_underflow", "bad_key", "bad_dtype", "bad_finite", "bad_shape"])
def test_structure_rejects_ss_latent_or_scale_mutations(tmp_path, mutation):
    root = make_stage(tmp_path)
    latent = component_root("ss64", root, "ss") / ASSET
    npz = latent / "view00.npz"
    scale = latent / "view00_scale.json"
    if mutation == "missing_npz":
        npz.unlink()
    elif mutation == "missing_scale":
        scale.unlink()
    elif mutation == "scale_missing":
        scale.write_text("{}")
    elif mutation == "scale_nan":
        write_scale(scale, float("nan"))
    elif mutation == "scale_zero":
        write_scale(scale, 0.0)
    elif mutation == "scale_underflow":
        write_scale(scale, 1e-46)
    elif mutation == "bad_key":
        np.savez(npz, other=np.ones((8, 16, 16, 16), dtype=np.float32))
    elif mutation == "bad_dtype":
        np.savez(npz, z=np.ones((8, 16, 16, 16), dtype=np.complex64))
    elif mutation == "bad_finite":
        z = np.ones((8, 16, 16, 16), dtype=np.float32); z[0, 0, 0, 0] = np.nan; np.savez(npz, z=z)
    else:
        np.savez(npz, z=np.ones((8, 16, 16), dtype=np.float32))
    with pytest.raises(ValueError) as caught:
        validate_stage_structure("ss64", root, [ASSET])
    assert_context(caught, "ss64", "view00")


@pytest.mark.parametrize("value", [True, "1.0", [1.0], [1.0, 2.0]])
@pytest.mark.parametrize("stage, component", [("ss64", "ss"), ("pbr1024", "shape1024"), ("pbr1024", "pbr")])
def test_structure_rejects_non_numeric_scalar_total_scale(tmp_path, stage, component, value):
    root = make_stage(tmp_path, stage)
    scale = component_root(stage, root, component) / ASSET / "view00_scale.json"
    write_scale(scale, value)
    with pytest.raises(ValueError) as caught:
        validate_stage_structure(stage, root, [ASSET])
    assert_context(caught, stage, "view00")


@pytest.mark.parametrize("stage, component, mutation", [
    ("shape512", "shape512", "keys"), ("shape512", "shape512", "coord_rank"),
    ("shape512", "shape512", "feature_rank"), ("shape512", "shape512", "rows"),
    ("shape512", "shape512", "width"), ("shape512", "shape512", "finite"),
    ("shape512", "shape512", "duplicate"), ("shape512", "shape512", "fractional"),
    ("shape512", "shape512", "bounds"), ("shape512", "shape512", "tokens"),
    ("pbr1024", "pbr", "keys"),
])
def test_structure_rejects_sparse_latent_contract_mutations(tmp_path, stage, component, mutation):
    root = make_stage(tmp_path, stage)
    path = component_root(stage, root, component) / ASSET / "view00.npz"
    coords = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    feats = np.ones((2, 32), dtype=np.float32)
    if mutation == "keys":
        np.savez(path, coords=coords)
    elif mutation == "coord_rank":
        np.savez(path, coords=np.zeros((2, 3, 1), dtype=np.int32), feats=feats)
    elif mutation == "feature_rank":
        np.savez(path, coords=coords, feats=np.ones((2, 32, 1), dtype=np.float32))
    elif mutation == "rows":
        np.savez(path, coords=coords, feats=np.ones((1, 32), dtype=np.float32))
    elif mutation == "width":
        np.savez(path, coords=coords, feats=np.ones((2, 31), dtype=np.float32))
    elif mutation == "finite":
        feats[0, 0] = np.nan; np.savez(path, coords=coords, feats=feats)
    elif mutation == "duplicate":
        np.savez(path, coords=np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32), feats=feats)
    elif mutation == "fractional":
        np.savez(path, coords=np.array([[0.5, 1, 2], [3, 4, 5]], dtype=np.float32), feats=feats)
    elif mutation == "bounds":
        np.savez(path, coords=np.array([[32, 1, 2], [3, 4, 5]], dtype=np.int32), feats=feats)
    else:
        np.savez(path, coords=np.zeros((8193, 3), dtype=np.int32), feats=np.ones((8193, 32), dtype=np.float32))
    with pytest.raises(ValueError) as caught:
        validate_stage_structure(stage, root, [ASSET])
    assert_context(caught, stage, "view00")


@pytest.mark.parametrize("mutation", ["coords", "scale"])
def test_structure_requires_pbr_shape_alignment_after_float32_conversion(tmp_path, mutation):
    root = make_stage(tmp_path, "pbr1024")
    shape = component_root("pbr1024", root, "shape1024") / ASSET
    pbr = component_root("pbr1024", root, "pbr") / ASSET
    if mutation == "coords":
        np.savez(pbr / "view00.npz", coords=np.array([[1, 1, 2], [3, 4, 5]], dtype=np.int32), feats=np.ones((2, 32), dtype=np.float32))
    else:
        write_scale(shape / "view00_scale.json", 1.0)
        write_scale(pbr / "view00_scale.json", 1.1)
    with pytest.raises(ValueError) as caught:
        validate_stage_structure("pbr1024", root, [ASSET])
    assert_context(caught, "pbr1024", "view00")


def test_structure_accepts_pbr_shape_scale_at_shared_absolute_tolerance(tmp_path):
    """Changing the shared tolerance must reject the approved float32 boundary."""
    root = make_stage(tmp_path, "pbr1024")
    shape = component_root("pbr1024", root, "shape1024") / ASSET
    pbr = component_root("pbr1024", root, "pbr") / ASSET
    write_scale(shape / "view00_scale.json", 0.5)
    write_scale(pbr / "view00_scale.json", 0.5000002)
    assert validate_stage_structure("pbr1024", root, [ASSET])["assets"] == 1


def test_structure_rejects_pbr_shape_scale_above_shared_absolute_tolerance(tmp_path):
    """Permitting a scale larger than the policy tolerance must stop preflight."""
    root = make_stage(tmp_path, "pbr1024")
    shape = component_root("pbr1024", root, "shape1024") / ASSET
    pbr = component_root("pbr1024", root, "pbr") / ASSET
    write_scale(shape / "view00_scale.json", 0.5)
    write_scale(pbr / "view00_scale.json", 0.5000003)
    with pytest.raises(ValueError, match="float32 total_scale"):
        validate_stage_structure("pbr1024", root, [ASSET])


def test_preflight_uses_shared_training_eligibility_policy_constants():
    """A local copy of the fine-tuning limits or tolerances can silently drift."""
    assert preflight.TOKEN_LIMITS is training_eligibility.TOKEN_LIMITS
    assert preflight.SCALE_RTOL == training_eligibility.SCALE_RTOL
    assert preflight.SCALE_ATOL == training_eligibility.SCALE_ATOL


def write_loader_config(tmp_path: Path, stage: str) -> Path:
    args = {
        "min_aesthetic_score": 4.5,
        "image_size": 4,
        "num_views": 2,
        "condition_num_views": 8,
        "min_condition_views": 2,
        "max_condition_views": 6,
        "skip_aesthetic_score_datasets": ["texverse"],
    }
    names = {
        "ss64": "MultiViewImageConditionedSparseStructureLatentView",
        "shape512": "MultiViewImageConditionedSLatShapeView",
        "shape1024": "MultiViewImageConditionedSLatShapeView",
        "pbr1024": "MultiViewImageConditionedSLatPbrView",
    }
    if stage == "ss64":
        args["pretrained_ss_dec"] = "not-loaded"
    elif stage == "shape512":
        args.update({"resolution": 512, "max_tokens": 8192, "pretrained_slat_dec": "not-loaded"})
    elif stage == "shape1024":
        args.update({"resolution": 1024, "max_tokens": 32768, "pretrained_slat_dec": "not-loaded"})
    else:
        args.update({
            "resolution": 1024, "max_tokens": 32768, "full_pbr": False,
            "pretrained_pbr_slat_dec": "not-loaded", "pretrained_shape_slat_dec": "not-loaded",
        })
    path = tmp_path / f"{stage}.json"
    path.write_text(json.dumps({"dataset": {"name": names[stage], "args": args}}))
    return path


@pytest.mark.parametrize("stage", ["ss64", "shape512", "pbr1024"])
def test_direct_loader_checks_every_anchor_without_dataset_retry(tmp_path, monkeypatch, stage):
    root = make_stage(tmp_path, stage)
    config = write_loader_config(tmp_path, stage)
    from pixal3d.datasets.components import StandardDatasetBase
    original_get_device_name = torch.cuda.get_device_name
    assert not torch.cuda.is_initialized()

    def no_retry(self, index):
        raise AssertionError("preflight must never call dataset[index]")

    monkeypatch.setattr(StandardDatasetBase, "__getitem__", no_retry)
    assert preflight.validate_direct_loader(stage, root, [ASSET], config) == 2
    assert torch.cuda.get_device_name is original_get_device_name
    assert not torch.cuda.is_initialized()


def test_direct_script_loader_imports_pixal3d_from_outside_repository(tmp_path):
    """Direct script execution must locate the repository package before real loading."""
    root = make_stage(tmp_path)
    config = write_loader_config(tmp_path, "ss64")
    script = Path(preflight.__file__).resolve()
    outside = tmp_path / "outside"
    outside.mkdir()
    worker = """
import runpy
import sys
from pathlib import Path

script = Path(sys.argv[1]).resolve()
repository = script.parent.parent
sys.path[:] = [
    str(script.parent),
    *[
        entry
        for entry in sys.path
        if entry and Path(entry).resolve() not in (repository, script.parent)
    ],
]
subject = runpy.run_path(str(script), run_name="direct_script_regression")
checked = subject["validate_direct_loader"](
    "ss64", Path(sys.argv[2]), [sys.argv[3]], Path(sys.argv[4])
)
print(checked)
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", worker, str(script), str(root), ASSET, str(config)],
        cwd=outside,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("2")


def test_direct_loader_surfaces_damaged_anchor_instead_of_retrying_another_sample(tmp_path, monkeypatch):
    root = make_stage(tmp_path)
    (component_root("ss64", root, "ss") / ASSET / "view01.npz").unlink()
    config = write_loader_config(tmp_path, "ss64")
    from pixal3d.datasets.components import StandardDatasetBase
    monkeypatch.setattr(StandardDatasetBase, "__getitem__", lambda self, index: pytest.fail("dataset retry"))
    with pytest.raises(RuntimeError, match=r"source=ABO.*asset=" + ASSET + r".*anchor=view01"):
        preflight.validate_direct_loader("ss64", root, [ASSET], config)


def test_preflight_stage_reads_materialization_scope_and_returns_frozen_result(tmp_path, monkeypatch):
    root = make_stage(tmp_path)
    results, materializations = handoff_inputs(tmp_path)
    evidence = materializations["ss64"]
    evidence["stage_root"] = str(root.resolve())
    results["ss64"] = replace(results["ss64"], root=root)
    results = with_evidence_digests(results, materializations)
    (root / "materialization.json").write_text(json.dumps(evidence))
    monkeypatch.setattr(preflight, "validate_stage_structure", lambda _stage, _root, assets: {
        "assets": len(assets), "renders": len(assets) * 8, "latents": len(assets) * 2, "scales": len(assets) * 2,
    })
    monkeypatch.setattr(preflight, "validate_direct_loader", lambda _stage, _root, assets, _config: len(assets) * 2)
    result = preflight.preflight_stage("ss64", root, write_loader_config(tmp_path, "ss64"))
    assert result.stage == "ss64"
    assert result.root == root
    assert result.asset_count == HANDOFF_STAGE_COUNTS["ss64"]
    assert result.asset_scope_sha256 == evidence["stage_scope_sha256"]
    assert result.anchors_checked == HANDOFF_STAGE_COUNTS["ss64"] * 2
    assert result.validation_counts == {"assets": 3660, "renders": 29280, "latents": 7320, "scales": 7320}
    with pytest.raises((AttributeError, TypeError)):
        result.stage = "changed"


@pytest.mark.parametrize("mutation", ["missing", "altered", "mismatched"])
def test_preflight_rejects_missing_altered_or_mismatched_materialization_root(tmp_path, mutation):
    root = make_stage(tmp_path)
    digest = __import__("hashlib").sha256(ASSET.encode()).hexdigest()
    evidence = {
        "schema_version": 1, "created_at": "2026-07-25T00:00:00Z", "source": "ABO", "shard_id": "ABO-00000",
        "source_index": {"path": "/synthetic/index.json", "sha256": "i" * 64}, "index_sha256": "i" * 64,
        "acceptance_mode": "valid_subset_user_waiver", "original_90_percent_gate_passed": False,
        "counts": {"frozen": 4485, "global_quarantine": 825, "shape512_family_exclusions": 29, "stages": {"ss64": 3660, "shape512": 3631, "shape1024": 3660, "pbr1024": 3660}},
        "stage": "ss64", "asset_count": 1, "stage_scope": [ASSET],
        "stage_scope_sha256": digest, "stage_root": str(root.resolve()),
    }
    if mutation == "missing":
        evidence.pop("stage_root")
    elif mutation == "altered":
        evidence["stage_root"] = str(root.parent / root.name / ".." / root.name)
    else:
        evidence["stage_root"] = str((tmp_path / "other" / "active").resolve())
    (root / "materialization.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="root"):
        preflight.preflight_stage("ss64", root, write_loader_config(tmp_path, "ss64"))


HANDOFF_CANDIDATE_STAGE_COUNTS = {
    "ss64": 3660,
    "shape512": 3631,
    "shape1024": 3660,
    "pbr1024": 3660,
}
HANDOFF_TRAINING_EXCLUSION_COUNTS = {
    "ss64": 0,
    "shape512": 3,
    "shape1024": 26,
    "pbr1024": 62,
}
HANDOFF_STAGE_COUNTS = {
    "ss64": 3660,
    "shape512": 3628,
    "shape1024": 3634,
    "pbr1024": 3598,
}
HANDOFF_COUNTS = {
    "frozen": 4485,
    "global_quarantine": 825,
    "shape512_family_exclusions": 29,
    "candidate_stages": HANDOFF_CANDIDATE_STAGE_COUNTS,
    "training_exclusions": HANDOFF_TRAINING_EXCLUSION_COUNTS,
    "stages": HANDOFF_STAGE_COUNTS,
}


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def handoff_inputs(tmp_path: Path, index_sha256: str = "i" * 64, index_path: Path | None = None) -> tuple[dict[str, preflight.StagePreflight], dict[str, dict[str, object]]]:
    results = {}
    materializations = {}
    for stage, asset_count in HANDOFF_STAGE_COUNTS.items():
        root = tmp_path / "isolated" / stage / "active"
        scope = [f"{stage}-{index:05d}" for index in range(asset_count)]
        scope_digest = hashlib.sha256("\n".join(scope).encode()).hexdigest()
        excluded_count = HANDOFF_TRAINING_EXCLUSION_COUNTS[stage]
        candidate_scope = sorted(list(scope) + [f"{stage}-excluded-{index:05d}" for index in range(excluded_count)])
        exclusions = [
            {"asset": asset, "reasons": [
                "shape_tokens_view00_exceed_8192" if stage == "shape512"
                else f"shape_tokens_view00_exceed_{training_eligibility.TOKEN_LIMITS[stage]}"
            ]}
            for asset in sorted(set(candidate_scope) - set(scope))
        ] if excluded_count else []
        candidate_digest = hashlib.sha256("\n".join(candidate_scope).encode()).hexdigest()
        results[stage] = preflight.StagePreflight(
            stage=stage,
            root=root,
            asset_count=asset_count,
            asset_scope_sha256=scope_digest,
            anchors_checked=asset_count * 2,
            validation_counts={"assets": asset_count, "renders": asset_count * 8},
            materialization_sha256="0" * 64,
        )
        materializations[stage] = {
            "schema_version": 1,
            "created_at": "2026-07-25T00:00:00Z",
            "source": "ABO",
            "shard_id": "ABO-00000",
            "source_index": {"path": str((index_path or tmp_path / "index.json").resolve()), "sha256": index_sha256},
            "acceptance_mode": "valid_subset_user_waiver",
            "original_90_percent_gate_passed": False,
            "counts": HANDOFF_COUNTS,
            "stage": stage,
            "stage_root": str(root),
            "candidate_asset_count": len(candidate_scope),
            "candidate_stage_scope": candidate_scope,
            "candidate_stage_scope_sha256": candidate_digest,
            "asset_count": asset_count,
            "stage_scope": scope,
            "stage_scope_sha256": scope_digest,
            "training_exclusion_count": excluded_count,
            "training_exclusions": exclusions,
            "training_exclusion_reason_counts": {
                reason: sum(reason in exclusion["reasons"] for exclusion in exclusions)
                for reason in sorted({reason for exclusion in exclusions for reason in exclusion["reasons"]})
            },
            "eligibility_policy": policy_evidence(),
            "index_sha256": index_sha256,
            "tool_commits": [f"{stage}-tool"],
            "packs": [{"tool_commit": f"{stage}-pack-tool"}],
        }
    return with_evidence_digests(results, materializations), materializations


def with_evidence_digests(results, materializations):
    return {
        stage: replace(result, materialization_sha256=hashlib.sha256(
            canonical_json_bytes(materializations[stage])
        ).hexdigest())
        for stage, result in results.items()
    }


@pytest.mark.parametrize("mutation", ["missing", "reordered", "duplicated", "scope_difference", "policy", "candidate_count", "reason"])
def test_build_report_rejects_any_eligibility_contract_mutation(tmp_path, mutation):
    """Publication must be blocked even when only Task 1 eligibility evidence changes."""
    results, materializations = handoff_inputs(tmp_path)
    evidence = materializations["shape512"]
    if mutation == "missing":
        evidence.pop("training_exclusions")
    elif mutation == "reordered":
        evidence["training_exclusions"] = list(reversed(evidence["training_exclusions"]))
    elif mutation == "duplicated":
        evidence["training_exclusions"][1]["asset"] = evidence["training_exclusions"][0]["asset"]
    elif mutation == "scope_difference":
        evidence["training_exclusions"][0]["asset"] = evidence["stage_scope"][0]
    elif mutation == "policy":
        evidence["eligibility_policy"]["token_limits"]["shape512"] = 8193
    elif mutation == "candidate_count":
        evidence["candidate_asset_count"] -= 1
    else:
        evidence["training_exclusions"][0]["reasons"] = ["changed_reason"]
        evidence["training_exclusion_reason_counts"] = {
            reason: 1
            for exclusion in evidence["training_exclusions"]
            for reason in exclusion["reasons"]
        }
    results = with_evidence_digests(results, materializations)
    with pytest.raises(ValueError, match="materialization"):
        preflight.build_report(tmp_path / "index.json", "i" * 64, results, materializations, "2026-07-25T00:00:00Z")


def test_materialization_scope_rejects_a_training_excluded_directory(tmp_path):
    """A removed asset cannot remain hidden under even one final component."""
    results, materializations = handoff_inputs(tmp_path)
    root = tmp_path / "isolated" / "shape512" / "active"
    root.mkdir(parents=True)
    evidence = materializations["shape512"]
    evidence["stage_root"] = str(root.resolve())
    (root / "materialization.json").write_text(json.dumps(evidence))
    excluded = evidence["training_exclusions"][0]["asset"]
    (root / "renders_cond" / excluded).mkdir(parents=True)
    with pytest.raises(ValueError, match="training-excluded"):
        preflight._materialization_scope("shape512", root)


@pytest.mark.parametrize("digest", ["", None, "not-a-sha"])
def test_build_report_requires_a_valid_preflight_evidence_digest(tmp_path, digest):
    """Omitting the preflight evidence digest would allow a TOCTOU publication."""
    results, materializations = handoff_inputs(tmp_path)
    results = with_evidence_digests(results, materializations)
    results["ss64"] = replace(results["ss64"], materialization_sha256=digest)
    with pytest.raises(ValueError, match="materialization"):
        preflight.build_report(tmp_path / "index.json", "i" * 64, results, materializations, "2026-07-25T00:00:00Z")


def test_build_report_rejects_a_materialization_source_index_path_mismatch(tmp_path):
    """A matching index digest cannot authorize evidence that names another index path."""
    index = tmp_path / "index.json"
    results, materializations = handoff_inputs(tmp_path)
    for evidence in materializations.values():
        evidence["source_index"]["path"] = str(index.resolve())
    materializations["shape512"]["source_index"]["path"] = str((tmp_path / "other-index.json").resolve())
    results = with_evidence_digests(results, materializations)
    with pytest.raises(ValueError, match="materialization"):
        preflight.build_report(index, "i" * 64, results, materializations, "2026-07-25T00:00:00Z")


def test_report_and_handoff_preserve_waiver_evidence_and_isolated_data_dirs(tmp_path):
    """Dropping a waiver count, evidence digest, or stage root must invalidate the handoff."""
    index = tmp_path / "immutable-index.json"
    results, materializations = handoff_inputs(tmp_path, index_path=index)
    report = preflight.build_report(
        index, "i" * 64, results, materializations, "2026-07-25T00:00:00Z"
    )

    assert report["acceptance_mode"] == "valid_subset_user_waiver"
    assert report["original_90_percent_gate_passed"] is False
    assert report["counts"] == {
        **HANDOFF_COUNTS,
    }
    assert report["observed_tool_commits"] == sorted([
        "pbr1024-pack-tool", "pbr1024-tool", "shape1024-pack-tool", "shape1024-tool",
        "shape512-pack-tool", "shape512-tool", "ss64-pack-tool", "ss64-tool",
    ])
    for stage, evidence in materializations.items():
        assert report["materialization_evidence"][stage]["sha256"] == hashlib.sha256(
            canonical_json_bytes(evidence)
        ).hexdigest()
        assert report["stages"][stage]["data_dir"] == preflight.stage_data_dir(
            stage, results[stage].root
        )

    report_path = tmp_path / "shared" / "report.json"
    handoff = preflight.build_handoff(
        report_path,
        hashlib.sha256(canonical_json_bytes(report)).hexdigest(),
        report,
        results,
        materializations,
        "2026-07-25T00:00:00Z",
    )
    assert handoff["report"] == {
        "path": str(report_path),
        "sha256": hashlib.sha256(canonical_json_bytes(report)).hexdigest(),
    }
    assert handoff["authorization"] == "training-input use only"
    assert handoff["original_90_percent_gate_passed"] is False
    assert report["eligibility_policy"] == policy_evidence()
    assert handoff["eligibility_policy"] == policy_evidence()
    assert handoff["stages"]["pbr1024"]["data_dir"] == {
        "ABO": {
            "base": str(tmp_path / "isolated" / "pbr1024" / "active"),
            "render_cond": str(tmp_path / "isolated" / "pbr1024" / "active" / "renders_cond"),
            "shape_latent": str(tmp_path / "isolated" / "pbr1024" / "active" / "shape_latents" / "shape_enc_next_dc_f16c32_fp16_1024_view"),
            "pbr_latent": str(tmp_path / "isolated" / "pbr1024" / "active" / "pbr_latents" / "tex_enc_next_dc_f16c32_fp16_1024_view_fix"),
        }
    }


@pytest.mark.parametrize("mutation", [
    "source_index_path", "source_index_digest", "stage_data", "evidence", "commits",
])
def test_build_handoff_rejects_report_not_bound_to_supplied_preflight_evidence(tmp_path, mutation):
    """A direct caller must not combine a valid report with another index or stage evidence."""
    results, materializations = handoff_inputs(tmp_path, index_path=tmp_path / "index-a.json")
    report = preflight.build_report(
        tmp_path / "index-a.json", "i" * 64, results, materializations,
        "2026-07-25T00:00:00Z",
    )
    report = json.loads(json.dumps(report))
    if mutation == "source_index_path":
        report["source_index"]["path"] = ""
    elif mutation == "source_index_digest":
        report["source_index"]["sha256"] = "b" * 64
    elif mutation == "stage_data":
        report["stages"]["pbr1024"]["data_dir"] = {"ABO": {"base": "/wrong"}}
    elif mutation == "evidence":
        report["materialization_evidence"]["pbr1024"]["sha256"] = "b" * 64
    else:
        report["observed_tool_commits"] = ["wrong-tool"]
    with pytest.raises(ValueError):
        preflight.build_handoff(
            tmp_path / "report.json",
            hashlib.sha256(canonical_json_bytes(report)).hexdigest(),
            report,
            results,
            materializations,
            "2026-07-25T00:00:00Z",
        )


@pytest.mark.parametrize("path_spelling", ["relative", "parent"])
def test_build_handoff_rejects_noncanonical_report_source_index_path(tmp_path, path_spelling):
    """A direct builder must not copy a relative or dot-dot source index into a handoff."""
    index = tmp_path / "index.json"
    results, materializations = handoff_inputs(tmp_path, index_path=index)
    report = preflight.build_report(index, "i" * 64, results, materializations, "2026-07-25T00:00:00Z")
    if path_spelling == "relative":
        report["source_index"]["path"] = os.path.relpath(index, Path.cwd())
    else:
        report["source_index"]["path"] = str(index.parent / "nested" / ".." / index.name)
    with pytest.raises(ValueError, match="source_index"):
        preflight.build_handoff(
            tmp_path / "report.json", hashlib.sha256(canonical_json_bytes(report)).hexdigest(),
            report, results, materializations, "2026-07-25T00:00:00Z",
        )


@pytest.mark.parametrize("mutation", ["missing", "reordered", "digest"])
def test_handoff_rejects_noncanonical_materialization_scope(tmp_path, mutation):
    """A published handoff must not certify evidence whose claimed scope was not proven."""
    results, materializations = handoff_inputs(tmp_path)
    evidence = materializations["ss64"]
    if mutation == "missing":
        evidence.pop("stage_scope", None)
    elif mutation == "reordered":
        evidence["stage_scope"] = ["b", "a"]
    else:
        evidence["stage_scope"] = ["a"]
        evidence["stage_scope_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        preflight.build_report(
            tmp_path / "index.json", "i" * 64, results, materializations,
            "2026-07-25T00:00:00Z",
        )


def test_publish_recovers_an_existing_report_with_its_original_timestamp(tmp_path):
    """A rerun after report-only publication must finish locally without changing shared bytes."""
    results, materializations = handoff_inputs(tmp_path)
    index = tmp_path / "index.json"
    index.write_text('{"source":"ABO"}\n')
    index_sha256 = hashlib.sha256(index.read_bytes()).hexdigest()
    for evidence in materializations.values():
        evidence["index_sha256"] = index_sha256
        evidence["source_index"]["sha256"] = index_sha256
    results = with_evidence_digests(results, materializations)
    report_path = tmp_path / "shared" / "report.json"
    report = preflight.build_report(index, index_sha256, results, materializations, "2026-01-01T00:00:00Z")
    preflight.write_create_only_json(report_path, report)
    original = report_path.read_bytes()
    handoff_path = tmp_path / "shared" / "handoff.json"
    training_path = tmp_path / "local" / "training_data.json"
    preflight.publish_handoff(
        index, results, materializations, report_path, handoff_path, training_path,
        "2026-12-31T23:59:59Z",
    )
    assert report_path.read_bytes() == original
    assert json.loads(handoff_path.read_text())["created_at"] == "2026-01-01T00:00:00Z"
    assert training_path.exists()


def test_publish_recovers_existing_report_and_handoff_after_local_failure(tmp_path):
    """A local-only interruption must not require changing immutable shared transaction bytes."""
    results, materializations = handoff_inputs(tmp_path)
    index = tmp_path / "index.json"
    index.write_text('{"source":"ABO"}\n')
    index_sha256 = hashlib.sha256(index.read_bytes()).hexdigest()
    for evidence in materializations.values():
        evidence["index_sha256"] = index_sha256
        evidence["source_index"]["sha256"] = index_sha256
    results = with_evidence_digests(results, materializations)
    report_path = tmp_path / "shared" / "report.json"
    handoff_path = tmp_path / "shared" / "handoff.json"
    training_path = tmp_path / "local" / "training_data.json"
    preflight.publish_handoff(index, results, materializations, report_path, handoff_path, training_path, "2026-01-01T00:00:00Z")
    shared = (report_path.read_bytes(), handoff_path.read_bytes())
    training_path.unlink()
    preflight.publish_handoff(index, results, materializations, report_path, handoff_path, training_path, "2026-12-31T23:59:59Z")
    assert (report_path.read_bytes(), handoff_path.read_bytes()) == shared
    assert training_path.exists()


def test_publish_rejects_evidence_reread_that_differs_from_preflight_bytes(tmp_path):
    """A valid but later-mutated evidence object must not replace the preflight-proven bytes."""
    results, materializations = handoff_inputs(tmp_path)
    index = tmp_path / "index.json"
    index.write_text('{"source":"ABO"}\n')
    index_sha256 = hashlib.sha256(index.read_bytes()).hexdigest()
    for evidence in materializations.values():
        evidence["index_sha256"] = index_sha256
        evidence["source_index"]["sha256"] = index_sha256
    results = with_evidence_digests(results, materializations)
    for stage, result in list(results.items()):
        results[stage] = replace(result, materialization_sha256=hashlib.sha256(canonical_json_bytes(materializations[stage])).hexdigest())
    materializations["ss64"]["tool_commits"] = ["later-mutation"]
    with pytest.raises(ValueError, match="materialization evidence"):
        preflight.publish_handoff(index, results, materializations, tmp_path / "report.json", tmp_path / "handoff.json", tmp_path / "training.json", "2026-01-01T00:00:00Z")


def test_create_only_json_is_idempotent_but_refuses_different_or_symlinked_content(tmp_path):
    """Replacing a published shared document must be rejected rather than overwritten."""
    path = tmp_path / "shared" / "artifact.json"
    first = preflight.write_create_only_json(path, {"a": 1})
    assert first == hashlib.sha256(b'{\n  "a": 1\n}\n').hexdigest()
    assert preflight.write_create_only_json(path, {"a": 1}) == first
    with pytest.raises(FileExistsError, match="different"):
        preflight.write_create_only_json(path, {"a": 2})
    linked = tmp_path / "shared" / "linked.json"
    linked.symlink_to(path)
    with pytest.raises(FileExistsError, match="regular non-symlink"):
        preflight.write_create_only_json(linked, {"a": 1})


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires a POSIX filesystem")
def test_create_only_json_rejects_a_fifo_without_blocking(tmp_path):
    """Opening a special file before checking its type can hang the preflight command."""
    fifo = tmp_path / "shared" / "artifact.json"
    fifo.parent.mkdir()
    os.mkfifo(fifo)

    def timeout(_signum, _frame):
        raise TimeoutError("FIFO open blocked")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 1)
    started = time.monotonic()
    try:
        with pytest.raises(FileExistsError, match="regular non-symlink"):
            preflight.write_create_only_json(fifo, {"a": 1})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert time.monotonic() - started < 0.2


@pytest.mark.parametrize("incomplete", ["absent", "failed"])
def test_publish_handoff_withholds_all_outputs_until_every_stage_passes(tmp_path, incomplete):
    """Publishing after a missing or incomplete strict preflight would authorize unsafe input."""
    results, materializations = handoff_inputs(tmp_path)
    if incomplete == "absent":
        results.pop("pbr1024")
    else:
        failed = results["pbr1024"]
        results["pbr1024"] = preflight.StagePreflight(
            failed.stage, failed.root, failed.asset_count, failed.asset_scope_sha256,
            failed.anchors_checked - 1, failed.validation_counts, failed.materialization_sha256,
        )
    index = tmp_path / "index.json"
    index.write_text('{"source":"ABO"}\n')
    report = tmp_path / "shared" / "report.json"
    handoff = tmp_path / "shared" / "handoff.json"
    training_data = tmp_path / "local" / "training_data.json"
    with pytest.raises(ValueError, match="strict preflight"):
        preflight.publish_handoff(
            index, results, materializations, report, handoff, training_data,
            "2026-07-25T00:00:00Z",
        )
    assert not report.exists()
    assert not handoff.exists()
    assert not training_data.exists()


def test_publish_handoff_cross_links_canonical_shared_artifacts_before_local_manifest(tmp_path):
    """A changed report serializer or early local write must break this immutable handoff."""
    results, materializations = handoff_inputs(tmp_path)
    index = tmp_path / "index.json"
    index.write_text('{"source":"ABO"}\n')
    index_sha256 = hashlib.sha256(index.read_bytes()).hexdigest()
    for evidence in materializations.values():
        evidence["index_sha256"] = index_sha256
        evidence["source_index"]["sha256"] = index_sha256
    results = with_evidence_digests(results, materializations)
    report_path = tmp_path / "shared" / "report.json"
    handoff_path = tmp_path / "shared" / "handoff.json"
    training_path = tmp_path / "local" / "training_data.json"
    assert preflight.publish_handoff(
        index, results, materializations, report_path, handoff_path, training_path,
        "2026-07-25T00:00:00Z",
    ) == (report_path, handoff_path, training_path)
    report = json.loads(report_path.read_text())
    handoff = json.loads(handoff_path.read_text())
    training_data = json.loads(training_path.read_text())
    assert handoff["report"]["sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert training_data["handoff"] == {
        "path": str(handoff_path),
        "sha256": hashlib.sha256(handoff_path.read_bytes()).hexdigest(),
    }
    assert training_data["authorization"] == "training-input use only"
    assert report["counts"] == handoff["counts"] == training_data["counts"] == HANDOFF_COUNTS
    assert report["eligibility_policy"] == handoff["eligibility_policy"] == training_data["eligibility_policy"] == policy_evidence()
    assert training_data["materialization_evidence"] == handoff["materialization_evidence"]
    assert training_data["observed_tool_commits"] == handoff["observed_tool_commits"]
    assert handoff["source_index"] == report["source_index"]
    assert training_data["source_index"] == report["source_index"]
    assert training_data["report"] == handoff["report"]


def test_publish_handoff_withholds_local_manifest_when_shared_handoff_rejects_content(tmp_path):
    """The local convenience file must not appear when the second shared publish fails."""
    results, materializations = handoff_inputs(tmp_path)
    index = tmp_path / "index.json"
    index.write_text('{"source":"ABO"}\n')
    index_sha256 = hashlib.sha256(index.read_bytes()).hexdigest()
    for evidence in materializations.values():
        evidence["index_sha256"] = index_sha256
        evidence["source_index"]["sha256"] = index_sha256
    results = with_evidence_digests(results, materializations)
    report_path = tmp_path / "shared" / "report.json"
    handoff_path = tmp_path / "shared" / "handoff.json"
    handoff_path.parent.mkdir(parents=True)
    handoff_path.write_text('{"different":true}\n')
    training_path = tmp_path / "local" / "training_data.json"
    with pytest.raises(ValueError, match="existing handoff"):
        preflight.publish_handoff(
            index, results, materializations, report_path, handoff_path, training_path,
            "2026-07-25T00:00:00Z",
        )
    assert report_path.exists()
    assert not training_path.exists()


@pytest.mark.parametrize("mutation", ["root", "count", "scope", "index"])
def test_publish_handoff_rejects_materialization_evidence_not_bound_to_preflight(tmp_path, mutation):
    """Publishing evidence for a different root, scope, count, or index is unsafe."""
    index = tmp_path / "index.json"
    index.write_text('{"source":"ABO"}\n')
    results, materializations = handoff_inputs(
        tmp_path, hashlib.sha256(index.read_bytes()).hexdigest()
    )
    evidence = materializations["pbr1024"]
    if mutation == "root":
        evidence["stage_root"] = str(tmp_path / "other" / "active")
    elif mutation == "count":
        evidence["asset_count"] = 1
    elif mutation == "scope":
        evidence["stage_scope_sha256"] = "x" * 64
    else:
        evidence["index_sha256"] = "x" * 64
    report = tmp_path / "shared" / "report.json"
    handoff = tmp_path / "shared" / "handoff.json"
    training_data = tmp_path / "local" / "training_data.json"
    with pytest.raises(ValueError, match="materialization evidence"):
        preflight.publish_handoff(
            index, results, materializations, report, handoff, training_data,
            "2026-07-25T00:00:00Z",
        )
    assert not report.exists()
    assert not handoff.exists()
    assert not training_data.exists()


def test_cli_completes_all_strict_preflights_before_requesting_handoff(tmp_path, monkeypatch):
    """Moving publication into a stage loop would authorize training after a partial check."""
    results, materializations = handoff_inputs(tmp_path)
    index = tmp_path / "index.json"
    index.write_text('{"source":"ABO"}\n')
    seen = []

    def fake_preflight(stage: str, root: Path, config: Path) -> preflight.StagePreflight:
        assert root == results[stage].root
        seen.append(stage)
        return results[stage]

    def fake_publish(
        received_index: Path,
        received_results: dict[str, preflight.StagePreflight],
        received_materializations: dict[str, dict[str, object]],
        report: Path,
        handoff: Path,
        training_data: Path,
        created_at: str,
    ) -> tuple[Path, Path, Path]:
        assert seen == list(HANDOFF_STAGE_COUNTS)
        assert received_index == index
        assert received_results == results
        assert received_materializations == materializations
        assert created_at.endswith("Z")
        return report, handoff, training_data

    for stage, evidence in materializations.items():
        root = results[stage].root
        root.mkdir(parents=True)
        (root / "materialization.json").write_bytes(canonical_json_bytes(evidence))
    monkeypatch.setattr(preflight, "preflight_stage", fake_preflight)
    monkeypatch.setattr(preflight, "publish_handoff", fake_publish)
    monkeypatch.setattr("sys.argv", [
        "preflight_multiview_production.py",
        "--root", str(tmp_path / "isolated"),
        "--index", str(index),
        "--report", str(tmp_path / "shared" / "report.json"),
        "--handoff", str(tmp_path / "shared" / "handoff.json"),
        "--training-data", str(tmp_path / "local" / "training_data.json"),
    ])
    preflight.main()
