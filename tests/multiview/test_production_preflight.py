import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

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


def test_direct_loader_surfaces_damaged_anchor_instead_of_retrying_another_sample(tmp_path, monkeypatch):
    root = make_stage(tmp_path)
    (component_root("ss64", root, "ss") / ASSET / "view01.npz").unlink()
    config = write_loader_config(tmp_path, "ss64")
    from pixal3d.datasets.components import StandardDatasetBase
    monkeypatch.setattr(StandardDatasetBase, "__getitem__", lambda self, index: pytest.fail("dataset retry"))
    with pytest.raises(RuntimeError, match=r"source=ABO.*asset=" + ASSET + r".*anchor=view01"):
        preflight.validate_direct_loader("ss64", root, [ASSET], config)


def test_preflight_stage_reads_materialization_scope_and_returns_frozen_result(tmp_path):
    root = make_stage(tmp_path)
    digest = __import__("hashlib").sha256(ASSET.encode()).hexdigest()
    (root / "materialization.json").write_text(json.dumps({
        "stage": "ss64", "asset_count": 1, "stage_scope": [ASSET],
        "stage_scope_sha256": digest, "stage_root": str(root.resolve()),
    }))
    result = preflight.preflight_stage("ss64", root, write_loader_config(tmp_path, "ss64"))
    assert result.stage == "ss64"
    assert result.root == root
    assert result.asset_count == 1 and result.asset_scope_sha256 == digest
    assert result.anchors_checked == 2
    assert result.validation_counts == {"assets": 1, "renders": 8, "latents": 2, "scales": 2}
    with pytest.raises((AttributeError, TypeError)):
        result.stage = "changed"


@pytest.mark.parametrize("mutation", ["missing", "altered", "mismatched"])
def test_preflight_rejects_missing_altered_or_mismatched_materialization_root(tmp_path, mutation):
    root = make_stage(tmp_path)
    digest = __import__("hashlib").sha256(ASSET.encode()).hexdigest()
    evidence = {
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
