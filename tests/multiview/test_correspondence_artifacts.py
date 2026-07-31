import json
from copy import deepcopy
from hashlib import sha256

import pytest
import torch
from PIL import Image, ImageChops

from pixal3d.experiments.correspondence import (
    BundleCorruption,
    CalibratedView,
    ControlledCorruption,
    ForegroundMask,
    validate_artifact_bundle,
    write_artifact_bundle,
    write_failed_artifact_bundle,
)


def _views_and_masks(tmp_path):
    views = []
    masks = []
    for index, color in enumerate(((20, 30, 40, 255), (60, 70, 80, 255))):
        source_path = tmp_path / f"input_{index}.png"
        image = Image.new("RGBA", (16, 12), color)
        image.save(source_path)
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, 3] = torch.tensor((index, index + 1, index + 2))
        views.append(
            CalibratedView(
                frame_index=index + 7,
                manifest_root=tmp_path,
                frame_file_path=source_path.name,
                image_path=source_path,
                image=image,
                camera_angle_x=0.65 + index / 10,
                distance=float(torch.linalg.vector_norm(transform[:3, 3])),
                transform_matrix=transform,
                source_sha256=sha256(source_path.read_bytes()).hexdigest(),
            )
        )
        mask = torch.zeros((12, 16), dtype=torch.bool)
        mask[2:10, 3:14] = True
        masks.append(
            ForegroundMask(
                mask=mask,
                provenance="explicit" if index == 0 else "alpha",
                source_sha256=f"{index + 1:064x}",
            )
        )
    return views, masks


def _corruptions():
    entries = []
    colors = {"c1": (0.9, 0.1, 0.1), "c2": (0.1, 0.9, 0.1), "c3": (0.1, 0.1, 0.9)}
    kinds = {"c1": "c1_color", "c2": "c2_pattern", "c3": "c3_deletion"}
    parameters = {
        "c1": {"hue": 0.1, "saturation": 1.2, "brightness": 0.8},
        "c2": {"seed": 42, "pattern": "sole"},
        "c3": {"fill_source": "background", "fill_rgb": [0.2, 0.3, 0.4]},
    }
    for arm in ("c3", "c1", "c2"):
        image = torch.tensor(colors[arm], dtype=torch.float32).view(3, 1, 1)
        image = image.expand(3, 12, 16).clone()
        oracle = torch.zeros((12, 16), dtype=torch.bool)
        oracle[4:8, 6:11] = True
        entries.append(
            BundleCorruption(
                arm=arm,
                view_index=1,
                corruption=ControlledCorruption(
                    image=image,
                    oracle_mask=oracle,
                    kind=kinds[arm],
                    parameters=parameters[arm],
                ),
            )
        )
    return entries


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


def _write_canonical_manifest(path, manifest):
    path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def test_bundle_atomically_writes_canonical_manifest_relative_paths_and_hashes(
    tmp_path,
):
    views, masks = _views_and_masks(tmp_path)
    output = tmp_path / "bundles"

    run_dir = write_artifact_bundle(
        output,
        "case-001",
        views,
        masks,
        _corruptions(),
        seed=42,
        mesh_scale=1.25,
    )

    assert run_dir == output / "case-001"
    assert not list(output.glob(".case-001.tmp-*"))
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest_path.read_text() == json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    assert manifest["status"] == "completed"
    assert manifest["run_id"] == "case-001"
    assert manifest["K"] == 2
    assert manifest["seed"] == 42
    assert manifest["mesh_scale"] == 1.25
    assert [view["view_index"] for view in manifest["views"]] == [0, 1]
    assert [view["frame_index"] for view in manifest["views"]] == [7, 8]
    assert [view["camera_angle_x"] for view in manifest["views"]] == [0.65, 0.75]
    assert [view["distance"] for view in manifest["views"]] == pytest.approx(
        [5**0.5, 14**0.5]
    )
    assert manifest["views"][1][
        "transform_matrix"
    ] == views[1].transform_matrix.tolist()

    for index, view_record in enumerate(manifest["views"]):
        source = view_record["source"]
        foreground = view_record["foreground_mask"]
        assert source["path"] == f"views/view_{index:02d}/source.png"
        assert foreground["path"] == f"views/view_{index:02d}/foreground_mask.png"
        assert source["sha256"] == _sha(run_dir / source["path"])
        assert foreground["sha256"] == _sha(run_dir / foreground["path"])
        assert source["input_sha256"] == views[index].source_sha256
        assert foreground["provenance"] == masks[index].provenance
        assert foreground["source_sha256"] == masks[index].source_sha256

    assert [entry["arm"] for entry in manifest["corruptions"]] == ["c1", "c2", "c3"]
    for entry in manifest["corruptions"]:
        expected_root = f"corruptions/{entry['arm']}/view_01"
        assert entry["view_index"] == 1
        assert entry["image"]["path"] == f"{expected_root}/image.png"
        assert entry["oracle_mask"]["path"] == f"{expected_root}/oracle_mask.png"
        assert entry["image"]["sha256"] == _sha(run_dir / entry["image"]["path"])
        assert entry["oracle_mask"]["sha256"] == _sha(
            run_dir / entry["oracle_mask"]["path"]
        )
        expected = next(item for item in _corruptions() if item.arm == entry["arm"])
        assert entry["kind"] == expected.corruption.kind
        assert entry["parameters"] == expected.corruption.parameters


def test_bundle_resume_validates_identical_complete_run_without_rewriting(tmp_path):
    views, masks = _views_and_masks(tmp_path)
    arguments = (tmp_path / "bundles", "case-001", views, masks, _corruptions())
    run_dir = write_artifact_bundle(*arguments, seed=42, mesh_scale=1.0)
    before = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    resumed = write_artifact_bundle(*arguments, seed=42, mesh_scale=1.0)

    after = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert resumed == run_dir
    assert after == before


def test_bundle_oracle_mask_uses_saved_png_change_threshold(tmp_path):
    views, masks = _views_and_masks(tmp_path)
    one_quantization_step = torch.tensor(
        (59 / 255, 69 / 255, 79 / 255), dtype=torch.float32
    ).view(3, 1, 1)
    oracle = torch.zeros((12, 16), dtype=torch.bool)
    oracle[4:8, 6:11] = True
    corruptions = [
        BundleCorruption(
            arm=entry.arm,
            view_index=1,
            corruption=ControlledCorruption(
                image=one_quantization_step.expand(3, 12, 16).clone(),
                oracle_mask=oracle,
                kind=entry.corruption.kind,
                parameters=entry.corruption.parameters,
            ),
        )
        for entry in _corruptions()
    ]

    run_dir = write_artifact_bundle(
        tmp_path / "bundles",
        "quantization-threshold",
        views,
        masks,
        corruptions,
        seed=42,
        mesh_scale=1.0,
    )

    for arm in ("c1", "c2", "c3"):
        saved_mask = Image.open(
            run_dir / f"corruptions/{arm}/view_01/oracle_mask.png"
        ).convert("L")
        assert saved_mask.getbbox() is None


def test_bundle_refuses_hash_mismatched_complete_run_without_overwriting(tmp_path):
    views, masks = _views_and_masks(tmp_path)
    arguments = (tmp_path / "bundles", "case-001", views, masks, _corruptions())
    run_dir = write_artifact_bundle(*arguments, seed=42, mesh_scale=1.0)
    corrupted_path = run_dir / "corruptions/c2/view_01/image.png"
    corrupted_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        write_artifact_bundle(*arguments, seed=42, mesh_scale=1.0)

    assert corrupted_path.read_bytes() == b"tampered"


def test_bundle_refuses_existing_run_symlink_outside_output_directory(tmp_path):
    views, masks = _views_and_masks(tmp_path)
    corruptions = _corruptions()
    external = write_artifact_bundle(
        tmp_path / "external",
        "case-001",
        views,
        masks,
        corruptions,
        seed=42,
        mesh_scale=1.0,
    )
    output = tmp_path / "bundles"
    output.mkdir()
    (output / "case-001").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="run directory must not be a symlink"):
        write_artifact_bundle(
            output,
            "case-001",
            views,
            masks,
            corruptions,
            seed=42,
            mesh_scale=1.0,
        )


def test_validator_rejects_manifest_symlink_outside_run_directory(tmp_path):
    views, masks = _views_and_masks(tmp_path)
    run_dir = write_artifact_bundle(
        tmp_path / "bundles",
        "case-001",
        views,
        masks,
        _corruptions(),
        seed=42,
        mesh_scale=1.0,
    )
    manifest_path = run_dir / "manifest.json"
    outside = tmp_path / "outside-manifest.json"
    manifest_path.rename(outside)
    manifest_path.symlink_to(outside)

    with pytest.raises(
        ValueError, match="manifest must be a non-symlink regular file"
    ):
        validate_artifact_bundle(run_dir)


def _drop_schema(manifest):
    manifest.pop("schema_version")


def _wrong_schema(manifest):
    manifest["schema_version"] = 2


def _wrong_run_id(manifest):
    manifest["run_id"] = "different-run"


def _wrong_k(manifest):
    manifest["K"] += 1


def _missing_view_camera(manifest):
    manifest["views"][0].pop("camera_angle_x")


def _missing_view_transform(manifest):
    manifest["views"][0].pop("transform_matrix")


def _missing_source_provenance(manifest):
    manifest["views"][0]["source"].pop("input_sha256")


def _missing_mask_provenance(manifest):
    manifest["views"][0]["foreground_mask"].pop("provenance")


def _missing_corruption_arm(manifest):
    manifest["corruptions"] = manifest["corruptions"][:2]


def _duplicate_corruption_arm(manifest):
    manifest["corruptions"][2]["arm"] = "c2"


def _wrong_corruption_path(manifest):
    source = manifest["views"][1]["source"]
    manifest["corruptions"][0]["image"] = {
        "path": source["path"],
        "sha256": source["sha256"],
    }


def _noninteger_corruption_view_index(manifest):
    manifest["corruptions"][0]["view_index"] = "1"


def _completed_with_failure_reason(manifest):
    manifest["failure_reason"] = "must be absent on success"


@pytest.mark.parametrize(
    "mutation",
    [
        _drop_schema,
        _wrong_schema,
        _wrong_run_id,
        _wrong_k,
        _missing_view_camera,
        _missing_view_transform,
        _missing_source_provenance,
        _missing_mask_provenance,
        _missing_corruption_arm,
        _duplicate_corruption_arm,
        _wrong_corruption_path,
        _noninteger_corruption_view_index,
        _completed_with_failure_reason,
    ],
)
def test_validator_rejects_canonical_incomplete_completed_contract(
    tmp_path, mutation
):
    views, masks = _views_and_masks(tmp_path)
    run_dir = write_artifact_bundle(
        tmp_path / "bundles",
        "case-001",
        views,
        masks,
        _corruptions(),
        seed=42,
        mesh_scale=1.0,
    )
    manifest_path = run_dir / "manifest.json"
    manifest = deepcopy(json.loads(manifest_path.read_text()))
    mutation(manifest)
    _write_canonical_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="invalid completed bundle contract"):
        validate_artifact_bundle(run_dir)


def test_validator_rejects_failed_contract_without_failure_reason(tmp_path):
    run_dir = write_failed_artifact_bundle(
        tmp_path / "bundles",
        "failed-mask",
        seed=42,
        mesh_scale=1.0,
        num_views=4,
        failure_reason="foreground mask is empty",
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["failure_reason"] = None
    _write_canonical_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="invalid failed bundle contract"):
        validate_artifact_bundle(run_dir)


def test_bundle_rejects_invalid_corruption_entry_with_value_error(tmp_path):
    views, masks = _views_and_masks(tmp_path)
    corruptions = _corruptions()
    corruptions[0] = object()

    with pytest.raises(ValueError, match="corruptions have invalid entries"):
        write_artifact_bundle(
            tmp_path / "bundles",
            "case-001",
            views,
            masks,
            corruptions,
            seed=42,
            mesh_scale=1.0,
        )


def test_bundle_rejects_nonfinite_manifest_data_without_publishing(tmp_path):
    views, masks = _views_and_masks(tmp_path)
    corruptions = _corruptions()
    bad = corruptions[0]
    corruptions[0] = BundleCorruption(
        arm=bad.arm,
        view_index=bad.view_index,
        corruption=ControlledCorruption(
            image=bad.corruption.image,
            oracle_mask=bad.corruption.oracle_mask,
            kind=bad.corruption.kind,
            parameters={**bad.corruption.parameters, "bad": float("nan")},
        ),
    )
    output = tmp_path / "bundles"

    with pytest.raises(ValueError, match="finite JSON"):
        write_artifact_bundle(
            output,
            "case-001",
            views,
            masks,
            corruptions,
            seed=42,
            mesh_scale=1.0,
        )

    assert not (output / "case-001").exists()
    assert not list(output.glob(".case-001.tmp-*"))


def test_contact_sheet_has_labeled_ordered_source_and_mask_overlay_columns(tmp_path):
    views, masks = _views_and_masks(tmp_path)

    run_dir = write_artifact_bundle(
        tmp_path / "bundles",
        "visual-order",
        views,
        masks,
        _corruptions(),
        seed=42,
        mesh_scale=1.0,
    )

    with Image.open(run_dir / "contact_sheet.png") as source:
        sheet = source.convert("RGB")
    assert sheet.size == (808, 680)
    expected_corrupted_colors = [(230, 26, 26), (26, 230, 26), (26, 26, 230)]
    for row, expected_color in enumerate(expected_corrupted_colors):
        label_y = 8 + row * 224
        image_y = label_y + 24
        center_y = image_y + 96
        centers = [
            sheet.getpixel((104 + column * 200, center_y))
            for column in range(4)
        ]
        assert centers[0] == (60, 70, 80)
        assert centers[1] != centers[0]
        assert centers[2] == expected_color
        assert centers[3] != centers[2]
        for column in range(4):
            label = sheet.crop(
                (8 + column * 200, label_y, 200 + column * 200, image_y)
            )
            white = Image.new("RGB", label.size, "white")
            assert ImageChops.difference(label, white).getbbox() is not None


def test_failed_bundle_records_reason_and_writes_labeled_failure_contact_sheet(
    tmp_path,
):
    output = tmp_path / "bundles"

    run_dir = write_failed_artifact_bundle(
        output,
        "failed-mask",
        seed=42,
        mesh_scale=1.0,
        num_views=4,
        failure_reason="foreground mask is empty",
    )

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest == {
        "K": 4,
        "contact_sheet": {
            "path": "contact_sheet.png",
            "sha256": _sha(run_dir / "contact_sheet.png"),
        },
        "corruptions": [],
        "failure_reason": "foreground mask is empty",
        "mesh_scale": 1.0,
        "run_id": "failed-mask",
        "schema_version": 1,
        "seed": 42,
        "status": "failed",
        "views": [],
    }
    with Image.open(run_dir / "contact_sheet.png") as source:
        failure_sheet = source.convert("RGB")
    assert failure_sheet.size == (808, 128)
    assert ImageChops.difference(
        failure_sheet, Image.new("RGB", failure_sheet.size, "white")
    ).getbbox() is not None

    assert write_failed_artifact_bundle(
        output,
        "failed-mask",
        seed=42,
        mesh_scale=1.0,
        num_views=4,
        failure_reason="foreground mask is empty",
    ) == run_dir
