from hashlib import sha256
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/prepare_correspondence_corruptions.py"


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


def _transform(translation: tuple[float, float, float]) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = translation
    return matrix.tolist()


def _write_synthetic_dataset(root: Path) -> tuple[Path, list[dict], list[Path]]:
    root.mkdir()
    image_paths = []
    frames = []
    translations = ((0.0, 0.0, 2.0), (3.0, 4.0, 0.0), (1.0, 2.0, 2.0), (4.0, 4.0, 7.0))
    y, x = np.mgrid[:40, :48]
    for index, translation in enumerate(translations):
        red = (31 + index * 37 + x * 3 + y) % 256
        green = (83 + index * 29 + x + y * 4) % 256
        blue = (149 + index * 17 + x * 2 + y * 3) % 256
        alpha = np.zeros((40, 48), dtype=np.uint8)
        if index:
            alpha[5:35, 7:41] = 255
        pixels = np.stack((red, green, blue, alpha), axis=-1).astype(np.uint8)
        image_path = root / f"ordered_view_{index}.png"
        Image.fromarray(pixels, mode="RGBA").save(image_path)
        frame = {
            "file_path": image_path.name,
            "transform_matrix": _transform(translation),
        }
        if index in {0, 2}:
            frame["camera_angle_x"] = 0.71 + index / 100
        image_paths.append(image_path)
        frames.append(frame)

    explicit_mask = np.zeros((40, 48), dtype=np.uint8)
    explicit_mask[8:32, 10:38] = 255
    mask_path = root / "view_0_foreground.png"
    Image.fromarray(explicit_mask, mode="L").save(mask_path)
    frames[0]["foreground_mask_path"] = mask_path.name

    transforms = root / "transforms.json"
    transforms.write_text(json.dumps({"camera_angle_x": 0.65, "frames": frames}))
    return transforms, frames, image_paths


def _artifact_records(manifest: dict) -> list[dict[str, str]]:
    records = []
    for view in manifest["views"]:
        records.extend((view["source"], view["foreground_mask"]))
    for corruption in manifest["corruptions"]:
        records.extend((corruption["image"], corruption["oracle_mask"]))
    records.append(manifest["contact_sheet"])
    return records


@pytest.mark.parametrize(
    "missing_option",
    ["--transforms", "--output-dir", "--mesh-scale"],
)
def test_cli_requires_every_dataset_and_scale_option(tmp_path, missing_option):
    values = {
        "--transforms": str(tmp_path / "transforms.json"),
        "--output-dir": str(tmp_path / "output"),
        "--mesh-scale": "1.0",
    }
    arguments = [
        value
        for option, option_value in values.items()
        if option != missing_option
        for value in (option, option_value)
    ]

    result = _run_cli(*arguments)

    assert result.returncode == 2
    assert missing_option in result.stderr


@pytest.mark.parametrize("corrupt_index", ["-1", "4"])
def test_cli_rejects_corrupt_view_index_outside_first_k(
    tmp_path, corrupt_index
):
    result = _run_cli(
        "--transforms",
        str(tmp_path / "transforms.json"),
        "--output-dir",
        str(tmp_path / "output"),
        "--mesh-scale",
        "1.0",
        "--num-views",
        "4",
        "--corrupt-view-index",
        corrupt_index,
    )

    assert result.returncode == 2
    assert "corrupt-view-index" in result.stderr


@pytest.mark.parametrize(
    "unsupported", ["--flow-checkpoint", "--device", "--model"]
)
def test_cli_has_no_flow_gpu_or_model_arguments(tmp_path, unsupported):
    result = _run_cli(
        "--transforms",
        str(tmp_path / "transforms.json"),
        "--output-dir",
        str(tmp_path / "output"),
        "--mesh-scale",
        "1.0",
        unsupported,
        "forbidden",
    )

    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
    assert unsupported in result.stderr


@pytest.mark.parametrize("mesh_scale", ["0", "-1", "nan", "inf"])
def test_cli_parser_requires_finite_positive_mesh_scale(tmp_path, mesh_scale):
    result = _run_cli(
        "--transforms",
        str(tmp_path / "transforms.json"),
        "--output-dir",
        str(tmp_path / "output"),
        "--mesh-scale",
        mesh_scale,
    )

    assert result.returncode == 2
    assert "--mesh-scale" in result.stderr
    assert "finite and positive" in result.stderr


@pytest.mark.parametrize("seed", [str(-(2**63) - 1), str(2**64)])
def test_cli_parser_rejects_seed_outside_torch_range(tmp_path, seed):
    result = _run_cli(
        "--transforms",
        str(tmp_path / "transforms.json"),
        "--output-dir",
        str(tmp_path / "output"),
        "--mesh-scale",
        "1.0",
        "--seed",
        seed,
    )

    assert result.returncode == 2
    assert "--seed" in result.stderr
    assert "PyTorch seed range" in result.stderr


@pytest.mark.parametrize("seed", [str(-(2**63)), str(2**64 - 1)])
def test_cli_accepts_both_inclusive_torch_seed_bounds(tmp_path, seed):
    output = tmp_path / "output"
    result = _run_cli(
        "--transforms",
        str(tmp_path / "missing-transforms.json"),
        "--output-dir",
        str(output),
        "--mesh-scale",
        "1.0",
        "--seed",
        seed,
    )

    assert result.returncode == 1
    assert "transforms path must name a manifest file" in result.stderr
    assert "PyTorch seed range" not in result.stderr
    _assert_failed_bundle(
        output, failure_text="transforms path must name a manifest file"
    )


def test_cli_builds_and_resumes_complete_synthetic_calibrated_bundle(tmp_path):
    transforms, frames, source_paths = _write_synthetic_dataset(
        tmp_path / "dataset"
    )
    output = tmp_path / "output"
    command = (
        "--transforms",
        str(transforms),
        "--output-dir",
        str(output),
        "--mesh-scale",
        "1.0",
    )

    first = _run_cli(*command)

    assert first.returncode == 0, first.stdout + first.stderr
    run_directories = [path for path in output.iterdir() if path.is_dir()]
    assert len(run_directories) == 1
    run_dir = run_directories[0]
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = _artifact_records(manifest)
    before = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in [manifest_path, *(run_dir / record["path"] for record in records)]
    }

    second = _run_cli(*command)

    assert second.returncode == 0, second.stdout + second.stderr
    assert {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in before
    } == before
    assert manifest["status"] == "completed"
    assert manifest["failure_reason"] is None
    assert manifest["K"] == 4
    assert manifest["seed"] == 42
    assert manifest["mesh_scale"] == 1.0
    assert [view["view_index"] for view in manifest["views"]] == [0, 1, 2, 3]
    assert [view["frame_index"] for view in manifest["views"]] == [0, 1, 2, 3]
    assert [view["camera_angle_x"] for view in manifest["views"]] == pytest.approx(
        [0.71, 0.65, 0.73, 0.65]
    )
    assert [view["distance"] for view in manifest["views"]] == pytest.approx(
        [2.0, 5.0, 3.0, math.sqrt(81.0)]
    )
    assert [view["transform_matrix"] for view in manifest["views"]] == [
        frame["transform_matrix"] for frame in frames
    ]
    assert [view["source"]["input_sha256"] for view in manifest["views"]] == [
        sha256(path.read_bytes()).hexdigest() for path in source_paths
    ]
    assert [
        view["foreground_mask"]["provenance"] for view in manifest["views"]
    ] == ["explicit", "alpha", "alpha", "alpha"]

    for record in records:
        artifact = run_dir / record["path"]
        assert artifact.is_file()
        assert sha256(artifact.read_bytes()).hexdigest() == record["sha256"]
    for view in manifest["views"]:
        mask = np.asarray(Image.open(run_dir / view["foreground_mask"]["path"]))
        assert np.any(mask)

    assert [item["arm"] for item in manifest["corruptions"]] == ["c1", "c2", "c3"]
    assert [item["view_index"] for item in manifest["corruptions"]] == [1, 1, 1]
    source = np.asarray(Image.open(source_paths[1]).convert("RGB"), dtype=np.int16)
    for item in manifest["corruptions"]:
        corrupted = np.asarray(
            Image.open(run_dir / item["image"]["path"]).convert("RGB"),
            dtype=np.int16,
        )
        oracle = np.asarray(Image.open(run_dir / item["oracle_mask"]["path"])) > 0
        changed_over_threshold = np.any(np.abs(corrupted - source) > 1, axis=-1)
        assert np.any(oracle)
        assert np.array_equal(oracle, changed_over_threshold)

    with Image.open(run_dir / manifest["contact_sheet"]["path"]) as contact:
        assert contact.size == (808, 680)


def _assert_failed_bundle(output: Path, *, failure_text: str) -> Path:
    run_directories = [path for path in output.iterdir() if path.is_dir()]
    assert len(run_directories) == 1
    run_dir = run_directories[0]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert failure_text in manifest["failure_reason"]
    assert manifest["views"] == []
    assert manifest["corruptions"] == []
    contact_record = manifest["contact_sheet"]
    contact_path = run_dir / contact_record["path"]
    assert sha256(contact_path.read_bytes()).hexdigest() == contact_record["sha256"]
    with Image.open(contact_path) as contact:
        assert contact.size == (808, 128)
    return run_dir


def test_cli_publishes_failed_manifest_for_empty_foreground(tmp_path):
    transforms, _, image_paths = _write_synthetic_dataset(tmp_path / "dataset")
    with Image.open(image_paths[1]) as source:
        pixels = np.asarray(source).copy()
    pixels[..., 3] = 0
    Image.fromarray(pixels, mode="RGBA").save(image_paths[1])
    output = tmp_path / "output"

    command = (
        "--transforms",
        str(transforms),
        "--output-dir",
        str(output),
        "--mesh-scale",
        "1.0",
    )
    result = _run_cli(*command)

    assert result.returncode == 1
    assert "foreground mask is empty" in result.stderr
    failed_dir = _assert_failed_bundle(
        output, failure_text="foreground mask is empty"
    )

    pixels[5:35, 7:41, 3] = 255
    Image.fromarray(pixels, mode="RGBA").save(image_paths[1])
    recovered = _run_cli(*command)

    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    run_directories = [path for path in output.iterdir() if path.is_dir()]
    assert len(run_directories) == 2
    assert failed_dir in run_directories
    statuses = {
        json.loads((path / "manifest.json").read_text())["status"]
        for path in run_directories
    }
    assert statuses == {"failed", "completed"}


def test_cli_publishes_failed_manifest_for_input_path_escape(tmp_path):
    transforms, frames, _ = _write_synthetic_dataset(tmp_path / "dataset")
    outside = tmp_path / "outside.png"
    Image.new("RGBA", (48, 40), (1, 2, 3, 255)).save(outside)
    frames[0]["file_path"] = f"../{outside.name}"
    transforms.write_text(json.dumps({"camera_angle_x": 0.65, "frames": frames}))
    output = tmp_path / "output"

    result = _run_cli(
        "--transforms",
        str(transforms),
        "--output-dir",
        str(output),
        "--mesh-scale",
        "1.0",
    )

    assert result.returncode == 1
    assert "inside the manifest directory" in result.stderr
    _assert_failed_bundle(output, failure_text="inside the manifest directory")


def test_cli_publishes_failed_manifest_for_insufficient_k(tmp_path):
    transforms, _, _ = _write_synthetic_dataset(tmp_path / "dataset")
    output = tmp_path / "output"

    result = _run_cli(
        "--transforms",
        str(transforms),
        "--output-dir",
        str(output),
        "--mesh-scale",
        "1.0",
        "--num-views",
        "5",
    )

    assert result.returncode == 1
    assert "at least 5 frames" in result.stderr
    _assert_failed_bundle(output, failure_text="at least 5 frames")


def test_cli_fails_closed_when_resumed_artifact_hash_mismatches(tmp_path):
    transforms, _, _ = _write_synthetic_dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    command = (
        "--transforms",
        str(transforms),
        "--output-dir",
        str(output),
        "--mesh-scale",
        "1.0",
    )
    first = _run_cli(*command)
    assert first.returncode == 0, first.stdout + first.stderr
    run_dir = next(path for path in output.iterdir() if path.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    corrupted_path = run_dir / manifest["corruptions"][0]["image"]["path"]
    corrupted_path.write_bytes(b"tampered")

    second = _run_cli(*command)

    assert second.returncode == 1
    assert "artifact hash mismatch" in second.stderr
    assert corrupted_path.read_bytes() == b"tampered"
