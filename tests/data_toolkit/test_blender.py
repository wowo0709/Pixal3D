from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from urllib.request import Request

import pandas as pd
from PIL import Image
import pytest

from data_toolkit import render_cond
from data_toolkit.pipeline import blender
from data_toolkit.pipeline.blender import ensure_blender, verify_archive
from data_toolkit.pipeline.camera import build_condition_views


def test_checksum_verification(tmp_path):
    path = tmp_path / "blender.tar.xz"
    path.write_bytes(b"fixture")

    verify_archive(path, sha256(b"fixture").hexdigest())
    with pytest.raises(ValueError, match="checksum"):
        verify_archive(path, "0" * 64)


def _blender_archive(contents: dict[str, bytes]) -> bytes:
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w:xz") as archive:
        for name, value in contents.items():
            member = tarfile.TarInfo(name)
            member.mode = 0o755
            member.size = len(value)
            archive.addfile(member, BytesIO(value))
    return stream.getvalue()


def test_installer_uses_pinned_verified_archive(monkeypatch, tmp_path):
    payload = _blender_archive(
        {
            f"{blender.BLENDER_DIR}/blender": b"blender fixture",
            f"{blender.BLENDER_DIR}/4.5/python/bin/python3.11": (
                b"#!/bin/sh\nexit 0\n"
            ),
        }
    )
    requests = []

    def open_fixture(request):
        requests.append(request)
        return BytesIO(payload)

    monkeypatch.setattr(blender, "urlopen", open_fixture)
    monkeypatch.setattr(blender, "BLENDER_SHA256", sha256(payload).hexdigest())

    binary = ensure_blender(tmp_path)

    assert [request.full_url for request in requests] == [blender.BLENDER_URL]
    assert binary == tmp_path / blender.BLENDER_DIR / "blender"
    assert binary.read_bytes() == b"blender fixture"
    assert not list(tmp_path.glob("*.part"))


def test_installer_sends_explicit_user_agent(monkeypatch, tmp_path):
    payload = _blender_archive(
        {
            f"{blender.BLENDER_DIR}/blender": b"blender fixture",
            f"{blender.BLENDER_DIR}/4.5/python/bin/python3.11": (
                b"#!/bin/sh\nexit 0\n"
            ),
        }
    )

    def require_user_agent(request):
        assert isinstance(request, Request)
        assert request.full_url == blender.BLENDER_URL
        assert request.get_header("User-agent") == "Pixal3D-data-toolkit/1"
        return BytesIO(payload)

    monkeypatch.setattr(blender, "urlopen", require_user_agent)
    monkeypatch.setattr(blender, "BLENDER_SHA256", sha256(payload).hexdigest())

    ensure_blender(tmp_path)


def test_existing_blender_installs_pinned_pillow_when_missing(tmp_path):
    blender_root = tmp_path / blender.BLENDER_DIR
    binary = blender_root / "blender"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"blender fixture")
    bundled_python = blender_root / "4.5/python/bin/python3.11"
    bundled_python.parent.mkdir(parents=True)
    marker = tmp_path / "pillow-installed"
    bundled_python.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        f"marker = Path({str(marker)!r})\n"
        "args = sys.argv[1:]\n"
        "if args == ['-c', 'from PIL import Image']:\n"
        "    raise SystemExit(0 if marker.exists() else 1)\n"
        "if args[:2] == ['-m', 'ensurepip']:\n"
        "    raise SystemExit(0)\n"
        "if args[:2] == ['-m', 'pip'] and 'Pillow==12.3.0' in args:\n"
        "    marker.touch()\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(2)\n"
    )
    bundled_python.chmod(0o755)

    assert ensure_blender(tmp_path) == binary
    assert marker.is_file()


def _write_render_fixture(
    output: Path,
    num_views: int,
    resolution: int = 512,
    selected_devices: list[str] | None = None,
) -> None:
    if selected_devices is None:
        selected_devices = ["GPU fixture"]
    frames = []
    for index in range(num_views):
        name = f"{index:03d}.png"
        image = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
        image.putpixel((resolution // 2, resolution // 2), (255, 255, 255, 255))
        image.save(output / name)
        frames.append(
            {
                "file_path": name,
                "camera_angle_x": 0.7,
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "radius": 2.0,
            }
        )
    (output / "transforms.json").write_text(
        json.dumps({"frames": frames, "selected_devices": selected_devices})
    )


def test_render_uses_config_and_atomically_publishes(
    monkeypatch, tmp_path, config
):
    sha = "a" * 64
    render_root = tmp_path / "render"
    final = render_root / "renders_cond" / sha
    final.mkdir(parents=True)
    (final / "stale").write_text("old")
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        output = Path(args[args.index("--cond_output_folder") + 1])
        assert output.parent == final.parent
        assert output != final
        assert (final / "stale").is_file()
        _write_render_fixture(output, config.render.num_views)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(render_cond.subprocess, "run", fake_run)

    result = render_cond._render_cond(
        "fixture.glb",
        sha,
        root=render_root,
        config=config.render,
        blender_path=Path("/tools/blender"),
        timeout_seconds=37,
    )

    args, kwargs = calls[0]
    assert args[0] == "/tools/blender"
    assert args[args.index("--cond_resolution") + 1] == "512"
    assert args[args.index("--cycles_device") + 1] == "OPTIX"
    assert json.loads(args[args.index("--cond_views") + 1]) == (
        build_condition_views(sha, config.render)
    )
    assert kwargs["check"] is True
    assert kwargs["timeout"] == 37
    assert result == {"sha256": sha, "cond_rendered": True}
    assert not (final / "stale").exists()
    assert len(list(final.glob("*.png"))) == config.render.num_views


def test_existing_render_is_replaced_with_atomic_exchange(monkeypatch, tmp_path):
    temporary = tmp_path / ".temporary"
    final = tmp_path / "final"
    temporary.mkdir()
    final.mkdir()
    (temporary / "value").write_text("new")
    (final / "value").write_text("old")
    exchange = render_cond._rename_exchange
    calls = []

    def tracked_exchange(left, right):
        calls.append((left, right))
        exchange(left, right)

    monkeypatch.setattr(render_cond, "_rename_exchange", tracked_exchange)
    render_cond._publish_render_output(temporary, final)

    assert calls == [(temporary, final)]
    assert (final / "value").read_text() == "new"
    assert not temporary.exists()


def test_render_keeps_existing_output_when_validation_fails(
    monkeypatch, tmp_path, config
):
    sha = "b" * 64
    render_root = tmp_path / "render"
    final = render_root / "renders_cond" / sha
    final.mkdir(parents=True)
    (final / "stale").write_text("old")

    def fake_run(args, **kwargs):
        output = Path(args[args.index("--cond_output_folder") + 1])
        (output / "transforms.json").write_text(
            '{"frames": [], "selected_devices": ["GPU fixture"]}'
        )
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(render_cond.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="frames"):
        render_cond._render_cond(
            "fixture.glb",
            sha,
            root=render_root,
            config=config.render,
            blender_path=Path("/tools/blender"),
            timeout_seconds=37,
        )

    assert (final / "stale").read_text() == "old"
    assert sorted(path.name for path in final.parent.iterdir()) == [sha]


def test_exchange_failure_keeps_existing_output_and_cleans_temporary(
    monkeypatch, tmp_path, config
):
    sha = "f" * 64
    render_root = tmp_path / "render"
    final = render_root / "renders_cond" / sha
    final.mkdir(parents=True)
    (final / "stale").write_text("old")

    def fake_run(args, **kwargs):
        output = Path(args[args.index("--cond_output_folder") + 1])
        _write_render_fixture(
            output, config.render.num_views, config.render.resolution
        )
        return subprocess.CompletedProcess(args, 0)

    def fail_exchange(temporary, destination):
        assert (temporary / "000.png").is_file()
        assert (destination / "stale").is_file()
        raise OSError("renameat2 unavailable")

    monkeypatch.setattr(render_cond.subprocess, "run", fake_run)
    monkeypatch.setattr(
        render_cond, "_rename_exchange", fail_exchange, raising=False
    )

    with pytest.raises(OSError, match="renameat2 unavailable"):
        render_cond._render_cond(
            "fixture.glb",
            sha,
            root=render_root,
            config=config.render,
            blender_path=Path("/tools/blender"),
            timeout_seconds=37,
        )

    assert (final / "stale").read_text() == "old"
    assert sorted(path.name for path in final.parent.iterdir()) == [sha]


def test_render_main_passes_download_root_to_adapter(
    monkeypatch, tmp_path
):
    source_root = tmp_path / "source"
    download_root = tmp_path / "download"
    render_root = tmp_path / "render"
    source_root.mkdir()
    pd.DataFrame(
        [{"sha256": "c" * 64, "local_path": "raw/fixture.glb"}]
    ).to_csv(source_root / "metadata.csv", index=False)
    calls = []

    def foreach_instance(metadata, output_dir, func, **kwargs):
        calls.append((metadata, output_dir, func, kwargs))
        return pd.DataFrame(columns=["sha256", "cond_rendered"])

    adapter = SimpleNamespace(
        add_args=lambda parser: None,
        foreach_instance=foreach_instance,
    )
    monkeypatch.setattr(render_cond, "_import_adapter", lambda name: adapter)

    render_cond.main(
        [
            "fixture",
            "--root",
            str(source_root),
            "--download_root",
            str(download_root),
            "--render_cond_root",
            str(render_root),
            "--blender_path",
            "/tools/blender",
            "--cond_resolution",
            "256",
            "--cycles_device",
            "OPTIX",
            "--timeout_seconds",
            "41",
            "--record_prefix",
            "chunk007_",
        ]
    )

    _, output_dir, func, kwargs = calls[0]
    assert output_dir == str(download_root)
    assert func.keywords["root"] == str(render_root)
    assert func.keywords["config"].resolution == 256
    assert func.keywords["config"].cycles_device == "OPTIX"
    assert func.keywords["blender_path"] == Path("/tools/blender")
    assert func.keywords["timeout_seconds"] == 41
    assert kwargs["max_workers"] == 8
    assert (
        render_root
        / "renders_cond/new_records/chunk007_part_0.csv"
    ).is_file()


def test_render_main_rejects_record_prefix_path_separators(
    monkeypatch, tmp_path
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    pd.DataFrame(
        [{"sha256": "e" * 64, "local_path": "raw/fixture.glb"}]
    ).to_csv(source_root / "metadata.csv", index=False)
    adapter = SimpleNamespace(
        add_args=lambda parser: None,
        foreach_instance=lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(render_cond, "_import_adapter", lambda name: adapter)

    with pytest.raises(ValueError, match="record prefix"):
        render_cond.main(
            [
                "fixture",
                "--root",
                str(source_root),
                "--blender_path",
                "/tools/blender",
                "--record_prefix",
                "../chunk",
            ]
        )


def test_render_main_defaults_to_eight_deterministic_views(
    monkeypatch, tmp_path
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    sha = "d" * 64
    pd.DataFrame(
        [{"sha256": sha, "local_path": "raw/fixture.glb"}]
    ).to_csv(source_root / "metadata.csv", index=False)
    rendered_views = []

    def fake_render(file_path, asset_sha, **kwargs):
        first = build_condition_views(asset_sha, kwargs["config"])
        rendered_views.append(first)
        assert first == build_condition_views(asset_sha, kwargs["config"])
        return {"sha256": asset_sha, "cond_rendered": True}

    def foreach_instance(metadata, output_dir, func, **kwargs):
        record = metadata.iloc[0]
        return pd.DataFrame(
            [func(record["local_path"], record["sha256"])]
        )

    adapter = SimpleNamespace(
        add_args=lambda parser: None,
        foreach_instance=foreach_instance,
    )
    monkeypatch.setattr(render_cond, "_import_adapter", lambda name: adapter)
    monkeypatch.setattr(render_cond, "_render_cond", fake_render)

    render_cond.main(
        [
            "fixture",
            "--root",
            str(source_root),
            "--blender_path",
            "/tools/blender",
        ]
    )

    assert len(rendered_views[0]) == 8
    assert len([view["radius"] for view in rendered_views[0]]) == 8


def test_resume_only_skips_valid_render_directories(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    render_root = tmp_path / "render"
    renders = render_root / "renders_cond"
    source_root.mkdir()
    renders.mkdir(parents=True)
    names = {
        "valid": "1" * 64,
        "partial": "2" * 64,
        "two_view": "3" * 64,
        "metadata_free": "4" * 64,
        "wrong_resolution": "5" * 64,
        "device_free": "6" * 64,
        "transform_free": "7" * 64,
    }
    pd.DataFrame(
        [
            {
                "sha256": sha,
                "local_path": f"raw/{name}.glb",
                "cond_rendered": True,
            }
            for name, sha in names.items()
        ]
    ).to_csv(source_root / "metadata.csv", index=False)

    for sha in names.values():
        (renders / sha).mkdir()
    _write_render_fixture(renders / names["valid"], 8)
    valid_previous = renders / f'.{names["valid"]}.previous'
    valid_previous.mkdir()
    (valid_previous / "stale").write_text("old")

    _write_render_fixture(renders / names["partial"], 8)
    (renders / names["partial"] / "007.png").unlink()
    partial_previous = renders / f'.{names["partial"]}.previous'
    partial_previous.mkdir()
    (partial_previous / "stale").write_text("old")

    _write_render_fixture(renders / names["two_view"], 2)
    for index in range(8):
        Image.new("RGBA", (512, 512)).save(
            renders / names["metadata_free"] / f"{index:03d}.png"
        )
    _write_render_fixture(
        renders / names["wrong_resolution"], 8, resolution=256
    )
    _write_render_fixture(renders / names["device_free"], 8)
    device_metadata = renders / names["device_free"] / "transforms.json"
    device_value = json.loads(device_metadata.read_text())
    device_value.pop("selected_devices")
    device_metadata.write_text(json.dumps(device_value))
    _write_render_fixture(renders / names["transform_free"], 8)
    transform_metadata = renders / names["transform_free"] / "transforms.json"
    transform_value = json.loads(transform_metadata.read_text())
    transform_value["frames"][0].pop("transform_matrix")
    transform_metadata.write_text(json.dumps(transform_value))
    processed = []

    def foreach_instance(metadata, output_dir, func, **kwargs):
        processed.extend(metadata["sha256"].tolist())
        return metadata[["sha256"]].assign(cond_rendered=True)

    adapter = SimpleNamespace(
        add_args=lambda parser: None,
        foreach_instance=foreach_instance,
    )
    monkeypatch.setattr(render_cond, "_import_adapter", lambda name: adapter)

    render_cond.main(
        [
            "fixture",
            "--root",
            str(source_root),
            "--render_cond_root",
            str(render_root),
            "--blender_path",
            "/tools/blender",
        ]
    )

    assert set(processed) == set(names.values()) - {names["valid"]}
    assert not valid_previous.exists()
    assert (partial_previous / "stale").read_text() == "old"


def test_blender_script_selects_gpu_and_scales_boundary():
    repository = Path(__file__).resolve().parents[2]
    source = (
        repository / "data_toolkit/blender_script/render_cond.py"
    ).read_text()

    assert "preferences.compute_device_type = arg.cycles_device" in source
    assert "device.use = device.type == arg.cycles_device" in source
    assert '"selected_devices": selected_devices' in source
    assert "130 * arg.cond_resolution / 1024" in source
    assert 'parser.add_argument("--cycles_device"' in source


def test_blender_render_script_uses_version_aware_obj_importer():
    repository = Path(__file__).resolve().parents[2]
    source = (
        repository / "data_toolkit/blender_script/render_cond.py"
    ).read_text()

    assert (
        '"obj": bpy.ops.import_scene.obj if bpy.app.version[0] < 4 '
        'else bpy.ops.wm.obj_import'
    ) in source


@pytest.mark.parametrize(
    "invocation",
    [["data_toolkit/render_cond.py"], ["-m", "data_toolkit.render_cond"]],
)
def test_render_imports_work_in_script_and_module_modes(invocation):
    repository = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, *invocation, "ABO", "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--blender_path" in result.stdout
    assert "--timeout_seconds" in result.stdout
