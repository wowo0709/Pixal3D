from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pandas as pd
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
        {f"{blender.BLENDER_DIR}/blender": b"blender fixture"}
    )
    requests = []

    def open_fixture(url):
        requests.append(url)
        return BytesIO(payload)

    monkeypatch.setattr(blender, "urlopen", open_fixture)
    monkeypatch.setattr(blender, "BLENDER_SHA256", sha256(payload).hexdigest())

    binary = ensure_blender(tmp_path)

    assert requests == [blender.BLENDER_URL]
    assert binary == tmp_path / blender.BLENDER_DIR / "blender"
    assert binary.read_bytes() == b"blender fixture"
    assert not list(tmp_path.glob("*.part"))


def _write_render_fixture(output: Path, num_views: int) -> None:
    frames = []
    for index in range(num_views):
        name = f"{index:03d}.png"
        (output / name).write_bytes(b"png")
        frames.append({"file_path": name})
    (output / "transforms.json").write_text(
        json.dumps({"frames": frames, "selected_devices": ["GPU fixture"]})
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
        (output / "transforms.json").write_text('{"frames": []}')
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


def test_blender_script_selects_gpu_and_scales_boundary():
    repository = Path(__file__).resolve().parents[2]
    source = (
        repository / "data_toolkit/blender_script/render_cond.py"
    ).read_text()

    assert "preferences.compute_device_type = arg.cycles_device" in source
    assert 'device.use = device.type != "CPU"' in source
    assert '"selected_devices": selected_devices' in source
    assert "130 * arg.cond_resolution / 1024" in source
    assert 'parser.add_argument("--cycles_device"' in source


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
