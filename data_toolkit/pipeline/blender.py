from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from urllib.request import Request, urlopen


BLENDER_URL = (
    "https://download.blender.org/release/Blender4.5/"
    "blender-4.5.1-linux-x64.tar.xz"
)
BLENDER_SHA256 = (
    "085a7ed4ed80c3cb66783bad76f236f39897de5d33884abd133e0c6db94c0f14"
)
BLENDER_DIR = "blender-4.5.1-linux-x64"
BLENDER_PYTHON = Path("4.5/python/bin/python3.11")
DOWNLOAD_USER_AGENT = "Pixal3D-data-toolkit/1"
PILLOW_REQUIREMENT = "Pillow==12.3.0"


def verify_archive(path: Path, expected: str) -> None:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    if value.hexdigest() != expected:
        raise ValueError(f"Blender checksum mismatch: {path}")


def ensure_blender(tool_root: Path) -> Path:
    binary = tool_root / BLENDER_DIR / "blender"
    if not binary.is_file():
        tool_root.mkdir(parents=True, exist_ok=True)
        archive = tool_root / Path(BLENDER_URL).name
        partial = archive.with_suffix(archive.suffix + ".part")
        request = Request(
            BLENDER_URL, headers={"User-Agent": DOWNLOAD_USER_AGENT}
        )
        with urlopen(request) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        os.replace(partial, archive)
        verify_archive(archive, BLENDER_SHA256)
        with tarfile.open(archive, "r:xz") as bundle:
            bundle.extractall(tool_root, filter="data")
        if not binary.is_file():
            raise FileNotFoundError(binary)

    bundled_python = binary.parent / BLENDER_PYTHON
    if not bundled_python.is_file():
        raise FileNotFoundError(bundled_python)
    probe = [str(bundled_python), "-c", "from PIL import Image"]
    result = subprocess.run(
        probe,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        subprocess.run(
            [str(bundled_python), "-m", "ensurepip"], check=True
        )
        subprocess.run(
            [
                str(bundled_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                PILLOW_REQUIREMENT,
            ],
            check=True,
        )
        subprocess.run(probe, check=True)
    return binary
