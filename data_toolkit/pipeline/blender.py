from hashlib import sha256
import os
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen


BLENDER_URL = (
    "https://download.blender.org/release/Blender4.5/"
    "blender-4.5.1-linux-x64.tar.xz"
)
BLENDER_SHA256 = (
    "085a7ed4ed80c3cb66783bad76f236f39897de5d33884abd133e0c6db94c0f14"
)
BLENDER_DIR = "blender-4.5.1-linux-x64"


def verify_archive(path: Path, expected: str) -> None:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    if value.hexdigest() != expected:
        raise ValueError(f"Blender checksum mismatch: {path}")


def ensure_blender(tool_root: Path) -> Path:
    binary = tool_root / BLENDER_DIR / "blender"
    if binary.is_file():
        return binary

    tool_root.mkdir(parents=True, exist_ok=True)
    archive = tool_root / Path(BLENDER_URL).name
    partial = archive.with_suffix(archive.suffix + ".part")
    with urlopen(BLENDER_URL) as source, partial.open("wb") as target:
        shutil.copyfileobj(source, target, 8 * 1024 * 1024)
    os.replace(partial, archive)
    verify_archive(archive, BLENDER_SHA256)
    with tarfile.open(archive, "r:xz") as bundle:
        bundle.extractall(tool_root, filter="data")
    if not binary.is_file():
        raise FileNotFoundError(binary)
    return binary
