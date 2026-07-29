#!/usr/bin/env python3
"""Generate a deployment manifest from an exact committed Git archive."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile


_EXACT_REVISION = re.compile(r"[0-9a-f]{40}")


def _validate_archive_path(path: str) -> None:
    parts = path.split("/")
    windows_drive = bool(parts and re.fullmatch(r"[A-Za-z]:", parts[0]))
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or windows_drive
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"unsafe archive path: {path!r}")


def _git(repo_root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
    ).stdout


def _tree_modes(repo_root: Path, revision: str) -> dict[str, str]:
    raw = _git(
        repo_root,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        revision,
    )
    modes = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        raw_mode, raw_kind, _object_id = metadata.split(b" ", 2)
        path = raw_path.decode("utf-8", errors="surrogateescape")
        mode = raw_mode.decode("ascii")
        kind = raw_kind.decode("ascii")
        if mode not in {"100644", "100755"} or kind != "blob":
            raise ValueError(
                "unsupported nonregular Git entry "
                f"mode={mode} type={kind} path={path!r}"
            )
        modes[path] = mode
    return modes


def _archive_files(
    repo_root: Path,
    revision: str,
    modes: dict[str, str],
) -> list[dict[str, object]]:
    try:
        raw = _git(repo_root, "archive", "--format=tar", revision)
    except subprocess.CalledProcessError as error:
        detail = os.fsdecode(error.stderr).strip()
        if not detail:
            detail = f"exit status {error.returncode}"
        raise ValueError(f"git archive failed: {detail}") from error
    records = []
    seen_paths = set()
    try:
        with tarfile.open(fileobj=BytesIO(raw), mode="r:") as archive:
            for member in archive:
                _validate_archive_path(member.name)
                if member.name in seen_paths:
                    raise ValueError(
                        f"duplicate archive path: {member.name!r}"
                    )
                seen_paths.add(member.name)
                if member.isdir():
                    continue
                if not member.isreg():
                    raise ValueError(
                        "unsupported nonregular archive entry "
                        f"type={member.type!r} path={member.name!r}"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(
                        f"archive entry is not a file: {member.name}"
                    )
                mode = modes.get(member.name)
                if mode is None:
                    raise ValueError(
                        "archive path is not a regular file in revision: "
                        f"{member.name!r}"
                    )
                digest = sha256()
                size = 0
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
                if size != member.size:
                    raise ValueError(
                        "archive file size does not match header: "
                        f"{member.name!r}"
                    )
                records.append(
                    {
                        "path": member.name,
                        "mode": mode,
                        "size": size,
                        "sha256": digest.hexdigest(),
                    }
                )
    except tarfile.TarError as error:
        raise ValueError("invalid or corrupt git archive") from error
    return sorted(records, key=lambda record: str(record["path"]))


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _repository_root(path: Path) -> Path:
    candidate = path.resolve(strict=True)
    try:
        raw_root = _git(candidate, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"repo root must be the Git working-tree root: {candidate}"
        ) from error
    if not raw_root.endswith(b"\n"):
        raise ValueError(
            f"repo root must be the Git working-tree root: {candidate}"
        )
    discovered = Path(os.fsdecode(raw_root[:-1])).resolve(strict=True)
    if candidate != discovered:
        raise ValueError(
            f"repo root must be the Git working-tree root: {candidate}"
        )
    return candidate


def _sidecar_path(path: Path) -> Path:
    absolute = path if path.is_absolute() else Path.cwd() / path
    if absolute.name in {"", ".", ".."}:
        raise ValueError(f"invalid output sidecar path: {path}")
    parent = absolute.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError(
            f"output sidecar parent must be a directory: {parent}"
        )
    return parent / absolute.name


def _exclusive_create(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing output: {path}"
        ) from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def generate_manifest(
    repo_root: Path,
    revision: str,
    output: Path,
) -> None:
    repo_root = _repository_root(repo_root)
    output = _sidecar_path(output)
    if output == repo_root or repo_root in output.parents:
        raise ValueError("output must be outside repository checkout")
    if _EXACT_REVISION.fullmatch(revision) is None:
        raise ValueError(
            "revision must be exactly 40 lowercase hexadecimal characters"
        )
    if _git(repo_root, "cat-file", "-t", revision).strip() != b"commit":
        raise ValueError("revision must name a commit object")
    modes = _tree_modes(repo_root, revision)
    files = _archive_files(repo_root, revision, modes)
    payload = _canonical_json_bytes(
        {
            "schema_version": 1,
            "revision": revision,
            "hash_algorithm": "sha256",
            "files": files,
        }
    )
    _exclusive_create(output, payload)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a revision-bound SHA-256 deployment manifest."
        )
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        generate_manifest(args.repo_root, args.revision, args.output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
