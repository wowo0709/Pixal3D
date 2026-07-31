from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/generate_node16_deployment_manifest.py"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _committed_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "deployment-test@example.invalid")
    _git(repo, "config", "user.name", "Deployment Test")
    _git(repo, "config", "core.autocrlf", "false")

    (repo / ".gitattributes").write_bytes(b"excluded.txt export-ignore\n")
    (repo / "plain.txt").write_bytes(b"reviewed\n")
    executable = repo / "bin" / "launch"
    executable.parent.mkdir()
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    (repo / "excluded.txt").write_bytes(b"not in the archive\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "reviewed deployment")
    return repo, _git(repo, "rev-parse", "HEAD")


def _run_generator(
    repo: Path,
    revision: str,
    output: Path,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--revision",
            revision,
            "--output",
            str(output),
        ],
        cwd=output.parent,
        env=env,
        capture_output=True,
        text=True,
    )


def _environment_with_archive(
    tmp_path: Path,
    archive_payload: bytes,
    *,
    exit_code: int = 0,
) -> dict[str, str]:
    real_git = shutil.which("git")
    assert real_git is not None
    archive = tmp_path / "injected.tar"
    archive.write_bytes(archive_payload)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    wrapper = fake_bin / "git"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "if 'archive' in sys.argv:\n"
        f"    sys.stdout.buffer.write(Path({str(archive)!r}).read_bytes())\n"
        f"    if {exit_code}:\n"
        "        sys.stderr.write('injected git archive failure\\n')\n"
        f"    raise SystemExit({exit_code})\n"
        f"os.execv({real_git!r}, [{real_git!r}, *sys.argv[1:]])\n"
    )
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join((str(fake_bin), env["PATH"]))
    return env


def _single_file_tar(path: str, payload: bytes) -> bytes:
    target = BytesIO()
    with tarfile.open(fileobj=target, mode="w:") as archive:
        member = tarfile.TarInfo(path)
        member.mode = 0o644
        member.size = len(payload)
        archive.addfile(member, BytesIO(payload))
    return target.getvalue()


def _single_symlink_tar(path: str, linkname: str) -> bytes:
    target = BytesIO()
    with tarfile.open(fileobj=target, mode="w:") as archive:
        member = tarfile.TarInfo(path)
        member.type = tarfile.SYMTYPE
        member.linkname = linkname
        archive.addfile(member)
    return target.getvalue()


def _duplicate_file_tar(path: str) -> bytes:
    target = BytesIO()
    with tarfile.open(fileobj=target, mode="w:") as archive:
        for payload in (b"first\n", b"second\n"):
            member = tarfile.TarInfo(path)
            member.mode = 0o644
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))
    return target.getvalue()


def test_generator_binds_exact_committed_archive_not_dirty_checkout(tmp_path):
    repo, revision = _committed_repo(tmp_path)
    (repo / "plain.txt").write_bytes(b"dirty checkout\n")
    (repo / "untracked.txt").write_bytes(b"not reviewed\n")
    output = tmp_path / "deployment-manifest.json"

    result = _run_generator(repo, revision, output)

    assert result.returncode == 0, result.stderr
    expected = {
        "schema_version": 1,
        "revision": revision,
        "hash_algorithm": "sha256",
        "files": [
            {
                "path": ".gitattributes",
                "mode": "100644",
                "size": 27,
                "sha256": sha256(
                    b"excluded.txt export-ignore\n"
                ).hexdigest(),
            },
            {
                "path": "bin/launch",
                "mode": "100755",
                "size": 17,
                "sha256": sha256(b"#!/bin/sh\nexit 0\n").hexdigest(),
            },
            {
                "path": "plain.txt",
                "mode": "100644",
                "size": 9,
                "sha256": sha256(b"reviewed\n").hexdigest(),
            },
        ],
    }
    assert json.loads(output.read_bytes()) == expected
    assert output.read_bytes() == (
        json.dumps(expected, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    archive = tmp_path / "reviewed.tar"
    _git(
        repo,
        "archive",
        "--format=tar",
        f"--output={archive}",
        revision,
    )
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:") as stream:
        stream.extractall(extracted)
    assert not (extracted / ".git").exists()

    extracted_files = {
        path.relative_to(extracted).as_posix()
        for path in extracted.rglob("*")
        if path.is_file()
    }
    assert extracted_files == {record["path"] for record in expected["files"]}
    for record in expected["files"]:
        path = extracted / record["path"]
        raw = path.read_bytes()
        assert len(raw) == record["size"]
        assert sha256(raw).hexdigest() == record["sha256"]
        assert (
            "100755" if path.stat().st_mode & 0o111 else "100644"
        ) == record["mode"]


@pytest.mark.parametrize("mutation", ("symbolic", "abbreviated", "uppercase"))
def test_generator_requires_exact_lowercase_40_hex_revision(
    tmp_path, mutation
):
    repo, revision = _committed_repo(tmp_path)
    invalid = {
        "symbolic": "HEAD",
        "abbreviated": revision[:-1],
        "uppercase": revision.upper(),
    }[mutation]
    output = tmp_path / "deployment-manifest.json"

    result = _run_generator(repo, invalid, output)

    assert result.returncode != 0
    assert (
        "revision must be exactly 40 lowercase hexadecimal characters"
        in result.stderr
    )
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_generator_rejects_exact_object_id_that_is_not_a_commit(tmp_path):
    repo, revision = _committed_repo(tmp_path)
    tree_object = _git(repo, "rev-parse", f"{revision}^{{tree}}")
    output = tmp_path / "deployment-manifest.json"

    result = _run_generator(repo, tree_object, output)

    assert result.returncode != 0
    assert "revision must name a commit object" in result.stderr
    assert not output.exists()


def test_generator_rejects_output_inside_repository_checkout(tmp_path):
    repo, revision = _committed_repo(tmp_path)
    output = repo / "deployment-manifest.json"

    result = _run_generator(repo, revision, output)

    assert result.returncode != 0
    assert "output must be outside repository checkout" in result.stderr
    assert not output.exists()


def test_generator_requires_repository_worktree_root(tmp_path):
    repo, revision = _committed_repo(tmp_path)
    output = repo / "deployment-manifest.json"

    result = _run_generator(repo / "bin", revision, output)

    assert result.returncode != 0
    assert "repo root must be the Git working-tree root" in result.stderr
    assert not output.exists()


def test_generator_refuses_to_overwrite_existing_sidecar(tmp_path):
    repo, revision = _committed_repo(tmp_path)
    output = tmp_path / "deployment-manifest.json"
    output.write_bytes(b"existing sidecar\n")

    result = _run_generator(repo, revision, output)

    assert result.returncode != 0
    assert "refusing to overwrite existing output" in result.stderr
    assert output.read_bytes() == b"existing sidecar\n"


def test_generator_refuses_dangling_symlink_sidecar(tmp_path):
    repo, revision = _committed_repo(tmp_path)
    target = tmp_path / "symlink-target.json"
    output = tmp_path / "deployment-manifest.json"
    output.symlink_to(target)

    result = _run_generator(repo, revision, output)

    assert result.returncode != 0
    assert "refusing to overwrite existing output" in result.stderr
    assert output.is_symlink()
    assert not target.exists()


def test_generator_rejects_tracked_symlink(tmp_path):
    repo, _revision = _committed_repo(tmp_path)
    (repo / "linked.txt").symlink_to("plain.txt")
    _git(repo, "add", "linked.txt")
    _git(repo, "commit", "--quiet", "-m", "add unsupported symlink")
    revision = _git(repo, "rev-parse", "HEAD")
    output = tmp_path / "deployment-manifest.json"

    result = _run_generator(repo, revision, output)

    assert result.returncode != 0
    assert "unsupported nonregular Git entry" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "unsafe_path",
    ("../escape.txt", "/absolute.txt", "safe/../../escape.txt", "win\\path"),
)
def test_generator_rejects_unsafe_archive_paths(tmp_path, unsafe_path):
    repo, revision = _committed_repo(tmp_path)
    output = tmp_path / "deployment-manifest.json"
    env = _environment_with_archive(
        tmp_path, _single_file_tar(unsafe_path, b"malicious\n")
    )

    result = _run_generator(repo, revision, output, env=env)

    assert result.returncode != 0
    assert "unsafe archive path" in result.stderr
    assert not output.exists()


def test_generator_rejects_nonregular_archive_entry(tmp_path):
    repo, revision = _committed_repo(tmp_path)
    output = tmp_path / "deployment-manifest.json"
    env = _environment_with_archive(
        tmp_path, _single_symlink_tar("plain.txt", ".gitattributes")
    )

    result = _run_generator(repo, revision, output, env=env)

    assert result.returncode != 0
    assert "unsupported nonregular archive entry" in result.stderr
    assert not output.exists()


def test_generator_rejects_corrupt_git_archive(tmp_path):
    repo, revision = _committed_repo(tmp_path)
    output = tmp_path / "deployment-manifest.json"
    env = _environment_with_archive(tmp_path, b"not a tar archive")

    result = _run_generator(repo, revision, output, env=env)

    assert result.returncode != 0
    assert "invalid or corrupt git archive" in result.stderr
    assert not output.exists()


def test_generator_rejects_failed_git_archive_process(tmp_path):
    repo, revision = _committed_repo(tmp_path)
    output = tmp_path / "deployment-manifest.json"
    env = _environment_with_archive(tmp_path, b"", exit_code=7)

    result = _run_generator(repo, revision, output, env=env)

    assert result.returncode != 0
    assert "git archive failed: injected git archive failure" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_generator_rejects_duplicate_archive_path(tmp_path):
    repo, revision = _committed_repo(tmp_path)
    output = tmp_path / "deployment-manifest.json"
    env = _environment_with_archive(
        tmp_path, _duplicate_file_tar("plain.txt")
    )

    result = _run_generator(repo, revision, output, env=env)

    assert result.returncode != 0
    assert "duplicate archive path" in result.stderr
    assert not output.exists()


def test_generator_rejects_archive_file_absent_from_revision(tmp_path):
    repo, revision = _committed_repo(tmp_path)
    output = tmp_path / "deployment-manifest.json"
    env = _environment_with_archive(
        tmp_path, _single_file_tar("injected.txt", b"not committed\n")
    )

    result = _run_generator(repo, revision, output, env=env)

    assert result.returncode != 0
    assert "archive path is not a regular file in revision" in result.stderr
    assert not output.exists()
