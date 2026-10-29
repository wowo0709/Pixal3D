"""Generate one appearance arm on fixed Selected common-501 geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def run(command: list[str], *, log: Path | None = None,
        env: dict[str, str] | None = None, timeout: int | None = None) -> None:
    if log is None:
        subprocess.run(command, check=True, env=env, timeout=timeout)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(command, check=True, env=env, timeout=timeout,
                       stdout=stream, stderr=subprocess.STDOUT)


def retryable_export_failure(log: str) -> bool:
    return ("export/untouched metric mesh differ in world space" in log
            or "fast simplifier produced" in log
            or "fast simplifier did not reach declared render budget" in log
            or ("RuntimeError: [CuMesh] CUDA error" in log
                and ("Error code: 9" in log
                     or "invalid configuration argument" in log)))


def ssh_commands(socket: Path, port: int) -> tuple[list[str], str]:
    if not socket.is_socket():
        raise FileNotFoundError("active direct-file4 control socket required")
    ssh = ["ssh", "-S", str(socket), "-o", "BatchMode=yes", "-p", str(port)]
    return ssh, " ".join(ssh)


def stage_shape(entry: dict, arm: str, host: str, ssh: list[str],
                rsync_ssh: str) -> Path:
    root = Path(entry["vanilla_shape_root"] if arm == "vanilla"
                else entry["topk_shape_root"])
    names = ("prediction.npz", "geometry-evaluation.npz", "generation.json",
             "archive-receipt.json")
    if not root.exists():
        root.mkdir(parents=True)
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        sources = [f"{host}:{entry['remote_shape']}/{name}" for name in missing]
        run(["rsync", "-a", "-e", rsync_ssh, *sources, str(root) + "/"])
    receipt = json.loads((root / "archive-receipt.json").read_text())
    if (receipt.get("status") != "PASS_FILE4_ARCHIVE_CHECKSUM_VERIFIED"
            or receipt.get("remote") != entry["remote_shape"]):
        raise ValueError("Selected shape archive receipt differs: " + entry["sample_id"])
    for name in names[:3]:
        expected = receipt.get("sha256", {}).get(name)
        if not expected or sha(root / name) != expected:
            raise ValueError("Selected shape checksum differs: " + name)
    return root


def ensure_export(repo: Path, python: Path, benchmark_repo: Path, shape: Path,
                  prediction_root: Path, arm: str, sid: str,
                  log_root: Path, env: dict[str, str]) -> None:
    glb = prediction_root / "benchmark.glb"
    report_path = prediction_root / "benchmark.json"
    if glb.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text())
        if (report.get("status") != "PASS_OURS_FROZEN_GT_SCALE_FREE_1024_EXPORT"
                or report.get("sample_id") != sid
                or report.get("texture_arm") != arm
                or report.get("prediction_sha256") != sha(prediction_root / "prediction.npz")
                or report.get("glb_sha256") != sha(glb)):
            raise ValueError("existing appearance export differs: " + sid)
        return
    if glb.exists() or report_path.exists():
        raise FileExistsError("partial appearance export requires inspection: " + sid)
    for budget in (250_000, 500_000, 1_000_000, 2_000_000, 5_000_000):
        command = [str(python), "-u", "-m",
                   "experiments.deformnext_retargeting.export_minimal_test_mesh",
                   "--pixal3d-repo", str(repo), "--benchmark-repo", str(benchmark_repo),
                   "--prediction", str(prediction_root / "prediction.npz"),
                   "--shape-generation", str(shape / "generation.json"),
                   "--anchor-evaluation", str(prediction_root / "geometry-evaluation.npz"),
                   "--texture-arm", arm, "--render-face-budget", str(budget),
                   "--output", str(glb)]
        texture_generation = prediction_root / "generation.json"
        if texture_generation.is_file() and prediction_root != shape:
            command.extend(("--texture-generation", str(texture_generation)))
        log = log_root / f"{sid}-{budget}.log"
        try:
            run(command, log=log, env=env,
                timeout=3600 if budget >= 2_000_000 else 1200)
            break
        except subprocess.CalledProcessError:
            failure = log.read_text(errors="replace")
            if (budget == 5_000_000 or glb.exists() or report_path.exists()
                    or not retryable_export_failure(failure)):
                raise
    report = json.loads(report_path.read_text())
    if (report.get("sample_id") != sid or report.get("texture_arm") != arm
            or report.get("glb_sha256") != sha(glb)):
        raise ValueError("new appearance export receipt differs: " + sid)


def archive_case(root: Path, names: list[str], host: str, remote: str,
                 ssh: list[str], rsync_ssh: str) -> dict:
    receipt_path = root / "archive-receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("status") != "PASS_FILE4_APPEARANCE_ARCHIVE_CHECKSUM_VERIFIED"
                or receipt.get("remote") != remote):
            raise ValueError("existing appearance archive receipt differs")
        return receipt
    files = [root / name for name in names]
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("appearance case incomplete before archive")
    run([*ssh, host, "mkdir", "-p", remote])
    run(["rsync", "-a", "-e", rsync_ssh, *map(str, files), f"{host}:{remote}/"])
    check = subprocess.run(
        ["rsync", "-a", "--checksum", "--dry-run", "--itemize-changes",
         "-e", rsync_ssh, *map(str, files), f"{host}:{remote}/"],
        check=True, capture_output=True, text=True)
    if check.stdout.strip() or check.stderr.strip():
        raise ValueError("appearance file4 archive checksum differs")
    receipt = {
        "status": "PASS_FILE4_APPEARANCE_ARCHIVE_CHECKSUM_VERIFIED",
        "remote": remote,
        "sha256": {path.name: sha(path) for path in files},
        "large_local_generated_arrays_reclaimed_after_archive": True,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    run(["rsync", "-a", "-e", rsync_ssh, str(receipt_path), f"{host}:{remote}/"])
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--arm", choices=("vanilla", "raw-topk-plus-feature"), required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--remote-host", default="rvi@10.0.12.170")
    parser.add_argument("--remote-output-base", required=True)
    parser.add_argument("--ssh-port", type=int, default=55555)
    parser.add_argument("--control-socket", type=Path, required=True)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--allowed-gpus", default="0,1,2,3,6,7")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--max-cases", type=int, default=0,
                        help="mechanical pilot limit; zero runs the complete shard")
    parser.add_argument("--reverse", action="store_true",
                        help="consume this rank from the tail for a disjoint helper")
    parser.add_argument("--skip-cases", type=int, default=0,
                        help="skip an initial slice after rank/reverse ordering")
    args = parser.parse_args()
    if (not args.runtime_python.is_file() or not args.physical_gpu.isdigit()
            or args.physical_gpu not in args.allowed_gpus.split(",")
            or not 0 <= args.rank < args.world_size
            or args.max_cases < 0 or args.skip_cases < 0
            or not args.remote_output_base.startswith(
                "/file4/youngwoo/deformation/experiments/retargeting-v1/")):
        raise ValueError("explicit runtime, authorized GPU and file4 output required")
    campaign = json.loads(args.campaign.read_text())
    if (campaign.get("status")
            != "PREPARED_COMMON501_PAIRED_NOISE_APPEARANCE_ABLATION_ON_SELECTED_GEOMETRY"
            or campaign.get("samples") != 501):
        raise ValueError("complete common-501 appearance campaign required")
    entries = campaign["plans"]
    ssh, rsync_ssh = ssh_commands(args.control_socket, args.ssh_port)
    repo = Path(__file__).resolve().parents[2]
    benchmark_repo = Path("/root/dev/.cache/deformnext-benchmark-runtime")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.physical_gpu,
               MATCHING_V2_ALLOWED_GPUS=args.allowed_gpus,
               BENCHMARK_ASSIGNED_GPU=args.physical_gpu, ATTN_BACKEND="sdpa",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OPENCV_IO_ENABLE_OPENEXR="1",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    log_root = args.campaign.parent / "logs" / args.arm / f"rank{args.rank:02d}"

    indices = list(range(args.rank, len(entries), args.world_size))
    if args.reverse:
        indices.reverse()
    if args.skip_cases:
        indices = indices[args.skip_cases:]
    if args.max_cases:
        indices = indices[:args.max_cases]
    for index in indices:
        entry = entries[index]
        sid = entry["sample_id"]
        case_root = Path(entry["vanilla_root"] if args.arm == "vanilla"
                         else entry["topk_root"])
        final_receipt = case_root / "archive-receipt.json"
        if final_receipt.is_file():
            receipt = json.loads(final_receipt.read_text())
            if receipt.get("status") != "PASS_FILE4_APPEARANCE_ARCHIVE_CHECKSUM_VERIFIED":
                raise ValueError("invalid existing appearance receipt: " + sid)
            continue
        shape = stage_shape(entry, args.arm, args.remote_host, ssh, rsync_ssh)
        replay = case_root / "replay"
        if not (replay / "generation.json").is_file():
            if replay.exists():
                raise FileExistsError("partial paired PBR replay requires inspection: " + sid)
            command = ("import json,sys;from pathlib import Path;"
                       "from experiments.deformnext_retargeting."
                       "replay_r_pbr_on_generated_shape import run;"
                       "run(json.loads(Path(sys.argv[1]).read_text()))")
            plan_key = "vanilla_plan" if args.arm == "vanilla" else "topk_plan"
            run([str(args.runtime_python), "-u", "-c", command,
                 entry[plan_key]], log=log_root / "replay" / f"{sid}.log", env=env)
        ensure_export(repo, args.runtime_python, benchmark_repo, shape, replay,
                      args.arm, sid, log_root / "export", env)
        case_root.mkdir(parents=True, exist_ok=True)
        for name in ("benchmark.glb", "benchmark.json"):
            source, target = replay / name, case_root / name
            if not target.exists():
                os.link(source, target)
        names = ["benchmark.glb", "benchmark.json"]
        remote = f"{args.remote_output_base}/{args.arm}/{sid}"
        # Keep the paired replay evidence remotely; local arrays are reclaimed below.
        archive_case(replay,
                     ["prediction.npz", "geometry-evaluation.npz", "generation.json",
                      "benchmark.glb", "benchmark.json"],
                     args.remote_host, remote + "/replay", ssh, rsync_ssh)
        receipt = archive_case(case_root, names, args.remote_host, remote,
                               ssh, rsync_ssh)

        # These are deterministic local staging copies.  Their checksum-verified
        # authoritative archives remain on file4; keep only GLB/JSON for rendering.
        for path in (shape / "prediction.npz", shape / "geometry-evaluation.npz"):
            path.unlink(missing_ok=True)
        for path in (replay / "prediction.npz",
                     replay / "geometry-evaluation.npz"):
            path.unlink(missing_ok=True)
        print(json.dumps({"status": receipt["status"], "arm": args.arm,
                          "sample_id": sid, "index": index, "rank": args.rank}),
              flush=True)
    completion = {
        "status": "PASS_COMMON501_APPEARANCE_GENERATION_SHARD",
        "arm": args.arm, "rank": args.rank, "world_size": args.world_size,
        "cases": len(indices), "pilot_limit": args.max_cases,
        "physical_gpu": args.physical_gpu, "reverse": args.reverse,
        "skip_cases": args.skip_cases,
    }
    suffix = ((f"-reverse-skip{args.skip_cases:03d}") if args.reverse
              else (f"-skip{args.skip_cases:03d}" if args.skip_cases else ""))
    path = (args.campaign.parent / "shards" / args.arm /
            f"rank{args.rank:02d}{suffix}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
