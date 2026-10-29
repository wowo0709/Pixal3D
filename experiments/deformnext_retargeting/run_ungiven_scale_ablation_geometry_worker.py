"""Run and compact one shard of a matched TEST508 geometry ablation arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .prepare_ungiven_scale_ablation_geometry import ARMS
from .score_source_sim3_baselines import apply, register_m0


def sha(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def run(command: list[str], log: Path, env: dict, timeout: int | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(command, check=True, env=env, timeout=timeout,
                       stdout=stream, stderr=subprocess.STDOUT)


def score(plan: dict, root: Path) -> dict:
    from .surface_metrics import compare_surfaces, sample_surface

    def canonical_hash(value) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()
        return hashlib.sha256(payload).hexdigest()

    def surface_metrics(prediction_points, reference_points, *, threshold=.02) -> dict:
        metrics = compare_surfaces(prediction_points, reference_points,
                                   threshold=threshold)
        return {"CD": metrics["chamfer_L1"],
                "CD_squared_L2": metrics["chamfer_squared_L2"],
                "FScore": metrics["surface_FScore"]}

    sid = plan["row"]["sample_id"]
    prediction = root / "prediction.npz"
    evaluation = root / "geometry-evaluation.npz"
    generation = root / "generation.json"
    with np.load(evaluation, allow_pickle=False) as arrays:
        predicted_points = arrays["predicted_points"].copy()
        registration_points = arrays["registration_points"].copy()
    if (predicted_points.shape != (20000, 3)
            or registration_points.shape != (4096, 3)
            or not np.isfinite(predicted_points).all()
            or not np.isfinite(registration_points).all()):
        raise ValueError("invalid generated surface evidence: " + sid)
    with np.load(plan["source_geometry"], allow_pickle=False) as arrays:
        source_vertices, source_faces = arrays["V"].copy(), arrays["F"].copy()
    target_vertices = np.load(plan["target_vertices"])
    if target_vertices.shape != source_vertices.shape:
        raise ValueError("paired source/target topology differs: " + sid)
    stable = int(canonical_hash(sid)[:8], 16)
    source_points = sample_surface(source_vertices, source_faces, count=4096, seed=stable + 101)
    correction, fit_after, fit_before = register_m0(registration_points, source_points)
    target_points = sample_surface(target_vertices, source_faces, count=20000, seed=stable + 1)
    aligned_points = apply(predicted_points, correction)
    evidence = root / "surface-evidence.npz"
    np.savez_compressed(evidence, predicted_points=predicted_points,
                        aligned_points=aligned_points, source_points=source_points,
                        target_points=target_points, correction_sim3=correction)
    result = {
        "status": "PASS_TEST508_ABLATION_RAW_AND_SOURCE_M0_SURFACE",
        "sample_id": sid,
        "source_cluster_id": plan["row"]["source_cluster_id"],
        "arm": plan["ablation_arm"],
        "registration_reference": "input_source_M0_only",
        "target_M1_used_for_registration": False,
        "prediction_SHA256": sha(prediction),
        "geometry_evaluation_SHA256": sha(evaluation),
        "generation_SHA256": sha(generation),
        "surface_evidence_SHA256": sha(evidence),
        "source_fit_before": fit_before,
        "source_fit_after": fit_after,
        "correction_scale": float(np.cbrt(np.linalg.det(correction[:3, :3]))),
        "raw_target_M1": surface_metrics(predicted_points, target_points, threshold=.02),
        "source_aligned_target_M1": surface_metrics(aligned_points, target_points, threshold=.02),
    }
    (root / "surface-score.json").write_text(json.dumps(result, indent=2,
                                                         allow_nan=False) + "\n")
    return result


def archive_and_reclaim(root: Path, arm: str, sid: str, args) -> None:
    receipt = root / "archive-receipt.json"
    remote = f"{args.remote_base}/{arm}/{sid}/geometry"
    if receipt.is_file():
        record = json.loads(receipt.read_text())
        if record.get("status") != "PASS_FILE4_COMPACT_GEOMETRY_EVIDENCE":
            raise ValueError("stale compact evidence receipt: " + sid)
        return
    retained = [root / name for name in
                ("generation.json", "surface-score.json", "surface-evidence.npz")]
    if any(not path.is_file() for path in retained):
        raise FileNotFoundError("incomplete compact geometry evidence: " + sid)
    ssh = ["ssh", "-S", str(args.control_socket), "-o", "BatchMode=yes",
           "-p", str(args.ssh_port)]
    subprocess.run([*ssh, args.remote_host, "mkdir", "-p", remote], check=True)
    transport = " ".join(ssh)
    destination = f"{args.remote_host}:{remote}/"
    subprocess.run(["rsync", "-a", "-e", transport, *map(str, retained), destination],
                   check=True)
    check = subprocess.run(["rsync", "-a", "--checksum", "--dry-run",
                            "--itemize-changes", "-e", transport,
                            *map(str, retained), destination], check=True,
                           capture_output=True, text=True)
    if check.stdout.strip() or check.stderr.strip():
        raise ValueError("file4 compact geometry evidence differs: " + sid)
    record = {
        "status": "PASS_FILE4_COMPACT_GEOMETRY_EVIDENCE",
        "remote": remote,
        "sha256": {path.name: sha(path) for path in retained},
        "discarded_recoverable_intermediates": {
            name: sha(root / name) for name in ("prediction.npz", "geometry-evaluation.npz")
        },
    }
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run(["rsync", "-a", "-e", transport, str(receipt), destination], check=True)
    # These two large files are deterministic disposable intermediates.  The
    # sampled surfaces, metrics, transforms and their input hashes are retained.
    (root / "prediction.npz").unlink()
    (root / "geometry-evaluation.npz").unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--allowed-gpus", default="0,1,2,3,6,7")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--remote-host", default="rvi@10.0.12.170")
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("--ssh-port", type=int, default=55555)
    parser.add_argument("--control-socket", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=0,
                        help="mechanical pilot limit; zero runs the complete shard")
    parser.add_argument("--reverse", action="store_true",
                        help="process the assigned shard from the campaign tail")
    args = parser.parse_args()
    if (args.physical_gpu not in args.allowed_gpus.split(",")
            or args.physical_gpu not in {"0", "1", "2", "3", "6", "7"}
            or not 0 <= args.rank < args.world_size or args.world_size not in {1, 2, 3}
            or args.max_cases < 0
            or not args.runtime_python.is_file() or not args.control_socket.is_socket()
            or not args.remote_base.startswith(
                "/file4/youngwoo/deformation/experiments/retargeting-v1/")):
        raise ValueError("authorized local GPU shard and file4 archive required")
    campaign = json.loads(args.campaign.read_text())
    if (campaign.get("status") != "PREPARED_TEST508_PREREGISTERED_ABLATION_GEOMETRY"
            or campaign.get("samples") != 508 or tuple(campaign.get("arms", ())) != ARMS):
        raise ValueError("matched TEST508 ablation geometry campaign required")
    entries = [row for row in campaign["plans"] if row["arm"] == args.arm]
    if len(entries) != 508 or len({row["sample_id"] for row in entries}) != 508:
        raise ValueError("complete one-arm TEST508 shard required")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.physical_gpu,
               MATCHING_V2_ALLOWED_GPUS=args.allowed_gpus,
               BENCHMARK_ASSIGNED_GPU=args.physical_gpu, ATTN_BACKEND="sdpa",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="1")
    repo = Path(__file__).resolve().parents[2]
    if args.reverse:
        entries = list(reversed(entries))
    shard = entries[args.rank::args.world_size]
    if args.max_cases:
        shard = shard[:args.max_cases]
    completed = 0
    for entry in shard:
        sid = entry["sample_id"]
        plan_path = Path(entry["plan"])
        plan = json.loads(plan_path.read_text())
        root = Path(plan["output"])
        receipt = root / "archive-receipt.json"
        if receipt.is_file():
            completed += 1
            continue
        if not (root / "generation.json").is_file():
            if root.exists():
                raise FileExistsError("partial ablation geometry case: " + str(root))
            root.parent.mkdir(parents=True, exist_ok=True)
            run([str(args.runtime_python), "-u", "-m",
                 "experiments.deformnext_retargeting.cascade_from_ss_artifact",
                 "--plan", str(plan_path)],
                args.campaign.parent / "logs" / args.arm / f"{sid}-cascade.log",
                env, timeout=1800)
        generation = json.loads((root / "generation.json").read_text())
        if (generation.get("status") not in {
                    "PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION",
                    "PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_GEOMETRY_GENERATION"}
                or generation.get("sample_id") != sid
                or generation.get("object_scale_policy") != "image-estimated-v1"
                or generation.get("M1_derived_scale_supplied_to_generation") is not False):
            raise ValueError("matched image-scale cascade differs: " + sid)
        if not (root / "surface-score.json").is_file():
            score(plan, root)
        archive_and_reclaim(root, args.arm, sid, args)
        completed += 1
        print(json.dumps({"status": "PASS_ABLATION_GEOMETRY_CASE", "arm": args.arm,
                          "rank": args.rank, "sample_id": sid,
                          "completed": completed}), flush=True)
    print(json.dumps({"status": "PASS_ABLATION_GEOMETRY_SHARD", "arm": args.arm,
                      "rank": args.rank, "world_size": args.world_size,
                      "cases": completed, "pilot_limit": args.max_cases,
                      "reverse": args.reverse}), flush=True)


if __name__ == "__main__":
    main()
