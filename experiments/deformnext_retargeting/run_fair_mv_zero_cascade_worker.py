"""Generate, export, and locally archive one fair MV zero-shot TEST shard."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .run_minimal_test_cascade_worker import archive_case


def sha(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def retryable_export_failure(log: str) -> bool:
    return ("export/untouched metric mesh differ in world space" in log
            or "fast simplifier produced" in log
            or "fast simplifier did not reach declared render budget" in log
            or ("RuntimeError: [CuMesh] CUDA error" in log
                and "invalid configuration argument" in log))


def run(argv: list[str], log: Path, env: dict, timeout: int | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(argv, check=True, env=env, timeout=timeout,
                       stdout=stream, stderr=subprocess.STDOUT)


def export_mesh(runtime: Path, repo: Path, root: Path, sid: str, env: dict) -> None:
    glb = root / "benchmark.glb"
    report = root / "benchmark.json"
    if glb.is_file() and report.is_file():
        return
    if glb.exists() or report.exists():
        raise FileExistsError("partial benchmark export: " + str(root))
    benchmark = Path("/root/dev/.cache/deformnext-benchmark-runtime")
    if not (benchmark / "deformnext_benchmark/io.py").is_file():
        raise FileNotFoundError("frozen common benchmark runtime required")
    for budget in (250_000, 500_000, 1_000_000, 2_000_000, 5_000_000):
        log = root.parents[2] / "logs" / "export" / f"{sid}-{budget}.log"
        command = [str(runtime), "-u", "-m",
            "experiments.deformnext_retargeting.export_minimal_test_mesh",
            "--pixal3d-repo", str(repo), "--benchmark-repo", str(benchmark),
            "--prediction", str(root / "prediction.npz"),
            "--shape-generation", str(root / "generation.json"),
            "--anchor-evaluation", str(root / "geometry-evaluation.npz"),
            "--texture-arm", "vanilla", "--render-face-budget", str(budget),
            "--output", str(glb)]
        try:
            run(command, log, env, timeout=3600 if budget >= 2_000_000 else 1200)
            break
        except subprocess.CalledProcessError:
            failure = log.read_text(errors="replace")
            if budget == 5_000_000 or glb.exists() or report.exists() \
                    or not retryable_export_failure(failure):
                raise
    record = json.loads(report.read_text())
    if (record.get("status") != "PASS_OURS_FROZEN_GT_SCALE_FREE_1024_EXPORT"
            or record.get("sample_id") != sid or record.get("texture_arm") != "vanilla"
            or record.get("prediction_sha256") != sha(root / "prediction.npz")
            or record.get("glb_sha256") != sha(glb)):
        raise ValueError("fair zero-shot export receipt differs: " + sid)


def score_surface(plan: dict, root: Path, benchmark: Path) -> None:
    """Fix source-only Sim(3), then read M1 solely for evaluation."""
    output = root / "surface-score.json"
    if output.is_file():
        return
    sys.path.insert(0, str(benchmark.resolve()))
    from deformnext_benchmark.io import canonical_hash, read_mesh
    from deformnext_benchmark.metrics import sample_surface, surface_metrics
    from .score_source_sim3_baselines import apply, register_m0

    sid = plan["row"]["sample_id"]
    export = json.loads((root / "benchmark.json").read_text())
    prediction = root / "prediction.npz"
    with np.load(prediction, allow_pickle=False) as arrays:
        vertices, faces = arrays["vertices"].copy(), arrays["faces"].copy()
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2
            or faces.shape[1] != 3 or not np.isfinite(vertices).all()
            or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("invalid fair zero-shot prediction: " + sid)
    to_world = np.asarray(export["to_world"], dtype=np.float64)
    if to_world.shape != (4, 4) or not np.allclose(to_world[3], [0, 0, 0, 1]):
        raise ValueError("invalid image-only output transform: " + sid)
    world = apply(vertices, to_world)
    source_path = Path(plan["source_geometry"])
    source_vertices, source_faces = read_mesh(source_path)
    stable = int(canonical_hash(sid)[:8], 16)
    source_points = sample_surface(source_vertices, source_faces, count=4096,
                                   seed=stable + 101)
    prediction_points = sample_surface(world, faces, count=20000, seed=stable)
    registration_points = sample_surface(world, faces, count=4096, seed=stable + 103)
    correction, fit_after, fit_before = register_m0(registration_points, source_points)
    # Target M1 is opened only after the input-source-only correction is fixed.
    target_path = Path(plan["evaluation_target_geometry"])
    target_vertices, target_faces = read_mesh(target_path)
    target_points = sample_surface(target_vertices, target_faces, count=20000,
                                   seed=stable + 1)
    result = {
        "status": "PASS_FAIR_MV_ZERO_RAW_AND_SOURCE_M0_SURFACE",
        "sample_id": sid, "source_cluster_id": plan["row"]["source_cluster_id"],
        "method": "mv_pixal3d_zero_shot_image_scale_1024",
        "texture_arm": "original-mv-pbr16850",
        "prediction_sha256": sha(prediction),
        "source_geometry_sha256": sha(source_path),
        "target_geometry_sha256": sha(target_path),
        "registration_reference": "input_source_M0_only",
        "image_estimated_scale": export["image_estimated_scale"],
        "to_world": to_world.tolist(), "source_fit_before": fit_before,
        "source_fit_after": fit_after, "correction_sim3": correction.tolist(),
        "correction_scale": float(np.cbrt(np.linalg.det(correction[:3, :3]))),
        "raw_target_M1": surface_metrics(prediction_points, target_points, threshold=.02),
        "source_aligned_target_M1": surface_metrics(
            apply(prediction_points, correction), target_points, threshold=.02),
        "target_M1_used_for_registration": False,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def retain_evaluation_cache(root: Path, cache_root: Path, sid: str) -> None:
    """Keep a temporary local hard-link cache until ten-metric evaluation closes."""
    target = cache_root / sid
    if target.exists():
        receipt = json.loads((target / "receipt.json").read_text())
        if (receipt.get("status") != "PASS_LOCAL_ZERO_EVALUATION_CACHE"
                or any(sha(target / name) != digest
                       for name, digest in receipt["sha256"].items())):
            raise ValueError("existing local evaluation cache differs: " + sid)
        return
    target.mkdir(parents=True)
    large = ("prediction.npz", "benchmark.glb")
    small = ("generation.json", "benchmark.json", "surface-score.json")
    for name in large:
        os.link(root / name, target / name)
    for name in small:
        shutil.copy2(root / name, target / name)
    hashes = {name: sha(target / name) for name in (*large, *small)}
    receipt = {"status": "PASS_LOCAL_ZERO_EVALUATION_CACHE", "sample_id": sid,
               "temporary_until_ten_metric_evaluation": True, "sha256": hashes}
    (target / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--ss-complete", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--remote-host", default="rvi@10.0.12.170")
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("--ssh-port", type=int, default=55555)
    parser.add_argument("--control-socket", type=Path, required=True)
    parser.add_argument("--evaluation-cache", type=Path, required=True)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--allowed-gpus", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--reverse", action="store_true",
                        help="process this rank from the tail for a bounded helper")
    parser.add_argument("--max-cases", type=int, default=0,
                        help="bounded tail helper size; zero runs the complete shard")
    args = parser.parse_args()
    allowed = args.allowed_gpus.split(",")
    if (args.physical_gpu not in allowed or args.physical_gpu not in {"0","1","2","3","6","7"}
            or not 0 <= args.rank < args.world_size or args.world_size < 1
            or args.max_cases < 0 or (args.reverse and args.max_cases < 1)
            or not args.runtime_python.is_file()
            or not args.remote_base.startswith(
                "/file4/youngwoo/deformation/experiments/retargeting-v1/")
            or not args.control_socket.is_socket()):
        raise ValueError("explicit authorized GPU, rank, runtime, and file4 archive required")
    campaign = json.loads(args.campaign.read_text())
    ss = json.loads(args.ss_complete.read_text())
    if (campaign.get("status") != "PREPARED_FAIR_MV_PIXAL3D_ZERO_SHOT_TEST508"
            or campaign.get("samples") != 508 or len(campaign.get("plans", [])) != 508
            or ss.get("status") != "PASS_TEST508_MV_PIXAL3D_ZERO_SHOT_SS_IMAGE_ONLY_SCALE"):
        raise ValueError("complete fair MV zero-shot campaign and SS receipt required")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.physical_gpu,
               BENCHMARK_ASSIGNED_GPU=args.physical_gpu,
               MATCHING_V2_ALLOWED_GPUS=args.allowed_gpus,
               ATTN_BACKEND="sdpa", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OPENCV_IO_ENABLE_OPENEXR="1", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="1",
               PIXAL3D_FILE4_CONTROL_SOCKET=str(args.control_socket))
    repo = Path(__file__).resolve().parents[2]
    completed = 0
    indices = list(range(args.rank, 508, args.world_size))
    if args.reverse:
        indices.reverse()
    if args.max_cases:
        indices = indices[:args.max_cases]
    for index in indices:
        entry = campaign["plans"][index]
        sid = entry["sample_id"]
        plan_path = Path(entry["plan"])
        plan = json.loads(plan_path.read_text())
        root = Path(plan["output"])
        local_receipt = root / "archive-receipt.json"
        if local_receipt.is_file():
            record = json.loads(local_receipt.read_text())
            if (record.get("status") != "PASS_FILE4_ARCHIVE_CHECKSUM_VERIFIED"
                    or record.get("remote")
                    != f"{args.remote_base}/MV-Pixal3D-zero-shot/{sid}/shape"
                    or not (args.evaluation_cache / sid / "receipt.json").is_file()):
                raise ValueError("stale local archive receipt: " + sid)
            completed += 1
            continue
        report = root / "generation.json"
        if not report.is_file():
            if root.exists():
                # The generator creates its output directory before importing the
                # runtime extensions.  A failed environment preflight can therefore
                # leave an otherwise empty directory.  Only that exact, file-free
                # state is safe to resume automatically; every material partial
                # output still requires inspection.
                if root.is_dir() and not any(root.iterdir()):
                    root.rmdir()
                else:
                    raise FileExistsError(
                        "partial zero-shot cascade requires inspection: " + str(root))
            root.parent.mkdir(parents=True, exist_ok=True)
            run([str(args.runtime_python), "-u", "-m",
                 "experiments.deformnext_retargeting.cascade_from_ss_artifact",
                 "--plan", str(plan_path)],
                args.campaign.parent / "logs" / "cascade" / (sid + ".log"), env)
        result = json.loads(report.read_text())
        expected_pbr = plan["stages"]["pbr1024"]["checkpoint"]
        if (result.get("status") != "PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION"
                or result.get("sample_id") != sid
                or result.get("object_scale_policy") != "image-estimated-v1"
                or result.get("M1_derived_scale_supplied_to_generation") is not False
                or result.get("PBR_checkpoint") != expected_pbr):
            raise ValueError("fair original four-stage generation differs: " + sid)
        export_mesh(args.runtime_python, repo, root, sid, env)
        score_surface(plan, root, Path("/root/dev/.cache/deformnext-benchmark-runtime"))
        retain_evaluation_cache(root, args.evaluation_cache, sid)
        previous_socket = os.environ.get("PIXAL3D_FILE4_CONTROL_SOCKET")
        os.environ["PIXAL3D_FILE4_CONTROL_SOCKET"] = str(args.control_socket)
        try:
            archive_case(root, args.remote_host, args.remote_base,
                         "MV-Pixal3D-zero-shot", sid, "shape", args.ssh_port,
                         with_export=True)
        finally:
            if previous_socket is None:
                os.environ.pop("PIXAL3D_FILE4_CONTROL_SOCKET", None)
            else:
                os.environ["PIXAL3D_FILE4_CONTROL_SOCKET"] = previous_socket
        completed += 1
        print(json.dumps({"status": "PASS_FAIR_MV_ZERO_CASE_ARCHIVED",
                          "index": index, "rank": args.rank,
                          "sample_id": sid, "completed": completed}), flush=True)
    result = {"status": "PASS_FAIR_MV_ZERO_SHARD_ARCHIVED", "rank": args.rank,
              "world_size": args.world_size, "cases": completed,
              "reverse": args.reverse, "max_cases": args.max_cases,
              "physical_gpu": args.physical_gpu}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
