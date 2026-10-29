"""Run six matched TEST508 geometry arms with two workers per local GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .prepare_ungiven_scale_ablation_geometry import ARMS


GPUS = ("0", "1", "2", "3", "6", "7")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--remote-host", default="rvi@10.0.12.170")
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("--ssh-port", type=int, default=55555)
    parser.add_argument("--control-socket", type=Path, required=True)
    parser.add_argument("--workers-per-arm", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    root = args.campaign.parent
    completion = root / "geometry-complete.json"
    if completion.exists():
        raise FileExistsError(completion)
    campaign = json.loads(args.campaign.read_text())
    if (campaign.get("status") != "PREPARED_TEST508_PREREGISTERED_ABLATION_GEOMETRY"
            or campaign.get("samples") != 508
            or tuple(campaign.get("arms", ())) != ARMS
            or not args.runtime_python.is_file() or not args.control_socket.is_socket()
            or not args.remote_base.startswith(
                "/file4/youngwoo/deformation/experiments/retargeting-v1/")):
        raise ValueError("complete matched geometry campaign/runtime/file4 binding required")
    logs = root / "orchestrator-logs"
    logs.mkdir(parents=True, exist_ok=True)
    children = []
    for arm, gpu in zip(ARMS, GPUS):
        for rank in range(args.workers_per_arm):
            log_path = logs / f"{arm}-rank{rank:02d}-gpu{gpu}.log"
            stream = log_path.open("w")
            command = [str(args.runtime_python), "-u", "-m",
                       "experiments.deformnext_retargeting."
                       "run_ungiven_scale_ablation_geometry_worker",
                       "--campaign", str(args.campaign), "--runtime-python",
                       str(args.runtime_python), "--arm", arm,
                       "--physical-gpu", gpu, "--allowed-gpus", ",".join(GPUS),
                       "--rank", str(rank), "--world-size", str(args.workers_per_arm),
                       "--remote-host", args.remote_host,
                       "--remote-base", args.remote_base,
                       "--ssh-port", str(args.ssh_port),
                       "--control-socket", str(args.control_socket)]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            children.append((arm, rank, gpu, process, stream, log_path))
            print(json.dumps({"phase": "LAUNCH", "arm": arm, "rank": rank,
                              "gpu": gpu, "pid": process.pid}), flush=True)
    failures = []
    for arm, rank, gpu, process, stream, log_path in children:
        code = process.wait(); stream.close()
        print(json.dumps({"phase": "JOIN", "arm": arm, "rank": rank,
                          "gpu": gpu, "returncode": code}), flush=True)
        if code:
            failures.append({"arm": arm, "rank": rank, "gpu": gpu,
                             "returncode": code, "log": str(log_path)})
    if failures:
        raise RuntimeError("geometry workers failed: " + json.dumps(failures))
    counts = {}
    for arm in ARMS:
        receipts = list((root / "cases" / arm).glob("*/shape/archive-receipt.json"))
        if len(receipts) != 508:
            raise ValueError(f"{arm} has {len(receipts)} compact receipts, expected 508")
        if any(json.loads(path.read_text()).get("status")
               != "PASS_FILE4_COMPACT_GEOMETRY_EVIDENCE" for path in receipts):
            raise ValueError("invalid compact geometry receipt: " + arm)
        counts[arm] = len(receipts)
    result = {
        "status": "PASS_TEST508_SIX_ARM_GEOMETRY_ARCHIVED",
        "samples": 508, "arms": list(ARMS), "counts": counts,
        "gpu_assignments": dict(zip(ARMS, GPUS)),
        "workers_per_arm": args.workers_per_arm,
        "large_deterministic_prediction_arrays_reclaimed_after_archive": True,
        "training_updates": 0,
    }
    completion.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
