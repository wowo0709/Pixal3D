"""Pilot and launch the fair MV-Pixal3D zero-shot TEST508 cascade on six GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


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
    parser.add_argument("--gpus", default="0,1,2,3,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if (args.output_root.exists() or gpus != ["0","1","2","3","6","7"]
            or args.workers_per_gpu not in {1, 2}
            or not args.campaign.is_file() or not args.ss_complete.is_file()
            or not args.runtime_python.is_file() or not args.control_socket.is_socket()
            or not args.remote_base.startswith(
                "/file4/youngwoo/deformation/experiments/retargeting-v1/")):
        raise ValueError("fresh output and exact authorized six-GPU assignment required")
    args.output_root.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    base = [str(args.runtime_python), "-u", "-m",
            "experiments.deformnext_retargeting.run_fair_mv_zero_cascade_worker",
            "--campaign", str(args.campaign), "--ss-complete", str(args.ss_complete),
            "--runtime-python", str(args.runtime_python),
            "--remote-host", args.remote_host, "--remote-base", args.remote_base,
            "--ssh-port", str(args.ssh_port),
            "--control-socket", str(args.control_socket),
            "--evaluation-cache", str(args.evaluation_cache),
            "--allowed-gpus", args.gpus]
    pilot = [*base, "--physical-gpu", "0", "--rank", "0", "--world-size", "508"]
    with (args.output_root / "pilot.log").open("w") as log:
        subprocess.run(pilot, cwd=repo, stdin=subprocess.DEVNULL,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({"phase": "PASS_ONE_FAIR_MV_ZERO_CASE_PILOT"}), flush=True)
    assignments = [(gpu, worker)
                   for gpu in gpus for worker in range(args.workers_per_gpu)]
    processes = []
    for rank, (gpu, worker) in enumerate(assignments):
        label = f"gpu{gpu}-worker{worker}"
        log = (args.output_root / f"{label}.log").open("w")
        command = [*base, "--physical-gpu", gpu, "--rank", str(rank),
                   "--world-size", str(len(assignments))]
        process = subprocess.Popen(command, cwd=repo, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        processes.append((label, process, log))
    print(json.dumps({"phase": "RUN_FAIR_MV_ZERO_TEST508",
                      "workers_per_gpu": args.workers_per_gpu,
                      "pids": {label: process.pid
                               for label, process, _ in processes}}), flush=True)
    codes = []
    for label, process, log in processes:
        codes.append((label, process.wait()))
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError("fair MV zero-shot workers failed: " + repr(codes))
    campaign = json.loads(args.campaign.read_text())
    archived = 0
    for entry in campaign["plans"]:
        plan = json.loads(Path(entry["plan"]).read_text())
        archived += (Path(plan["output"]) / "archive-receipt.json").is_file()
    if archived != 508:
        raise ValueError(f"only {archived}/508 fair zero-shot cases archived")
    completion = {"status": "PASS_FAIR_MV_PIXAL3D_ZERO_SHOT_TEST508_ARCHIVED",
                  "samples": 508, "physical_gpus": gpus,
                  "workers_per_gpu": args.workers_per_gpu,
                  "remote": args.remote_base, "training_updates": 0}
    (args.output_root / "complete.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
