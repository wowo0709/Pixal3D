"""Run vanilla and Top-K common-501 appearance controls on six local GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ASSIGNMENTS = {
    "vanilla": ("0", "1", "2"),
    "raw-topk-plus-feature": ("3", "6", "7"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--remote-host", default="rvi@10.0.12.170")
    parser.add_argument("--remote-output-base", required=True)
    parser.add_argument("--ssh-port", type=int, default=55555)
    parser.add_argument("--control-socket", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign.parent
    complete_path = root / "generation-complete.json"
    if complete_path.exists():
        raise FileExistsError(complete_path)
    campaign = json.loads(args.campaign.read_text())
    if (campaign.get("status")
            != "PREPARED_COMMON501_PAIRED_NOISE_APPEARANCE_ABLATION_ON_SELECTED_GEOMETRY"
            or campaign.get("samples") != 501
            or campaign.get("paired_texture_noise") is not True):
        raise ValueError("complete common-501 appearance campaign required")
    if not args.runtime_python.is_file() or not args.control_socket.is_socket():
        raise FileNotFoundError("runtime and active file4 control socket required")

    logs = root / "orchestrator-logs"
    logs.mkdir(parents=True, exist_ok=True)
    children = []
    for arm, gpus in ASSIGNMENTS.items():
        for rank, gpu in enumerate(gpus):
            log_path = logs / f"{arm}-rank{rank:02d}-gpu{gpu}.log"
            stream = log_path.open("w")
            command = [str(args.runtime_python), "-u", "-m",
                       "experiments.deformnext_retargeting.run_ungiven_scale_appearance_worker",
                       "--campaign", str(args.campaign), "--arm", arm,
                       "--runtime-python", str(args.runtime_python),
                       "--remote-host", args.remote_host,
                       "--remote-output-base", args.remote_output_base,
                       "--ssh-port", str(args.ssh_port),
                       "--control-socket", str(args.control_socket),
                       "--physical-gpu", gpu, "--allowed-gpus", "0,1,2,3,6,7",
                       "--rank", str(rank), "--world-size", str(len(gpus))]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            children.append((arm, rank, gpu, process, stream, log_path))
            print(json.dumps({"phase": "LAUNCH", "arm": arm, "rank": rank,
                              "gpu": gpu, "pid": process.pid}), flush=True)
    failures = []
    for arm, rank, gpu, process, stream, log_path in children:
        code = process.wait()
        stream.close()
        print(json.dumps({"phase": "JOIN", "arm": arm, "rank": rank,
                          "gpu": gpu, "returncode": code}), flush=True)
        if code:
            failures.append({"arm": arm, "rank": rank, "gpu": gpu,
                             "returncode": code, "log": str(log_path)})
    if failures:
        raise RuntimeError("appearance workers failed: " + json.dumps(failures))

    counts = {}
    for arm in ASSIGNMENTS:
        roots = [Path(row["vanilla_root"] if arm == "vanilla" else row["topk_root"])
                 for row in campaign["plans"]]
        receipts = [root / "archive-receipt.json" for root in roots]
        if any(not path.is_file() for path in receipts):
            raise FileNotFoundError("appearance arm lacks 501 archive receipts: " + arm)
        for path in receipts:
            record = json.loads(path.read_text())
            if record.get("status") != "PASS_FILE4_APPEARANCE_ARCHIVE_CHECKSUM_VERIFIED":
                raise ValueError("appearance archive receipt differs: " + str(path))
        counts[arm] = len(receipts)
    result = {
        "status": "PASS_COMMON501_APPEARANCE_GENERATION_ARCHIVED",
        "samples": 501, "arms": list(ASSIGNMENTS), "counts": counts,
        "gpu_assignments": ASSIGNMENTS, "training_updates": 0,
        "large_generated_arrays_reclaimed_after_file4_archive": True,
    }
    complete_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
