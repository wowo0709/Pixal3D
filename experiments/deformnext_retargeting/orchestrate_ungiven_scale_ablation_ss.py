"""Launch six matched TEST508 SS ablation arms in parallel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ARMS = {
    "M_FIXED_SPATIAL": ("shared-Fixed-correspondence-campaign-v301", "fixed-correspondence"),
    "M_FIXED_RAW": ("shared-Fixed-correspondence-campaign-v301", "fixed-correspondence"),
    "M_FIXED_D1_COS": ("shared-Fixed-correspondence-campaign-v301", "fixed-correspondence"),
    "C_DECODER": (None, "frozen-retarget"),
    "A_FIXED_UNIFORM": ("shared-evidence-final-controls-v284", "shared-evidence-ablation"),
    "A_FIXED_NO_R": ("shared-evidence-final-controls-v284", "shared-evidence-ablation"),
}
SAFE_GPUS = (0, 1, 2, 3, 6, 7)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--panel-file", type=Path, required=True)
    parser.add_argument("--seed-map-file", type=Path, required=True)
    parser.add_argument("--scale-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,6,7")
    parser.add_argument("--generation-batch", type=int, default=8)
    args = parser.parse_args()

    gpus = tuple(int(value) for value in args.gpus.split(","))
    if (len(gpus) != len(ARMS) or len(set(gpus)) != len(gpus)
            or any(gpu not in SAFE_GPUS for gpu in gpus)):
        raise ValueError("exactly six unique local GPUs from 0,1,2,3,6,7 are required")
    if args.output_root.exists():
        raise ValueError("fresh campaign output required")
    common = (args.selection, args.freeze, args.test_manifest, args.panel_file,
              args.seed_map_file, args.scale_model)
    if any(not path.is_file() for path in common):
        raise FileNotFoundError("one or more frozen common inputs are missing")

    records = []
    for (arm, (campaign, family)), gpu in zip(ARMS.items(), gpus):
        if arm == "C_DECODER":
            freeze = json.loads(args.freeze.read_text())
            binding = freeze["models"]["RETARGET"]["ss64"]
            paths = [Path(binding[key]["path"]) for key in
                     ("native_config", "checkpoint", "conditioner_checkpoint")]
            if any(not path.is_file() for path in paths):
                raise FileNotFoundError("incomplete frozen C_DECODER checkpoint")
            records.append({
                "arm": arm,
                "runtime_family": family,
                "gpu": gpu,
                "freeze": str(args.freeze),
                "freeze_SHA256": sha(args.freeze),
                "checkpoint_SHA256": sha(paths[1]),
                "conditioner_SHA256": sha(paths[2]),
                "output": str(args.output_root / arm),
            })
            continue
        root = args.model_root / campaign / arm
        bundle = root / "final-evaluation-v1/bundle480-v1/bundle.json"
        if not bundle.is_file():
            raise FileNotFoundError(bundle)
        binding = json.loads(bundle.read_text())
        paths = [bundle, *(Path(binding[key]) for key in
                           ("native_config", "checkpoint", "conditioner_checkpoint"))]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError(f"incomplete staged checkpoint for {arm}")
        records.append({
            "arm": arm,
            "runtime_family": family,
            "gpu": gpu,
            "bundle": str(bundle),
            "bundle_SHA256": sha(bundle),
            "checkpoint_SHA256": sha(Path(binding["checkpoint"])),
            "conditioner_SHA256": sha(Path(binding["conditioner_checkpoint"])),
            "output": str(args.output_root / arm),
        })

    args.output_root.mkdir(parents=True)
    manifest = {
        "status": "RUNNING_TEST508_PREREGISTERED_SS_ABLATION",
        "UTC": datetime.now(timezone.utc).isoformat(),
        "protocol": "TEST508 image-estimated scale; source-M0-only Sim3 downstream",
        "optimizer_updates": 0,
        "selection": str(args.selection),
        "selection_SHA256": sha(args.selection),
        "test_manifest": str(args.test_manifest),
        "test_manifest_SHA256": sha(args.test_manifest),
        "panel": str(args.panel_file),
        "panel_SHA256": sha(args.panel_file),
        "seeds": str(args.seed_map_file),
        "seeds_SHA256": sha(args.seed_map_file),
        "scale_model": str(args.scale_model),
        "scale_model_SHA256": sha(args.scale_model),
        "arms": records,
    }
    campaign_path = args.output_root / "campaign.json"
    campaign_path.write_text(json.dumps(manifest, indent=2) + "\n")

    processes = []
    for record in records:
        if record["arm"] == "C_DECODER":
            command = [
                sys.executable, "-m",
                "experiments.deformnext_retargeting.evaluate_ungiven_scale_val_ss",
                "--freeze", str(args.freeze), "--arm", "RETARGET",
                "--mode", "image-estimated", "--split", "test",
                "--scale-model", str(args.scale_model),
                "--panel-file", str(args.panel_file),
                "--seed-map-file", str(args.seed_map_file),
                "--output", record["output"],
                "--generation-batch", str(args.generation_batch),
            ]
        else:
            command = [
                sys.executable, "-m",
                "experiments.deformnext_retargeting.evaluate_ungiven_scale_ablation_ss",
                "--selection", str(args.selection),
                "--bundle", record["bundle"],
                "--runtime-family", record["runtime_family"],
                "--test-manifest", str(args.test_manifest),
                "--panel-file", str(args.panel_file),
                "--seed-map-file", str(args.seed_map_file),
                "--scale-model", str(args.scale_model),
                "--output", record["output"],
                "--generation-batch", str(args.generation_batch),
            ]
        log_path = args.output_root / f"{record['arm']}.log"
        log = log_path.open("w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(record["gpu"]),
                   MATCHING_V2_ALLOWED_GPUS=",".join(map(str, SAFE_GPUS)))
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=Path(__file__).resolve().parents[2], env=env)
        processes.append((record, process, log, log_path))

    failures = []
    for record, process, log, log_path in processes:
        code = process.wait()
        log.close()
        if code:
            failures.append({"arm": record["arm"], "exit_code": code,
                             "log": str(log_path)})
    if failures:
        manifest.update(status="FAILED_TEST508_PREREGISTERED_SS_ABLATION",
                        failures=failures)
        campaign_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise SystemExit(1)

    summaries = {}
    for record in records:
        result = json.loads((Path(record["output"]) / "evaluation.json").read_text())
        summaries[record["arm"]] = result["quick_metrics"]
        complete_path = Path(record["output"]) / "complete.json"
        if record["arm"] == "C_DECODER":
            scale = json.loads((Path(record["output"]) / "scale-contract.json").read_text())
            if (result.get("status") != "PASS" or result.get("arm") != "RETARGET"
                    or scale.get("status") != "PASS_TEST_SS_SCALE_UNPROVIDED_FROZEN"
                    or scale.get("samples") != 508):
                raise ValueError("frozen C_DECODER TEST508 evaluation differs")
            complete_path.write_text(json.dumps({
                "status": "PASS_TEST508_PREREGISTERED_ABLATION_IMAGE_ONLY_SCALE",
                "arm": "C_DECODER", "frozen_alias": "RETARGET", "samples": 508,
                "voxel_iou": result["quick_metrics"]["voxel_iou"],
            }, indent=2) + "\n")
    manifest.update(status="PASS_TEST508_PREREGISTERED_SS_ABLATION",
                    completed_UTC=datetime.now(timezone.utc).isoformat(),
                    quick_metrics=summaries)
    campaign_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_root / "complete.json").write_text(json.dumps({
        "status": manifest["status"], "arms": list(ARMS),
        "optimizer_updates": 0,
    }, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "quick_metrics": summaries}), flush=True)


if __name__ == "__main__":
    main()
