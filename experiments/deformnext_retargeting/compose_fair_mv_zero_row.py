"""Compose the ten-metric MV-Pixal3D zero-shot common-501 table row."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD = "mv_pixal3d_zero_shot_source_m0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--target8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    evaluation = json.loads(args.evaluation_summary.read_text())
    target = json.loads(args.target8.read_text())
    if (set(evaluation.get("methods", {})) != {METHOD}
            or target.get("status") != "PASS_FAIR_MV_ZERO_SOURCE_M0_TARGET8"
            or target.get("method") != METHOD or target.get("samples") != 501):
        raise ValueError("complete matched common-501 zero-shot evidence required")
    source = evaluation["methods"][METHOD]
    if (source.get("successful_predictions") != 501
            or not source.get("full_metric_computation")
            or source.get("failure_count") != 0):
        raise ValueError("zero-shot learned/geometry evaluation is incomplete")
    weighted = source["deformation_weighted"]
    distribution = source["distribution"]
    metrics = {
        "LPIPS": weighted["LPIPS"], "DreamSim": weighted["DreamSim"],
        "CLIP": weighted["CLIP"], "FID": distribution["FID"],
        "KID": distribution["KID"], "PSNR": target["summary"]["PSNR"]["mean"],
        "SSIM": target["summary"]["SSIM"]["mean"],
        "M_IoU": target["summary"]["M_IoU"]["mean"],
        "CD": weighted["CD"], "FScore": weighted["FScore"],
    }
    report = {
        "status": "PASS_FAIR_MV_PIXAL3D_ZERO_SHOT_COMMON501_TEN_METRICS",
        "method": METHOD, "samples": 501, "metrics": metrics,
        "source_views": [9, 10], "target_view": 8,
        "source_registration_reference": "input_source_M0_only",
        "target_M1_used_for_registration": False, "training_updates": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
