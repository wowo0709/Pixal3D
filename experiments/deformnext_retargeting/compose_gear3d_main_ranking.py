"""Merge the matched MV zero-shot row and recompute paper-table ranks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ZERO_METHOD = "mv_pixal3d_zero_shot_source_m0"
DIRECTIONS = {
    "LPIPS": "lower", "DreamSim": "lower", "CLIP": "higher",
    "FID": "lower", "KID": "lower", "PSNR": "higher",
    "SSIM": "higher", "M_IoU": "higher", "CD": "lower",
    "FScore": "higher",
}
FORMATS = {
    "LPIPS": ".4f", "DreamSim": ".4f", "CLIP": ".4f", "FID": ".2f",
    "KID": ".3f", "PSNR": ".2f", "SSIM": ".4f", "M_IoU": ".4f",
    "CD": ".5f", "FScore": ".4f",
}


def display(metric: str, value: float, rank: str | None) -> str:
    shown = value * 1000 if metric == "KID" else value
    text = format(shown, FORMATS[metric])
    if rank == "best":
        return rf"\textbf{{{text}}}"
    if rank == "second":
        return rf"\underline{{{text}}}"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-table", type=Path, required=True)
    parser.add_argument("--zero-row", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = json.loads(args.base_table.read_text())
    zero = json.loads(args.zero_row.read_text())
    metrics = list(DIRECTIONS)
    if (base.get("status") != "PASS_SCALE_UNGIVEN_COMMON_SUCCESS_MAIN_TABLE"
            or base.get("samples") != 501 or base.get("metrics") != metrics
            or len(base.get("methods", ())) != 10
            or zero.get("status")
            != "PASS_FAIR_MV_PIXAL3D_ZERO_SHOT_COMMON501_TEN_METRICS"
            or zero.get("method") != ZERO_METHOD or zero.get("samples") != 501
            or list(zero.get("metrics", {})) != metrics
            or zero.get("training_updates") != 0):
        raise ValueError("complete matched common-501 base table and zero-shot row required")
    methods = [ZERO_METHOD, *base["methods"]]
    values = {ZERO_METHOD: zero["metrics"], **base["values"]}
    ranks: dict[str, dict[str, str]] = {}
    for metric, direction in DIRECTIONS.items():
        ordered = sorted(methods, key=lambda method: values[method][metric],
                         reverse=direction == "higher")
        ranks[metric] = {ordered[0]: "best", ordered[1]: "second"}
    formatted = {
        method: {metric: display(metric, values[method][metric],
                                 ranks[metric].get(method))
                 for metric in metrics}
        for method in methods
    }
    row = " & ".join(formatted[ZERO_METHOD][metric] for metric in metrics)
    report = {
        "status": "PASS_GEAR3D_COMMON501_ELEVEN_METHOD_RANKING",
        "samples": 501, "methods": methods, "metrics": metrics,
        "values": values, "ranks": ranks, "formatted": formatted,
        "mv_zero_latex_cells": row,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"],
                      "mv_zero_latex_cells": row}), flush=True)


if __name__ == "__main__":
    main()
