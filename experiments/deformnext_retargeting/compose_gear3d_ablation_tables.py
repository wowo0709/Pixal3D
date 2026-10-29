"""Validate common-501 ablations and compose paper-ready table cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


GEOMETRY_METHODS = (
    "Full", "M_FIXED_SPATIAL", "M_FIXED_RAW", "M_FIXED_D1_COS",
    "C_DECODER", "A_FIXED_UNIFORM", "A_FIXED_NO_R",
)
GEOMETRY_LABELS = {
    "Full": r"\textbf{Ours (Ray-fused Top-$K$)}",
    "M_FIXED_SPATIAL": "Spatial retrieval",
    "M_FIXED_RAW": "Semantic retrieval",
    "M_FIXED_D1_COS": "Decoder matching (cosine)",
    "C_DECODER": "Top-1 selection",
    "A_FIXED_UNIFORM": r"Uniform Top-$K$ weighting",
    "A_FIXED_NO_R": "Without retargeting",
}
APPEARANCE_METHODS = ("vanilla", "top1-only", "raw-topk-plus-feature")
APPEARANCE_LABELS = {
    "vanilla": "Original conditioning",
    "top1-only": r"Top-1 retargeting",
    "raw-topk-plus-feature": r"Top-$K$ retargeting",
}


def ranks(values: dict[str, float], *, lower: bool) -> dict[str, str]:
    ordered = sorted(values, key=values.get, reverse=not lower)
    if len(set(values.values())) != len(values):
        raise ValueError("paper ranking requires unambiguous non-tied values")
    return {ordered[0]: "best", ordered[1]: "second"}


def mark(text: str, rank: str | None) -> str:
    if rank == "best":
        return rf"\gearAblBest{{{text}}}"
    if rank == "second":
        return rf"\gearAblSecond{{{text}}}"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--appearance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    geometry = json.loads(args.geometry.read_text())
    appearance = json.loads(args.appearance.read_text())
    if (geometry.get("status") != "PASS_COMMON501_MATCHED_GEOMETRY_ABLATION"
            or geometry.get("samples") != 501
            or geometry.get("source_clusters") != 133
            or tuple(geometry.get("methods", ())) != GEOMETRY_METHODS
            or geometry.get("primary_metrics") != ["CD", "FScore"]
            or geometry.get("registration") != "input-source-M0-only Sim(3)"
            or geometry.get("object_GT_scale_given") is not False
            or geometry.get("image_estimated_scale") is not True):
        raise ValueError("complete matched common-501 geometry result required")
    if (appearance.get("status")
            != "PASS_COMMON501_APPEARANCE_ABLATION_THREE_METRICS"
            or appearance.get("samples") != 501
            or appearance.get("selected_geometry_fixed") is not True
            or appearance.get("source_registration") != "input_source_M0_only"
            or set(appearance.get("rows", ())) != set(APPEARANCE_METHODS)
            or len(appearance.get("rows", ())) != len(APPEARANCE_METHODS)
            or appearance.get("training_updates") != 0):
        raise ValueError("complete paired common-501 appearance result required")

    geometry_values = {
        method: geometry["values"][method]["source_M0_aligned"]
        for method in GEOMETRY_METHODS
    }
    geometry_ranks = {
        "CD": ranks({m: geometry_values[m]["CD"] for m in GEOMETRY_METHODS},
                    lower=True),
        "FScore": ranks({m: geometry_values[m]["FScore"]
                          for m in GEOMETRY_METHODS}, lower=False),
    }
    geometry_rows = {}
    for method in GEOMETRY_METHODS:
        cd = mark(f"{geometry_values[method]['CD'] * 1e4:.2f}",
                  geometry_ranks["CD"].get(method))
        fscore = mark(f"{geometry_values[method]['FScore'] * 1e2:.2f}",
                      geometry_ranks["FScore"].get(method))
        geometry_rows[method] = (
            f"{GEOMETRY_LABELS[method]} & {cd} & {fscore} \\\\"
        )

    appearance_values = {
        method: appearance["rows"][method]["deformation_weighted"]
        for method in APPEARANCE_METHODS
    }
    appearance_ranks = {
        metric: ranks({m: appearance_values[m][metric]
                       for m in APPEARANCE_METHODS}, lower=metric != "CLIP")
        for metric in ("LPIPS", "DreamSim", "CLIP")
    }
    appearance_rows = {}
    for method in APPEARANCE_METHODS:
        cells = []
        for metric in ("LPIPS", "DreamSim", "CLIP"):
            value = appearance_values[method][metric]
            text = (f"{value * 1e3:.2f}" if metric != "CLIP"
                    else f"{value:.4f}")
            cells.append(mark(text, appearance_ranks[metric].get(method)))
        appearance_rows[method] = (
            f"{APPEARANCE_LABELS[method]} & " + " & ".join(cells) + r" \\"
        )

    report = {
        "status": "PASS_GEAR3D_COMMON501_ABLATION_TABLE_COMPOSITION",
        "samples": 501,
        "geometry": {
            "values": geometry_values, "ranks": geometry_ranks,
            "latex_rows": geometry_rows,
            "paired_vs_full": {
                method: geometry["values"][method]["paired_vs_full"]
                for method in GEOMETRY_METHODS if method != "Full"
            },
        },
        "appearance": {
            "values": appearance_values, "ranks": appearance_ranks,
            "latex_rows": appearance_rows,
            "paired_cluster_bootstrap": appearance["paired_cluster_bootstrap"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"],
                      "geometry_rows": geometry_rows,
                      "appearance_rows": appearance_rows}), flush=True)


if __name__ == "__main__":
    main()
