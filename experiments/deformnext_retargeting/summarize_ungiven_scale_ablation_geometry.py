"""Summarize the matched common-501 geometry ablation with cluster intervals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .prepare_ungiven_scale_ablation_geometry import ARMS


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values: np.ndarray, clusters: list[str], *, draws: int, seed: int,
              quantiles: tuple[float, float]) -> list[float]:
    names = sorted(set(clusters))
    index = {name: i for i, name in enumerate(names)}
    sums = np.zeros(len(names), dtype=np.float64)
    counts = np.zeros(len(names), dtype=np.int64)
    for value, cluster in zip(values, clusters):
        i = index[cluster]
        sums[i] += value
        counts[i] += 1
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        stop = min(draws, start + 1000)
        choices = rng.integers(0, len(names), size=(stop - start, len(names)))
        result[start:stop] = sums[choices].sum(1) / counts[choices].sum(1)
    return [float(value) for value in np.quantile(result, quantiles)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--common-manifest", type=Path, required=True)
    parser.add_argument("--full-results", type=Path, required=True,
                        help="Common-501 evaluator result for ours_selected_source_m0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    if args.output.exists() or args.draws < 1000:
        raise ValueError("fresh output and at least 1000 cluster draws required")
    campaign = json.loads(args.campaign.read_text())
    common = json.loads(args.common_manifest.read_text())
    full = json.loads(args.full_results.read_text())
    if (campaign.get("status") != "PREPARED_TEST508_PREREGISTERED_ABLATION_GEOMETRY"
            or tuple(campaign.get("arms", ())) != ARMS
            or common.get("samples") != 501
            or full.get("method") != "ours_selected_source_m0"
            or len(full.get("per_deformation_seed", ())) != 501):
        raise ValueError("complete campaign, common-501 binding and Full result required")
    ids = common["sample_ids"]
    if len(ids) != 501 or len(set(ids)) != 501:
        raise ValueError("unique common-501 sample ids required")
    plan_index = {(row["arm"], row["sample_id"]): Path(row["plan"])
                  for row in campaign["plans"]}
    full_rows = {row["sample_id"]: row for row in full["per_deformation_seed"]}
    if set(full_rows) != set(ids):
        raise ValueError("Full result population differs from common-501")

    methods: dict[str, dict[str, dict]] = {"Full": {}}
    for sid in ids:
        row = full_rows[sid]
        methods["Full"][sid] = {
            "source_cluster_id": row["source_cluster_id"],
            "source_aligned_target_M1": {
                key: row["metrics"][key] for key in ("CD", "CD_squared_L2", "FScore")
            },
        }
    for arm in ARMS:
        records = {}
        for sid in ids:
            plan_path = plan_index.get((arm, sid))
            if plan_path is None:
                raise ValueError(f"missing common plan: {arm} {sid}")
            root = Path(json.loads(plan_path.read_text())["output"])
            score_path, receipt_path = root / "surface-score.json", root / "archive-receipt.json"
            if not score_path.is_file() or not receipt_path.is_file():
                raise FileNotFoundError(f"incomplete common geometry evidence: {arm} {sid}")
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("status") != "PASS_FILE4_COMPACT_GEOMETRY_EVIDENCE"
                    or receipt["sha256"].get("surface-score.json") != sha(score_path)):
                raise ValueError(f"geometry evidence receipt differs: {arm} {sid}")
            row = json.loads(score_path.read_text())
            if row.get("sample_id") != sid or row.get("arm") != arm:
                raise ValueError(f"geometry score identity differs: {arm} {sid}")
            records[sid] = row
        methods[arm] = records

    metrics = ("CD", "FScore")
    summary = {}
    full_values = {
        metric: np.asarray([methods["Full"][sid]["source_aligned_target_M1"][metric]
                            for sid in ids], dtype=np.float64)
        for metric in metrics
    }
    clusters = [methods["Full"][sid]["source_cluster_id"] for sid in ids]
    if len(set(clusters)) != 133:
        raise ValueError("expected 133 source clusters in common-501")
    family_alpha = .05 / (len(ARMS) * len(metrics))
    for method, rows in methods.items():
        aligned = {}
        raw = {}
        contrasts = {}
        for metric in metrics:
            values = np.asarray([rows[sid]["source_aligned_target_M1"][metric]
                                 for sid in ids], dtype=np.float64)
            aligned[metric] = float(values.mean())
            if method != "Full":
                raw_values = np.asarray([rows[sid]["raw_target_M1"][metric]
                                         for sid in ids], dtype=np.float64)
                raw[metric] = float(raw_values.mean())
                delta = values - full_values[metric]
                contrasts[metric] = {
                    "arm_minus_full": float(delta.mean()),
                    "cluster_CI95": bootstrap(delta, clusters, draws=args.draws,
                                               seed=args.seed, quantiles=(.025, .975)),
                    "family12_bonferroni_CI95": bootstrap(
                        delta, clusters, draws=args.draws, seed=args.seed,
                        quantiles=(family_alpha / 2, 1 - family_alpha / 2)),
                }
        summary[method] = {
            "source_M0_aligned": aligned,
            "raw": raw or None,
            "paired_vs_full": contrasts or None,
        }
    result = {
        "status": "PASS_COMMON501_MATCHED_GEOMETRY_ABLATION",
        "samples": 501,
        "source_clusters": 133,
        "methods": ["Full", *ARMS],
        "primary_metrics": list(metrics),
        "registration": "input-source-M0-only Sim(3)",
        "object_GT_scale_given": False,
        "image_estimated_scale": True,
        "values": summary,
        "bootstrap": {
            "unit": "source_cluster_id", "draws": args.draws, "seed": args.seed,
            "paired_family": "six arms times two geometry endpoints",
        },
        "campaign_SHA256": sha(args.campaign),
        "common_manifest_SHA256": sha(args.common_manifest),
        "full_results_SHA256": sha(args.full_results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "values": summary}), flush=True)


if __name__ == "__main__":
    main()
