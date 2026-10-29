"""Score one shard of appearance-only source-view pairs with frozen metrics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metric-registry", type=Path, required=True)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--allowed-gpus", default="0,1,2,3,6,7")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (not args.physical_gpu.isdigit()
            or args.physical_gpu not in args.allowed_gpus.split(",")
            or not 0 <= args.rank < args.world_size or args.output.exists()):
        raise ValueError("explicit authorized GPU, valid shard and fresh output required")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.physical_gpu
    os.environ["BENCHMARK_ASSIGNED_GPU"] = args.physical_gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from deformnext_benchmark.io import canonical_hash, digest, read_rgba
    from deformnext_benchmark.learned import OfflineMetrics

    protocol = json.loads(args.protocol.read_text())
    protocol_hash = canonical_hash(protocol)
    render_hash = canonical_hash(protocol["render"])
    tasks = rows(args.tasks)
    assigned = tasks[args.rank::args.world_size]
    if not assigned:
        raise ValueError("empty appearance metric shard")
    learned = OfflineMetrics(args.metric_registry, device="cuda:0")
    results = []
    for position, task in enumerate(assigned, 1):
        prediction = Path(task["prediction_rgba"])
        reference = args.reference_root / task["reference_rgba"]
        if not prediction.is_file() or not reference.is_file():
            raise FileNotFoundError("appearance image pair missing")
        pred_receipt = json.loads(prediction.with_suffix(".receipt.json").read_text())
        ref_receipt = json.loads(reference.with_suffix(".receipt.json").read_text())
        expected = {
            "protocol_sha256": protocol_hash,
            "camera_sha256": task["camera_sha256"],
            "render_sha256": render_hash,
        }
        if (any(pred_receipt.get(key) != value for key, value in expected.items())
                or any(ref_receipt.get(key) != value for key, value in expected.items())
                or pred_receipt.get("output_sha256") != digest(prediction)
                or ref_receipt.get("output_sha256") != digest(reference)):
            raise ValueError("appearance render/reference receipt differs")
        pred_rgb, pred_alpha = read_rgba(
            prediction, resolution=protocol["render"]["resolution"],
            background=protocol["render"]["background"])
        ref_rgb, _ = read_rgba(
            reference, resolution=protocol["render"]["resolution"],
            background=protocol["render"]["background"])
        if not (pred_alpha > .5).any():
            raise ValueError("empty appearance prediction render")
        metrics = learned.image_pair(pred_rgb, ref_rgb)
        if set(metrics) != {"LPIPS", "DreamSim", "CLIP"}:
            raise ValueError("exact three source-preservation metrics required")
        results.append({**task, "metrics": metrics,
                        "prediction_sha256": digest(prediction),
                        "reference_sha256": digest(reference)})
        if position % 20 == 0:
            print(json.dumps({"phase": "SCORE", "rank": args.rank,
                              "position": position, "total": len(assigned)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                    for row in results))
    print(json.dumps({"status": "PASS_APPEARANCE_METRIC_SHARD",
                      "rank": args.rank, "world_size": args.world_size,
                      "physical_gpu": args.physical_gpu, "rows": len(results),
                      "extractor": learned.identity}), flush=True)


if __name__ == "__main__":
    main()
