"""Measure target alignment plus source FID/KID for retained appearance arms."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-render-root", type=Path, required=True)
    p.add_argument("--source-render-root", type=Path, required=True)
    p.add_argument("--template-plan-root", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--benchmark-root", type=Path, required=True)
    p.add_argument("--metric-registry", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--physical-gpu", default="0")
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    if a.output.exists() or a.physical_gpu not in "0,1,2,3,6,7".split(","):
        raise ValueError("fresh output and authorized GPU required")
    os.environ["CUDA_VISIBLE_DEVICES"] = a.physical_gpu
    os.environ["BENCHMARK_ASSIGNED_GPU"] = a.physical_gpu
    sys.path.insert(0, str(a.benchmark_root))
    from deformnext_benchmark.io import read_rgba
    from deformnext_benchmark.metrics import image_metrics, frechet_distance, kernel_distance
    protocol = json.loads(a.protocol.read_text())
    refs = {r["sample_id"]: r for r in rows(a.references)}
    if len(refs) != 501:
        raise ValueError("common-501 references required")
    target_gt = {}
    for path in a.template_plan_root.glob("chunk*/jobs.jsonl"):
        for job in rows(path):
            if job.get("camera_id") == 8 and job.get("method") == "gt_target8":
                target_gt[job["sample_id"]] = path.parent / "stage" / job["output_rgba"]
    jobs = {arm: {} for arm in ("vanilla", "raw-topk-plus-feature")}
    source_paths = {arm: [] for arm in jobs}
    reference_paths = []
    for chunk in a.target_render_root.glob("chunk*"):
        if not (chunk / "jobs.jsonl").is_file(): continue
        for job in rows(chunk / "jobs.jsonl"):
            arm = "vanilla" if job["method"] == "appearance_vanilla" else "raw-topk-plus-feature"
            jobs[arm][job["sample_id"]] = chunk / "stage" / job["output_rgba"]
    for sid, ref in refs.items():
        view = {v["camera_id"]: v for v in ref["views"]}
        for camera in (9, 10):
            reference_paths.append(a.reference_root / view[camera]["rgba"])
        for arm in jobs:
            matches = []
            for chunk in a.source_render_root.glob("chunk*"):
                q = chunk / "stage" / "renders" / arm / sid
                matches.extend([q / "view09.png", q / "view10.png"] if q.is_dir() else [])
            if len(matches) != 2:
                raise ValueError(f"source renders differ: {arm}/{sid}")
            source_paths[arm].extend(matches)
    if set(target_gt) != set(refs) or any(set(v) != set(refs) for v in jobs.values()):
        raise ValueError("target render population differs")
    target = {}
    for arm, images in jobs.items():
        values = []
        for sid, pred_path in images.items():
            pred, pa = read_rgba(pred_path, resolution=512, background=[1, 1, 1])
            ref, ra = read_rgba(target_gt[sid], resolution=512, background=[1, 1, 1])
            values.append(image_metrics(pred, ref, pa, ra))
        target[arm] = {key: float(np.mean([v[key] for v in values])) for key in values[0]}
    import torch
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    registry = json.loads(a.metric_registry.read_text())
    weight = a.metric_registry.parent / registry["paths"]["inception"]
    model = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"],
        feature_extractor_weights_path=str(weight)).eval().cuda().requires_grad_(False)
    def features(paths: list[Path]) -> np.ndarray:
        output = []
        with torch.inference_mode():
            for start in range(0, len(paths), a.batch_size):
                batch = []
                for path in paths[start:start + a.batch_size]:
                    rgb, _ = read_rgba(path, resolution=512, background=[1, 1, 1])
                    batch.append(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8).transpose(2, 0, 1))
                output.append(model(torch.from_numpy(np.stack(batch)).cuda())[0].float().cpu().numpy())
        return np.concatenate(output)
    reference_features = features(reference_paths)
    settings = protocol["distribution"]
    distribution = {}
    for arm, paths in source_paths.items():
        prediction = features(paths)
        distribution[arm] = {"FID": frechet_distance(prediction, reference_features),
            **kernel_distance(prediction, reference_features,
                subset_size=settings["kid_subset_size"], subsets=settings["kid_subsets"],
                seed=settings["seed"]), "images": len(paths)}
    result = {"status": "PASS_COMMON501_APPEARANCE_TARGET_AND_DISTRIBUTION",
              "samples": 501, "target_camera": 8, "source_cameras": [9, 10],
              "target": target, "distribution": distribution}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
