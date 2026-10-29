"""Finish the frozen common-501 V1+source-M0 evaluation on local GPUs.

The learned forwards are split over contiguous sample shards.  This program
then restores the original reference/view order before computing the single
official FID/KID reduction, so the sharded execution is numerically equivalent
to the serial evaluator rather than an average of per-shard distributions.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


METHOD = "ours_v1_source_m0"
EXPECTED_EXTRACTOR = (
    "official-local-metrics-v1:"
    "00d6bc3ebdbcec88f5e7d8160f4fe18a415cd54f5a6ef7887ba08dc3378aa801:"
    "6870a7a0bc7f3bf87c8cfc0095db7c24dd8c762f775b2c99e92f29e122f333cd"
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, values: list[dict]) -> None:
    with path.open("x") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metric-registry", type=Path, required=True)
    parser.add_argument("--metric-python", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--extra-pythonpath", action="append", type=Path, default=[])
    parser.add_argument("--existing-results", type=Path)
    parser.add_argument("--method", default=METHOD)
    parser.add_argument("--standalone-one-method", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3", "6", "7"])
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--balance-by-mesh-bytes", action="store_true")
    parser.add_argument("--merge-existing-shards", action="store_true")
    parser.add_argument("--expected-samples", type=int, default=501)
    args = parser.parse_args()
    method = args.method
    evaluation_name = ("evaluation-one-method" if args.standalone_one_method
                       else "evaluation-ours-four")
    if (not method or (not args.standalone_one_method
                       and (method != METHOD or args.existing_results is None))):
        raise ValueError("explicit standalone method or original four-method inputs required")

    sys.path.insert(0, str(args.benchmark_root.parent))
    for path in args.extra_pythonpath:
        sys.path.append(str(path))

    if args.output_root.exists() and not args.merge_existing_shards:
        raise ValueError("fresh output root required")
    if (args.merge_existing_shards
            and ((args.output_root / "complete.json").exists()
                 or (args.output_root / evaluation_name).exists())):
        raise ValueError("existing shard merge requires no merged output")
    if args.cpu_threads < 1:
        raise ValueError("positive CPU thread count required")
    if not args.merge_existing_shards:
        args.output_root.mkdir(parents=True)
    refs = rows(args.binding / "references.jsonl")
    all_preds = rows(args.binding / "predictions.jsonl")
    preds = [value for value in all_preds if value["method"] == method]
    if len(refs) != args.expected_samples or len(preds) != args.expected_samples:
        raise ValueError("frozen common-501 method population required")
    if [x["sample_id"] for x in refs] != [x["sample_id"] for x in preds]:
        raise ValueError("prediction order must match frozen reference order")
    expected_binding_methods = ({method} if args.standalone_one_method else {
        "ours_v1_raw", method, "ours_selected_raw", "ours_selected_source_m0"
    })
    if set(x["method"] for x in all_preds) != expected_binding_methods:
        raise ValueError("declared evaluator binding methods differ")

    protocol = json.loads(args.protocol.read_text())
    from deformnext_benchmark.io import canonical_hash, digest, implementation_fingerprint

    if any(x["protocol_sha256"] != canonical_hash(protocol) for x in refs + preds):
        raise ValueError("protocol hash differs from binding")

    # Arbitrary worker assignment is safe because the merge below restores the
    # frozen reference/view order before the single FID/KID reduction.  The LPT
    # option avoids multi-hour tails caused by 24--412 MB compressed meshes.
    if args.merge_existing_shards:
        assignments = []
    elif args.balance_by_mesh_bytes:
        bins: list[list[int]] = [[] for _ in args.gpus]
        totals = [0] * len(args.gpus)
        weights = []
        for index, prediction in enumerate(preds):
            mesh = Path("/") / prediction["mesh"]
            weights.append(mesh.stat().st_size)
        for sample_index in sorted(range(len(refs)), key=lambda x: (-weights[x], x)):
            worker = min(range(len(bins)), key=lambda x: (totals[x], len(bins[x]), x))
            bins[worker].append(sample_index)
            totals[worker] += weights[sample_index]
        assignments = [sorted(indices) for indices in bins]
    else:
        bounds = np.linspace(0, len(refs), len(args.gpus) + 1, dtype=int)
        assignments = [list(range(int(bounds[index]), int(bounds[index + 1])))
                       for index in range(len(args.gpus))]
    children: list[tuple[int, str, Path, subprocess.Popen | None, object | None]] = []
    pythonpath = os.pathsep.join(
        [str(args.benchmark_root), *(str(path) for path in args.extra_pythonpath)]
    )
    for index, gpu in enumerate(args.gpus):
        if args.merge_existing_shards:
            shard = args.output_root / f"shard{index:02d}"
            if not shard.is_dir():
                raise FileNotFoundError(shard)
            children.append((index, gpu, shard, None, None))
            continue
        indices = assignments[index]
        if not indices:
            raise ValueError("more evaluator workers than samples")
        shard = args.output_root / f"shard{index:02d}"
        shard.mkdir()
        write_jsonl(shard / "references.jsonl", [refs[x] for x in indices])
        write_jsonl(shard / "predictions.jsonl", [preds[x] for x in indices])
        log = (shard / "evaluate.log").open("w")
        env = dict(os.environ)
        pythonpath = os.pathsep.join(
            [str(args.benchmark_root.parent), *(str(x) for x in args.extra_pythonpath)]
        )
        env.update(
            CUDA_VISIBLE_DEVICES=gpu,
            BENCHMARK_ASSIGNED_GPU=gpu,
            PYTHONPATH=pythonpath,
            OMP_NUM_THREADS=str(args.cpu_threads),
            OPENBLAS_NUM_THREADS=str(args.cpu_threads),
            MKL_NUM_THREADS=str(args.cpu_threads),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        command = [
            str(args.metric_python), "-m", "deformnext_benchmark", "evaluate",
            "--references", str(shard / "references.jsonl"),
            "--predictions", str(shard / "predictions.jsonl"),
            "--protocol", str(args.protocol),
            "--reference-root", str(args.reference_root),
            "--prediction-root", "/",
            "--output", str(shard / "evaluation"),
            "--metric-registry", str(args.metric_registry),
            "--device", "cuda:0",
        ]
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append((index, gpu, shard, child, log))
        print(json.dumps({"phase": "LAUNCH", "shard": index, "gpu": gpu,
                          "samples": len(indices),
                          "mesh_bytes": sum(weights[x] for x in indices)
                          if args.balance_by_mesh_bytes else None,
                          "pid": child.pid}), flush=True)

    failures = []
    for index, gpu, shard, child, log in children:
        if child is None:
            continue
        returncode = child.wait()
        log.close()
        print(json.dumps({"phase": "JOIN", "shard": index, "gpu": gpu,
                          "returncode": returncode}), flush=True)
        if returncode:
            failures.append((index, returncode, shard / "evaluate.log"))
    if failures:
        raise RuntimeError(f"local evaluator shard failures: {failures}")

    # Import the exact evaluator helpers only after child GPU work has closed.
    from deformnext_benchmark.aggregate import summarize
    from deformnext_benchmark.metrics import frechet_distance, kernel_distance

    per_sample: dict[str, dict] = {}
    per_view: dict[tuple[str, int], dict] = {}
    features: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    learned_versions = None
    software_versions = None
    for index, _, shard, _, _ in children:
        evaluation = shard / "evaluation"
        result_files = list(evaluation.glob("*-results.json"))
        feature_files = list(evaluation.glob("*-features.npz"))
        if len(result_files) != 1 or len(feature_files) != 1:
            raise ValueError(f"shard {index} lacks one result/feature artifact")
        result = json.loads(result_files[0].read_text())
        summary = json.loads((evaluation / "summary.json").read_text())
        if result["method"] != method or result["failures"]:
            raise ValueError(f"shard {index} method/failure mismatch")
        distribution = summary["methods"][method]["distribution"]
        if ("extractor" in distribution
                and distribution["extractor"] != EXPECTED_EXTRACTOR):
            raise ValueError("local extractor identity differs from frozen evaluator")
        if ("extractor" not in distribution
                and distribution.get("status")
                    != "NOT_COMPUTED_INSUFFICIENT_POPULATION"):
            raise ValueError("unexpected shard-level distribution state")
        if learned_versions is None:
            learned_versions = summary["learned_software_versions"]
            software_versions = summary["software_versions"]
        elif (learned_versions != summary["learned_software_versions"]
              or software_versions != summary["software_versions"]):
            raise ValueError("local shard software identities differ")
        elif learned_versions != summary["learned_software_versions"]:
            raise ValueError("learned software differs between shards")
        for value in result["per_deformation_seed"]:
            if value["sample_id"] in per_sample:
                raise ValueError("duplicate sample across shards")
            per_sample[value["sample_id"]] = value
        for value in result["per_view"]:
            key = (value["sample_id"], value["camera_id"])
            if key in per_view:
                raise ValueError("duplicate view across shards")
            per_view[key] = value
        with np.load(feature_files[0], allow_pickle=False) as loaded:
            if str(loaded["extractor_id"].item()) != EXPECTED_EXTRACTOR:
                raise ValueError("feature cache extractor identity differs")
            for row, prediction, reference in zip(
                loaded["row_ids"], loaded["prediction"], loaded["reference"]
            ):
                sid, seed, camera = json.loads(str(row))
                key = (sid, seed, camera)
                if key in features:
                    raise ValueError("duplicate feature row across shards")
                features[key] = (prediction, reference)

    ordered_samples = [per_sample[x["sample_id"]] for x in refs]
    ordered_views = [per_view[(x["sample_id"], camera)] for x in refs for camera in (9, 10)]
    feature_keys = [(x["sample_id"], 0, camera) for x in refs for camera in (9, 10)]
    if (len(ordered_samples) != args.expected_samples
            or len(ordered_views) != 2 * args.expected_samples
            or set(feature_keys) != set(features)):
        raise ValueError("merged sample/view/feature closure failed")
    pfeatures = np.asarray([features[key][0] for key in feature_keys])
    rfeatures = np.asarray([features[key][1] for key in feature_keys])

    report = summarize(
        ordered_samples,
        draws=protocol["bootstrap"]["draws"],
        seed=protocol["bootstrap"]["seed"],
    )
    report.update(
        expected_predictions=args.expected_samples,
        successful_predictions=args.expected_samples,
        failure_count=0,
        failure_stages={},
        conditional_on_success_only=True,
        complete_population=True,
        numeric_only=False,
        full_metric_computation=True,
    )
    distribution = protocol["distribution"]
    report["distribution"] = {
        "FID": frechet_distance(pfeatures, rfeatures),
        **kernel_distance(
            pfeatures,
            rfeatures,
            subset_size=distribution["kid_subset_size"],
            subsets=distribution["kid_subsets"],
            seed=distribution["seed"],
        ),
        "image_count": len(pfeatures),
        "independent_UIDs": report["source_uids"],
        "extractor": EXPECTED_EXTRACTOR,
    }

    evaluation = args.output_root / evaluation_name
    evaluation.mkdir()
    result_name = canonical_hash(method)[:16] + "-results.json"
    feature_name = canonical_hash(method)[:16] + "-features.npz"
    write_json(evaluation / result_name, {
        "method": method,
        "summary": report,
        "per_deformation_seed": ordered_samples,
        "per_view": ordered_views,
        "failures": [],
    })
    np.savez_compressed(
        evaluation / feature_name,
        prediction=pfeatures,
        reference=rfeatures,
        row_ids=np.asarray([json.dumps(list(key)) for key in feature_keys]),
        protocol_sha256=canonical_hash(protocol),
        extractor_id=EXPECTED_EXTRACTOR,
        reference_manifest_sha256=digest(args.binding / "references.jsonl"),
        prediction_manifest_sha256=digest(args.binding / "predictions.jsonl"),
    )

    if not args.standalone_one_method:
        for source in sorted(args.existing_results.glob("*-results.json")):
            existing = json.loads(source.read_text())
            if existing["method"] == method:
                continue
            shutil.copy2(source, evaluation / source.name)
    result_files = list(evaluation.glob("*-results.json"))
    results = [json.loads(path.read_text()) for path in result_files]
    methods = {value["method"]: value["summary"] for value in results}
    expected_methods = ({method} if args.standalone_one_method else {
        "ours_v1_raw", method, "ours_selected_raw", "ours_selected_source_m0"
    })
    if set(methods) != expected_methods:
        raise ValueError(f"final four-method closure failed: {set(methods)}")
    final_summary = {
        "schema": "deformnext-evaluation-results-v1",
        "protocol_sha256": canonical_hash(protocol),
        "implementation_sha256": implementation_fingerprint(),
        "software_versions": software_versions,
        "learned_software_versions": learned_versions,
        "reference_manifest_sha256": digest(args.binding / "references.jsonl"),
        "prediction_manifest_sha256": digest(args.binding / "predictions.jsonl"),
        "split": refs[0]["split"],
        "methods": methods,
        "appearance_reference": protocol["appearance_reference"],
        "per_output_alignment": False,
    }
    write_json(evaluation / "summary.json", final_summary)
    metric_names = ["LPIPS", "DreamSim", "CLIP", "FID", "KID", "PSNR", "SSIM",
                    "M_IoU", "CD", "FScore"]
    with (evaluation / "table.csv").open("x", newline="") as handle:
        fields = ["method", "weighting", "expected_predictions", "successful_predictions",
                  "failure_count", "complete_population", *metric_names]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in sorted(methods):
            method_report = methods[method]
            for weighting in ["deformation_weighted", "UID_weighted"]:
                row = {"method": method, "weighting": weighting,
                       **{key: method_report[key] for key in ["expected_predictions",
                           "successful_predictions", "failure_count", "complete_population"]}}
                row.update({key: value for key, value in method_report[weighting].items()
                            if key in metric_names})
                if weighting == "deformation_weighted":
                    row.update({key: method_report["distribution"][key]
                                for key in ("FID", "KID")})
                writer.writerow(row)

    receipt = {
        "status": ("PASS_ONE_METHOD_TEN_METRICS_LOCAL_SHARDED"
                   if args.standalone_one_method else
                   "PASS_FOUR_NEW_OURS_METHODS_TEN_METRICS_REUSING_NINE_LOCAL_SHARDED"),
        "samples": args.expected_samples,
        "methods": sorted(expected_methods),
        "source_alignment_reference": "input_source_M0_only",
        "target_M1_used_for_registration": False,
        "gpus": args.gpus,
        "sharding": ("existing_longest-processing-time_mesh_bytes"
                     if args.merge_existing_shards else
                     "longest-processing-time_mesh_bytes"
                     if args.balance_by_mesh_bytes else "contiguous_equal_count"),
        "evaluation": str(evaluation),
        "extractor": EXPECTED_EXTRACTOR,
        "training_updates": 0,
    }
    write_json(args.output_root / "complete.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
