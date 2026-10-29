"""Finalize the frozen common-501 fair-test evidence entirely from local files.

This is the local counterpart of the older node16 watcher chain.  It waits for
the sharded V1+source-M0 evaluator, then creates both ten-metric tables, the
raw and registered V1/Selected paired intervals, the official-RGB1 contrasts,
and the guarded claim audit.  No model inference or training is launched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


EVALUATION_STATUS = (
    "PASS_FOUR_NEW_OURS_METHODS_TEN_METRICS_REUSING_NINE_LOCAL_SHARDED"
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def run(command: list[str], *, cwd: Path) -> None:
    print(json.dumps({"phase": "RUN", "command": command}), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def require_fresh(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("immutable output already exists: " + ", ".join(existing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--static-root", type=Path, required=True)
    parser.add_argument("--raw-external", type=Path, required=True)
    parser.add_argument("--native-success", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--paper-results", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        raise ValueError("bounded polling interval must be at least ten seconds")

    eval_complete = args.evaluation_root / "complete.json"
    while not eval_complete.is_file():
        print(json.dumps({"phase": "WAIT_LOCAL_FOUR_METHOD_EVALUATION"}), flush=True)
        time.sleep(args.poll_seconds)
    evaluation_receipt = read(eval_complete)
    if (evaluation_receipt.get("status") != EVALUATION_STATUS
            or evaluation_receipt.get("samples") != 501
            or len(evaluation_receipt.get("methods", [])) != 4):
        raise ValueError("local four-method evaluation receipt differs")

    main_root = args.output_base / "main"
    raw_root = args.output_base / "raw"
    paired_root = args.output_base / "paired"
    raw_paired_root = args.output_base / "raw-paired"
    official_root = args.output_base / "official-contrasts"
    claim_root = args.output_base / "claim-audit"
    completion = args.output_base / "complete.json"
    main_tex = args.paper_results / "scale-ungiven-main-table.tex"
    raw_tex = args.paper_results / "scale-ungiven-raw-table.tex"
    registration_tex = args.paper_results / "scale-ungiven-target8-registration.tex"
    official_tex = args.paper_results / "scale-ungiven-official-contrasts.tex"
    require_fresh([
        completion, main_tex, raw_tex, registration_tex, official_tex,
        main_root / "ours-summary.json", main_root / "scale-ungiven-main-table.json",
        raw_root, paired_root, raw_paired_root, official_root, claim_root,
    ])
    main_root.mkdir(parents=True, exist_ok=True)
    args.paper_results.mkdir(parents=True, exist_ok=True)

    static_names = (
        "nine-summary.json", "common-manifest.json", "external-target8.json",
        "official-target8.json", "ours-target8.json",
    )
    for name in static_names:
        source = args.static_root / name
        target = main_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if not target.exists():
            shutil.copy2(source, target)
        elif target.read_bytes() != source.read_bytes():
            raise ValueError("staged static evidence differs: " + name)
    shutil.copy2(args.evaluation_root / "evaluation-ours-four" / "summary.json",
                 main_root / "ours-summary.json")

    python = sys.executable
    module = "experiments.deformnext_retargeting."
    main_json = main_root / "scale-ungiven-main-table.json"
    run([
        python, "-m", module + "compose_scale_ungiven_main_table",
        "--nine-summary", str(main_root / "nine-summary.json"),
        "--ours-summary", str(main_root / "ours-summary.json"),
        "--common-manifest", str(main_root / "common-manifest.json"),
        "--external-target8", str(main_root / "external-target8.json"),
        "--official-target8", str(main_root / "official-target8.json"),
        "--ours-target8", str(main_root / "ours-target8.json"),
        "--native-success", str(args.native_success),
        "--output-tex", str(main_tex), "--output-json", str(main_json),
    ], cwd=args.repo_root)
    main = read(main_json)
    if (main["status"] != "PASS_SCALE_UNGIVEN_COMMON_SUCCESS_MAIN_TABLE"
            or main["samples"] != 501 or len(main["methods"]) != 10
            or len(main["metrics"]) != 10 or not registration_tex.is_file()):
        raise ValueError("registered main table closure failed")
    write_new(main_root / "complete.json", {
        "status": "PASS_COMMON_SUCCESS_TABLE_TEN_METRICS", "samples": 501,
        "methods": 10, "metrics": 10, "json": str(main_json),
        "tex": str(main_tex), "target8_registration_tex": str(registration_tex),
    })

    paired_eval = paired_root / "evaluation-ours-four"
    paired_eval.mkdir(parents=True)
    for source in sorted((args.evaluation_root / "evaluation-ours-four").glob(
            "*-results.json")):
        shutil.copy2(source, paired_eval / source.name)
    if len(list(paired_eval.glob("*-results.json"))) != 4:
        raise ValueError("four local Ours result records required")

    paired_json = paired_root / "paired-eight-endpoints.json"
    run([
        python, "-m", module + "score_scale_ungiven_ours_paired",
        "--ours-evaluation", str(paired_eval),
        "--ours-target8", str(main_root / "ours-target8.json"),
        "--common-manifest", str(main_root / "common-manifest.json"),
        "--output", str(paired_json),
    ], cwd=args.repo_root)
    paired = read(paired_json)
    if (paired["status"]
            != "PASS_GT_SCALE_FREE_OURS_V1_SELECTED_COMMON_PAIRED_EIGHT_ENDPOINTS"
            or paired["samples"] != 501 or len(paired["metrics"]) != 8):
        raise ValueError("registered paired closure failed")
    write_new(paired_root / "complete.json", {
        "status": "PASS_COMMON_SUCCESS_OURS_PAIRED_EIGHT_ENDPOINTS",
        "samples": 501, "source_clusters": paired["source_clusters"],
        "output": str(paired_json),
    })

    raw_json = raw_root / "raw-common501-table.json"
    raw_root.mkdir()
    run([
        python, "-m", module + "compose_raw_common501_table",
        "--raw-external", str(args.raw_external),
        "--nine-summary", str(main_root / "nine-summary.json"),
        "--ours-summary", str(main_root / "ours-summary.json"),
        "--common-manifest", str(main_root / "common-manifest.json"),
        "--external-target8", str(main_root / "external-target8.json"),
        "--official-target8", str(main_root / "official-target8.json"),
        "--ours-target8", str(main_root / "ours-target8.json"),
        "--native-success", str(args.native_success),
        "--output-tex", str(raw_tex), "--output-json", str(raw_json),
    ], cwd=args.repo_root)
    raw = read(raw_json)
    if (raw["status"] != "PASS_RAW_COMMON501_FULL_TABLE"
            or raw["samples"] != 501 or len(raw["methods"]) != 10):
        raise ValueError("raw main table closure failed")
    write_new(raw_root / "complete.json", {
        "status": raw["status"], "samples": 501,
        "json": str(raw_json), "tex": str(raw_tex),
    })

    raw_paired_root.mkdir()
    raw_paired_json = raw_paired_root / "raw-paired-eight-endpoints.json"
    run([
        python, "-m", module + "score_scale_ungiven_ours_paired",
        "--ours-evaluation", str(paired_eval),
        "--ours-target8", str(main_root / "ours-target8.json"),
        "--common-manifest", str(main_root / "common-manifest.json"),
        "--mode", "raw", "--output", str(raw_paired_json),
    ], cwd=args.repo_root)
    raw_paired = read(raw_paired_json)
    if (raw_paired["status"]
            != "PASS_GT_SCALE_FREE_OURS_V1_SELECTED_RAW_COMMON_PAIRED_EIGHT_ENDPOINTS"
            or raw_paired["samples"] != 501 or len(raw_paired["metrics"]) != 8):
        raise ValueError("raw paired closure failed")
    write_new(raw_paired_root / "complete.json", {
        "status": "PASS_COMMON_SUCCESS_OURS_RAW_PAIRED_EIGHT_ENDPOINTS",
        "samples": 501, "source_clusters": raw_paired["source_clusters"],
        "output": str(raw_paired_json),
    })

    official_eval = official_root / "evaluation-nine"
    official_eval.mkdir(parents=True)
    for source in sorted((args.static_root / "evaluation-nine").glob("*-results.json")):
        shutil.copy2(source, official_eval / source.name)
    if len(list(official_eval.glob("*-results.json"))) != 9:
        raise ValueError("nine local baseline result records required")
    official_json = official_root / "official-contrasts-sixteen-endpoints.json"
    run([
        python, "-m", module + "score_scale_ungiven_official_contrasts",
        "--official-evaluation", str(official_eval),
        "--ours-evaluation", str(paired_eval),
        "--official-target8", str(main_root / "official-target8.json"),
        "--ours-target8", str(main_root / "ours-target8.json"),
        "--common-manifest", str(main_root / "common-manifest.json"),
        "--output", str(official_json),
    ], cwd=args.repo_root)
    official = read(official_json)
    if (official["status"]
            != "PASS_COMMON_TEST_OURS_VERSUS_OFFICIAL_PAIRED_SIXTEEN_ENDPOINTS"
            or official["samples"] != 501 or len(official["contrasts"]) != 2):
        raise ValueError("official contrast closure failed")
    run([
        python, "-m", module + "compose_scale_ungiven_official_contrasts_tex",
        "--contrasts", str(official_json), "--output", str(official_tex),
    ], cwd=args.repo_root)
    write_new(official_root / "complete.json", {
        "status": "PASS_COMMON_TEST_OFFICIAL_CONTRASTS_ARCHIVED",
        "samples": 501, "source_clusters": official["source_clusters"],
        "output": str(official_json), "paper_tex": str(official_tex),
    })

    claim_root.mkdir()
    claim_json = claim_root / "claim-audit.json"
    run([
        python, "-m", module + "audit_fair_test_claims",
        "--main-table-json", str(main_json), "--raw-table-json", str(raw_json),
        "--paired-json", str(paired_json),
        "--raw-paired-json", str(raw_paired_json),
        "--official-contrasts-json", str(official_json),
        "--common-manifest", str(main_root / "common-manifest.json"),
        "--output", str(claim_json),
    ], cwd=args.repo_root)
    claim = read(claim_json)
    if (claim["status"] != "PASS_FAIR_TEST_CLAIM_AUDIT"
            or claim["samples"] != 501
            or len(claim["comparisons"]["selected_vs_v1"]) != 8
            or len(claim["comparisons"]["raw_selected_vs_v1"]) != 8):
        raise ValueError("guarded claim audit closure failed")
    write_new(claim_root / "complete.json", {
        "status": "PASS_FAIR_TEST_CLAIM_AUDIT_ARCHIVED", "samples": 501,
        "output": str(claim_json),
    })
    write_new(completion, {
        "status": "PASS_LOCAL_FAIR_TEST_TABLES_INTERVALS_AND_CLAIMS",
        "samples": 501, "evaluation_receipt": str(eval_complete),
        "main": str(main_json), "raw": str(raw_json),
        "paired": str(paired_json), "raw_paired": str(raw_paired_json),
        "official_contrasts": str(official_json), "claim_audit": str(claim_json),
    })
    print(completion.read_text(), flush=True)


if __name__ == "__main__":
    main()
