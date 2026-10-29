"""Reconcile all 508 archived cases and close the MV zero-shot cascade."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--evaluation-cache", type=Path, required=True)
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or not args.remote_base.startswith(
            "/file4/youngwoo/deformation/experiments/retargeting-v1/")):
        raise ValueError("fresh completion output and authorized file4 base required")
    campaign = json.loads(args.campaign.read_text())
    plans = campaign.get("plans", [])
    if (campaign.get("status") != "PREPARED_FAIR_MV_PIXAL3D_ZERO_SHOT_TEST508"
            or campaign.get("samples") != 508 or len(plans) != 508
            or len({row.get("sample_id") for row in plans}) != 508):
        raise ValueError("complete unique TEST508 zero-shot campaign required")

    verified = []
    for entry in plans:
        sid = entry["sample_id"]
        plan = json.loads(Path(entry["plan"]).read_text())
        root = Path(plan["output"])
        receipt_path = root / "archive-receipt.json"
        cache = args.evaluation_cache / sid
        cache_receipt_path = cache / "receipt.json"
        if not receipt_path.is_file() or not cache_receipt_path.is_file():
            raise FileNotFoundError("missing archive/cache receipt: " + sid)
        receipt = json.loads(receipt_path.read_text())
        cache_receipt = json.loads(cache_receipt_path.read_text())
        remote = f"{args.remote_base}/MV-Pixal3D-zero-shot/{sid}/shape"
        if (receipt.get("status") != "PASS_FILE4_ARCHIVE_CHECKSUM_VERIFIED"
                or receipt.get("remote") != remote
                or set(receipt.get("sha256", {}))
                != {"prediction.npz", "geometry-evaluation.npz", "generation.json",
                    "benchmark.glb", "benchmark.json"}
                or cache_receipt.get("status") != "PASS_LOCAL_ZERO_EVALUATION_CACHE"
                or cache_receipt.get("sample_id") != sid
                or set(cache_receipt.get("sha256", {}))
                != {"prediction.npz", "benchmark.glb", "generation.json",
                    "benchmark.json", "surface-score.json"}):
            raise ValueError("archive/cache contract differs: " + sid)
        for name, digest in cache_receipt["sha256"].items():
            path = cache / name
            if not path.is_file() or sha(path) != digest:
                raise ValueError("evaluation cache checksum differs: " + sid + "/" + name)
        for name in ("prediction.npz", "benchmark.glb", "generation.json",
                     "benchmark.json"):
            if cache_receipt["sha256"][name] != receipt["sha256"][name]:
                raise ValueError("archive/cache cross-check differs: " + sid + "/" + name)
        verified.append(sid)

    completion = {
        "status": "PASS_FAIR_MV_PIXAL3D_ZERO_SHOT_TEST508_ARCHIVED",
        "samples": len(verified),
        "unique_samples": len(set(verified)),
        "physical_gpus": ["0", "1", "2", "3", "6", "7"],
        "remote": args.remote_base,
        "training_updates": 0,
        "completion_basis": "508 local archive receipts cross-checked against local evaluation-cache SHA256",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
