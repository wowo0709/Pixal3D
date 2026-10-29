"""Prepare common-501 full-cascade/Top-1-PBR plans for six geometry arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry-campaign", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--top1-template-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    if a.output_root.exists():
        raise FileExistsError(a.output_root)
    source = json.loads(a.geometry_campaign.read_text())
    arms = source["arms"]
    ids = [r["sample_id"] for r in rows(a.references)]
    if len(arms) != 6 or len(ids) != 501 or len(set(ids)) != 501:
        raise ValueError("six arms and common-501 required")
    source_plans = {(r["arm"], r["sample_id"]): Path(r["plan"]) for r in source["plans"]}
    entries = []
    for arm in arms:
        shape_dir = a.output_root / "plans" / arm / "shape"
        pbr_dir = a.output_root / "plans" / arm / "top1-pbr"
        shape_dir.mkdir(parents=True, exist_ok=True); pbr_dir.mkdir(parents=True, exist_ok=True)
        for sid in ids:
            shape = json.loads(source_plans[arm, sid].read_text())
            shape["output"] = str(a.output_root / "cases" / arm / sid / "shape")
            shape["execute_full_cascade_pbr"] = True
            shape_path = shape_dir / f"{sid}.json"
            shape_path.write_text(json.dumps(shape, indent=2) + "\n")
            top1_path = a.top1_template_root / f"{sid}.json"
            if not top1_path.is_file():
                raise FileNotFoundError(top1_path)
            pbr = json.loads(top1_path.read_text())
            pbr["generated_shape_root"] = shape["output"]
            pbr["output"] = str(a.output_root / "cases" / arm / sid / "top1-pbr")
            out_pbr = pbr_dir / f"{sid}.json"
            out_pbr.write_text(json.dumps(pbr, indent=2) + "\n")
            entries.append({"arm": arm, "sample_id": sid, "shape_plan": str(shape_path),
                            "top1_pbr_plan": str(out_pbr)})
    result = {"status": "PREPARED_COMMON501_SIX_ARM_FULL_RENDER_ABLATION",
              "samples": 501, "arms": arms, "plans": entries,
              "PBR_arm": "top1-only", "training_updates": 0}
    (a.output_root / "campaign.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "plans"}))


if __name__ == "__main__":
    main()
