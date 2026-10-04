"""Stage complete, manifest-verified mixed-source batches directly from NFS."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import tarfile

FAMILIES = ["common", "ss/64", "shape/256", "shape/512", "shape/1024",
            "pbr/256", "pbr/512", "pbr/1024"]
REQUIRED = ["common", "ss/64", "shape/512", "shape/1024", "pbr/1024"]
COUNTS = {"ObjaverseXL_sketchfab": 80, "ABO": 10, "3D-FUTURE": 5, "HSSD": 5}
LATENTS = {
    "ss64": "ss_latents/ss_enc_conv3d_16l8_fp16_64_view",
    "shape512": "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
    "shape1024": "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
    "pbr1024": "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix",
}


def select(source, destination, seed):
    rng = random.Random(seed)
    rows, seen = [], set()
    for name, count in COUNTS.items():
        candidates = sorted((source / "common" / name).glob("*/*.tar.manifest.json"))
        rng.shuffle(candidates)
        accepted = 0
        for path in candidates:
            relative = path.relative_to(source / "common")
            manifests = [source / family / relative for family in FAMILIES]
            if not all(p.exists() and Path(str(p).removesuffix(".manifest.json")).exists()
                       for p in manifests):
                continue
            data = {f: json.loads(p.read_text()) for f, p in zip(FAMILIES, manifests)}
            common = set(data["common"]["included_asset_sha256s"])
            included = set.intersection(*(set(data[f]["included_asset_sha256s"]) for f in REQUIRED))
            if not included or common & seen:
                continue
            seen.update(common)
            rows.append({"source": name, "shard": relative.parts[1],
                         "batch": path.name.removesuffix(".tar.manifest.json"),
                         "split": "train" if accepted < count else "validation",
                         "manifest_assets": len(common),
                         "manifest_common_assets": len(included),
                         "payload_bytes": sum(m["size"] for d in data.values() for m in d["members"]),
                         "manifest_sha256": {f: hashlib.sha256(p.read_bytes()).hexdigest()
                                             for f, p in zip(FAMILIES, manifests)}})
            accepted += 1
            if accepted == count + 1:
                break
        if accepted != count + 1:
            raise ValueError(f"Insufficient nonempty, disjoint batches for {name}: {accepted}")
    return {"source_root": str(source), "destination_root": str(destination),
            "seed": seed, "families": FAMILIES, "batches": rows}


def extract_batch(selection, row):
    source = Path(selection["source_root"])
    destination = Path(selection["destination_root"]) / row["source"] / row["shard"] / row["batch"]
    if destination.exists():
        report = json.loads((destination / "extraction_report.json").read_text())
        if report["selection_row"] != row:
            raise ValueError(f"Different extraction already exists: {destination}")
        return report
    staging = destination.with_name(destination.name + ".staging")
    staging.mkdir(parents=True, exist_ok=False)
    evidence = staging / "pack_manifests"
    evidence.mkdir()
    reports = []
    for family in selection["families"]:
        pack = source / family / row["source"] / row["shard"] / (row["batch"] + ".tar")
        manifest = Path(str(pack) + ".manifest.json")
        content = manifest.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if "manifest_sha256" in row and digest != row["manifest_sha256"][family]:
            raise ValueError(f"Manifest changed after selection: {manifest}")
        data = json.loads(content)
        expected = {m["path"]: m for m in data["members"]}
        checked, payload = set(), 0
        with tarfile.open(pack, "r|", bufsize=1024 * 1024) as archive:
            for member in archive:
                entry = expected[member.name]
                relative = Path(member.name)
                if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe member: {member.name}")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                member_digest = hashlib.sha256()
                with archive.extractfile(member) as stream, target.open("xb") as output:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        output.write(chunk)
                        member_digest.update(chunk)
                if target.stat().st_size != entry["size"] or member_digest.hexdigest() != entry["sha256"]:
                    raise ValueError(f"Member integrity mismatch: {member.name}")
                checked.add(member.name)
                payload += member.size
        if checked != set(expected):
            raise ValueError(f"Incomplete archive: {pack}")
        (evidence / (family.replace("/", "-") + ".manifest.json")).write_bytes(content)
        reports.append({"family": family, "files_verified": len(checked),
                        "payload_bytes": payload, "manifest_sha256": digest})
    report = {"selection_row": row, "tar_copied": False, "packs": reports}
    (staging / "extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    staging.rename(destination)
    print(json.dumps({"extracted": str(destination), "bytes": sum(r["payload_bytes"] for r in reports)}), flush=True)
    return report


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metadata_batch(selection, row):
    import numpy as np

    root = Path(selection["destination_root"]) / row["source"] / row["shard"] / row["batch"]
    report_path = root / "metadata_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if "rejected_assets" in report:
            return report
    manifests = {f: json.loads((root / "pack_manifests" / (f.replace("/", "-") + ".manifest.json")).read_text())
                 for f in REQUIRED}
    assets = sorted(set.intersection(*(set(m["included_asset_sha256s"]) for m in manifests.values())))
    records, eligible, rejected = {stage: [] for stage in LATENTS}, [], {}
    for sha in assets:
        token_counts = {}
        reasons = []
        for stage, subdir in LATENTS.items():
            key = "ss_latent" if stage == "ss64" else "pbr_latent" if stage == "pbr1024" else "shape_latent"
            item = {"sha256": sha}
            counts = []
            for anchor in (0, 1):
                with np.load(root / subdir / sha / f"view{anchor:02d}.npz") as array:
                    values = array["z"] if stage == "ss64" else array["feats"]
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite latent: {root}/{subdir}/{sha}")
                    if stage != "ss64":
                        counts.append(len(array["coords"]))
                    if stage == "pbr1024":
                        with np.load(root / LATENTS["shape1024"] / sha / f"view{anchor:02d}.npz") as shape:
                            if not np.array_equal(array["coords"], shape["coords"]):
                                reasons.append(f"shape_pbr_coordinates_view{anchor:02d}")
                scale = json.loads((root / subdir / sha / f"view{anchor:02d}_scale.json").read_text())["total_scale"]
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError(f"Invalid scale: {sha}, {stage}, view{anchor:02d}")
                item[f"{key}_view{anchor:02d}_encoded"] = True
            if counts:
                item[f"{key}_tokens"] = token_counts[stage] = max(counts)
            records[stage].append(item)
        for stage, limit in [("shape512", 8192), ("shape1024", 32768), ("pbr1024", 32768)]:
            if token_counts[stage] > limit:
                reasons.append(f"tokens_{stage}")
        if reasons:
            rejected[sha] = reasons
        else:
            eligible.append(sha)
    # All four loaders use the same eligible set; images/latents of other assets remain intact.
    retained = set(eligible)
    write_csv(root / "renders_cond/metadata.csv", [{"sha256": sha, "cond_rendered": True} for sha in eligible],
              ["sha256", "cond_rendered"])
    for stage, rows in records.items():
        key = "ss_latent" if stage == "ss64" else "pbr_latent" if stage == "pbr1024" else "shape_latent"
        fields = ["sha256", f"{key}_view00_encoded", f"{key}_view01_encoded"]
        if stage != "ss64":
            fields.append(f"{key}_tokens")
        write_csv(root / LATENTS[stage] / "metadata.csv", [r for r in rows if r["sha256"] in retained], fields)
    report = {"source": row["source"], "shard": row["shard"], "batch": row["batch"],
              "split": row["split"], "manifest_common_assets": len(assets),
              "eligible_assets": eligible, "rejected_assets": rejected,
              "token_exclusions": sum(any(r.startswith("tokens_") for r in reasons) for reasons in rejected.values()),
              "metadata_policy": "Verified completion and max token count across both anchors; original aesthetic/material metadata unavailable."}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"metadata": str(root), "eligible": len(eligible)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("operation", choices=["select", "extract", "metadata"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.operation == "select":
        selection = select(args.source, args.destination, args.seed)
        args.selection.parent.mkdir(parents=True, exist_ok=True)
        with args.selection.open("x") as output:
            json.dump(selection, output, indent=2)
        print(json.dumps({"batches": len(selection["batches"]), "payload_bytes": sum(r["payload_bytes"] for r in selection["batches"])}))
        return
    selection = json.loads(args.selection.read_text())
    if args.operation == "extract":
        destination = Path(selection["destination_root"])
        destination.mkdir(parents=True, exist_ok=True)
        pending = [r for r in selection["batches"] if not (destination / r["source"] / r["shard"] / r["batch"]).exists()]
        if sum(r.get("payload_bytes", 0) for r in pending) > shutil.disk_usage(destination).free:
            raise ValueError("Insufficient SSD space for selected full bundles")
        worker = extract_batch
    else:
        worker = metadata_batch
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(lambda row: worker(selection, row), selection["batches"]))
    args.selection.with_name(args.operation + "_summary.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
