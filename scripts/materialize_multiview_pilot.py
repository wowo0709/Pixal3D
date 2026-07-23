from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tarfile
import tempfile
from hashlib import sha256
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_toolkit.pipeline.packing import verify_pack


PREPARED = Path("/root/data2/pixal3d/prepared/qualification/pilot")
OUTPUT = Path("/root/node17/data/pixal3d/train/development/abo-pilot64")
SOURCE = "ABO"
SHARD = "ABO-00000"
BATCH = "batch000.tar"
FAMILIES = {
    "common": PREPARED / "common" / SOURCE / SHARD / BATCH,
    "ss64": PREPARED / "ss" / "64" / SOURCE / SHARD / BATCH,
    "shape512": PREPARED / "shape" / "512" / SOURCE / SHARD / BATCH,
    "shape1024": PREPARED / "shape" / "1024" / SOURCE / SHARD / BATCH,
    "pbr1024": PREPARED / "pbr" / "1024" / SOURCE / SHARD / BATCH,
}
STAGES = {
    "ss64": ("common", "ss64"),
    "shape512": ("common", "shape512"),
    "shape1024": ("common", "shape1024"),
    "pbr1024": ("common", "shape1024", "pbr1024"),
}
LATENT_DIRS = {
    "ss64": "ss_latents/ss_enc_conv3d_16l8_fp16_64_view",
    "shape512": "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
    "shape1024": "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
    "pbr1024": "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_verified_pack(pack: Path, manifest_path: Path, destination: Path) -> None:
    verify_pack(pack, manifest_path)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(pack, "r:") as bundle:
        for member in bundle:
            member_path = (destination / member.name).resolve()
            if not member_path.is_relative_to(destination):
                raise ValueError(f"unsafe tar member: {member.name}")
            if member.isdir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsafe non-regular tar member: {member.name}")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable tar member: {member.name}")
            member_path.parent.mkdir(parents=True, exist_ok=True)
            with source, member_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def write_metadata(root: Path, assets: list[str], values: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fields = ["sha256", *values]
    with (root / "metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for asset in assets:
            writer.writerow({"sha256": asset, **values})


def manifest_for(pack: Path) -> Path:
    return pack.with_suffix(".tar.manifest.json")


def materialize_stage(stage: str, output_root: Path) -> Path:
    final = output_root / stage / "active"
    if final.exists():
        raise FileExistsError(f"refusing to overwrite existing stage root: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".materializing-", dir=final.parent))
    evidence = {"stage": stage, "packs": []}
    try:
        asset_scope = None
        for family in STAGES[stage]:
            pack = FAMILIES[family]
            manifest_path = manifest_for(pack)
            manifest = json.loads(manifest_path.read_text())
            assets = list(manifest["included_asset_sha256s"])
            if manifest.get("completed_count") != 64 or len(assets) != 64:
                raise ValueError(f"{family} is not a complete 64-asset pilot pack")
            if asset_scope is None:
                asset_scope = assets
            elif assets != asset_scope:
                raise ValueError(f"asset scope differs for family {family}")
            extract_verified_pack(pack, manifest_path, temporary)
            evidence["packs"].append({
                "family": family,
                "pack": str(pack),
                "pack_sha256": file_sha256(pack),
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
            })

        write_metadata(temporary / "renders_cond", asset_scope, {"cond_rendered": True})
        if stage == "ss64":
            write_metadata(temporary / LATENT_DIRS[stage], asset_scope, {
                "ss_latent_view_scale00_encoded": True,
                "ss_latent_view_scale01_encoded": True,
            })
        if stage in ("shape512", "shape1024"):
            write_metadata(temporary / LATENT_DIRS[stage], asset_scope, {
                "shape_latent_view00_encoded": True,
                "shape_latent_view01_encoded": True,
            })
        if stage == "pbr1024":
            write_metadata(temporary / LATENT_DIRS["shape1024"], asset_scope, {
                "shape_latent_view00_encoded": True,
                "shape_latent_view01_encoded": True,
            })
            write_metadata(temporary / LATENT_DIRS[stage], asset_scope, {
                "pbr_latent_view00_encoded": True,
                "pbr_latent_view01_encoded": True,
            })
        evidence["asset_count"] = len(asset_scope)
        (temporary / "materialization.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=tuple(STAGES), action="append")
    args = parser.parse_args()
    for stage in args.stage or list(STAGES):
        print(materialize_stage(stage, args.output_root))


if __name__ == "__main__":
    main()
