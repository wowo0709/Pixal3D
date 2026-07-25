from __future__ import annotations

import argparse
import ctypes
import csv
from datetime import datetime, timezone
from dataclasses import dataclass
from hashlib import sha256
import json
import os
import errno
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tarfile
import tempfile
from typing import Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_toolkit.pipeline.packing import (  # noqa: E402
    PACK_FAMILIES,
    PackMember,
    _load_manifest,
    file_sha,
    verify_pack,
)
from data_toolkit.pipeline.validation import ValidationError  # noqa: E402


DEFAULT_INDEX = Path("/root/data2/pixal3d/prepared/index/ABO/ABO-00000.json")
DEFAULT_PREPARED = Path("/root/data2/pixal3d/prepared")
DEFAULT_OUTPUT = Path("/root/node17/data/pixal3d/train/production/abo")
SOURCE = "ABO"
SHARD_ID = "ABO-00000"
EXPECTED_BATCHES = tuple(f"batch{index:03d}" for index in range(18))
STAGE_FAMILIES = {
    "ss64": ("common", "SS-64"),
    "shape512": ("common", "shape-512"),
    "shape1024": ("common", "shape-1024"),
    "pbr1024": ("common", "shape-1024", "PBR-1024"),
}
EXPECTED_STAGE_COUNTS = {
    "ss64": 3660,
    "shape512": 3631,
    "shape1024": 3660,
    "pbr1024": 3660,
}
EXPECTED_WAIVER_COUNTS = {
    "frozen_assets": 4485,
    "quarantined_assets": 825,
    "shape512_exclusions": 29,
}
_FAMILY_ROOTS = {
    "SS-64": "ss_latents/ss_enc_conv3d_16l8_fp16_64_view",
    "shape-512": "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
    "shape-1024": "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
    "PBR-1024": "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix",
}


@dataclass(frozen=True)
class FamilyPack:
    batch_id: str
    family: str
    pack: Path
    manifest: Path
    frozen_assets: tuple[str, ...]
    included_assets: tuple[str, ...]
    members: tuple[PackMember, ...]
    config_hash: str
    tool_commit: str
    pack_sha256: str
    manifest_sha256: str


def _read_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"invalid {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {description}: {path}")
    return value


def _safe_relative(value: object, prepared_root: Path, description: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"invalid {description} path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ValueError(f"unsafe {description} path: {value}")
    root = prepared_root.resolve()
    candidate = root / Path(*relative.parts)
    if not candidate.resolve().is_relative_to(root):
        raise ValueError(f"escaping {description} path: {value}")
    return candidate


def _require_regular(path: Path, description: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ValueError(f"missing {description}: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"non-regular {description}: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"empty {description}: {path}")


def _canonical_assets(values: tuple[str, ...], description: str) -> tuple[str, ...]:
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError(f"non-canonical {description}")
    return values


def load_production_catalog(
    index_path: Path,
    prepared_root: Path,
    source: str,
    shard_id: str,
    *,
    expected_batches: Sequence[str] = EXPECTED_BATCHES,
) -> dict[str, tuple[FamilyPack, ...]]:
    """Load only verified, production-gated packs from a complete shard index."""
    index = _read_json(Path(index_path), "production index")
    if (index.get("gate"), index.get("source"), index.get("shard_id")) != (
        "production", source, shard_id
    ):
        raise ValueError("invalid production index identity")
    batches = index.get("batches")
    expected = tuple(expected_batches)
    if not isinstance(batches, dict) or set(batches) != set(expected):
        raise ValueError("invalid production index batch set")
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate expected batch")
    selected = {family for families in STAGE_FAMILIES.values() for family in families}
    catalog: dict[str, list[FamilyPack]] = {family: [] for family in selected}
    for batch_id in expected:
        records = batches[batch_id]
        if not isinstance(records, dict) or set(records) != set(PACK_FAMILIES):
            raise ValueError(f"incomplete production family set: {batch_id}")
        for family in PACK_FAMILIES:
            entry = records[family]
            if not isinstance(entry, dict):
                raise ValueError(f"invalid production family record: {batch_id}: {family}")
            pack = _safe_relative(entry.get("pack"), Path(prepared_root), "pack")
            manifest_path = _safe_relative(entry.get("manifest"), Path(prepared_root), "manifest")
            _require_regular(pack, "pack")
            _require_regular(manifest_path, "manifest")
            manifest_sha = file_sha(manifest_path)
            if entry.get("manifest_sha256") != manifest_sha:
                raise ValueError(f"manifest digest mismatch: {batch_id}: {family}")
            try:
                manifest = _load_manifest(manifest_path)
            except ValidationError as error:
                raise ValueError(f"invalid manifest: {batch_id}: {family}: {error}") from error
            if manifest.schema_version != 2 or (
                manifest.shard_id, manifest.batch_id, manifest.family, manifest.gate
            ) != (shard_id, batch_id, family, "production"):
                raise ValueError(f"manifest identity mismatch: {batch_id}: {family}")
            if not manifest.validated_at:
                raise ValueError(f"missing manifest validated_at: {batch_id}: {family}")
            if entry.get("pack_sha256") != manifest.pack_sha256:
                raise ValueError(f"pack digest disagreement: {batch_id}: {family}")
            try:
                verify_pack(pack, manifest_path)
            except ValidationError as error:
                raise ValueError(f"invalid pack: {batch_id}: {family}: {error}") from error
            if family in selected:
                catalog[family].append(FamilyPack(
                    batch_id, family, pack, manifest_path,
                    _canonical_assets(manifest.asset_sha256s, "frozen scope"),
                    _canonical_assets(manifest.included_asset_sha256s, "included scope"),
                    manifest.members, manifest.config_hash, manifest.tool_commit,
                    manifest.pack_sha256, manifest_sha,
                ))
    return {family: tuple(records) for family, records in catalog.items()}


def compute_stage_scopes(
    catalog: Mapping[str, Sequence[FamilyPack]],
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, tuple[str, ...]]:
    scopes = {}
    for stage, families in STAGE_FAMILIES.items():
        try:
            family_sets = []
            for family in families:
                seen = set()
                for record in catalog[family]:
                    overlap = seen.intersection(record.included_assets)
                    if overlap:
                        raise ValueError(f"duplicate asset across batches: {family}: {sorted(overlap)[0]}")
                    seen.update(record.included_assets)
                family_sets.append(seen)
        except KeyError as error:
            raise ValueError(f"missing catalog family: {error.args[0]}") from error
        scope = tuple(sorted(set.intersection(*family_sets)))
        if expected_counts is not None and stage in expected_counts and len(scope) != expected_counts[stage]:
            raise ValueError(f"unexpected {stage} scope count: {len(scope)}")
        scopes[stage] = scope
    return scopes


def _waiver_population(catalog: Mapping[str, Sequence[FamilyPack]]) -> dict[str, int]:
    """Calculate the frozen-base waiver and its required cross-family subsets."""
    required = {family for families in STAGE_FAMILIES.values() for family in families}
    try:
        frozen = {
            family: {asset for record in catalog[family] for asset in record.frozen_assets}
            for family in required
        }
        included = {
            family: {asset for record in catalog[family] for asset in record.included_assets}
            for family in required
        }
    except KeyError as error:
        raise ValueError(f"missing waiver family: {error.args[0]}") from error
    frozen_base = frozen["common"]
    if any(scope != frozen_base for scope in frozen.values()):
        raise ValueError("waiver frozen populations differ across families")
    baseline = included["common"]
    for family in ("SS-64", "shape-1024", "PBR-1024"):
        if included[family] != baseline:
            raise ValueError(f"waiver baseline inclusion differs for {family}")
    if not included["shape-512"].issubset(baseline):
        raise ValueError("waiver shape-512 scope is not a baseline subset")
    return {
        "frozen_assets": len(frozen_base),
        "quarantined_assets": len(frozen_base - baseline),
        "shape512_exclusions": len(baseline - included["shape-512"]),
    }


def _validate_waiver(
    catalog: Mapping[str, Sequence[FamilyPack]], expected_waiver: Mapping[str, int]
) -> dict[str, int]:
    actual = _waiver_population(catalog)
    if actual != dict(expected_waiver):
        raise ValueError(f"unexpected production waiver population: {actual}")
    return actual


def _expected_member_paths(family: str, asset: str) -> tuple[str, ...]:
    if family == "common":
        return tuple(f"renders_cond/{asset}/{frame:03d}.png" for frame in range(8)) + (f"renders_cond/{asset}/transforms.json",)
    root = _FAMILY_ROOTS[family]
    return tuple(f"{root}/{asset}/{name}" for name in ("view00.npz", "view00_scale.json", "view01.npz", "view01_scale.json"))


def _copy_selected(pack: FamilyPack, assets: tuple[str, ...], temporary: Path) -> None:
    expected = {path for asset in assets for path in _expected_member_paths(pack.family, asset)}
    manifest_members = {member.path: member for member in pack.members}
    if not expected.issubset(manifest_members):
        missing = sorted(expected - set(manifest_members))[0]
        raise ValueError(f"missing selected manifest member: {missing}")
    selected_assets = set(assets)
    for path in manifest_members:
        if selected_assets.intersection(PurePosixPath(path).parts) and path not in expected:
            raise ValueError(f"unexpected selected manifest member: {path}")
    with tarfile.open(pack.pack, "r:") as bundle:
        actual = set()
        for member in bundle:
            if selected_assets.intersection(PurePosixPath(member.name).parts) and member.name not in expected:
                raise ValueError(f"unexpected selected tar member: {member.name}")
            if member.name not in expected:
                continue
            if member.issym() or member.islnk() or not member.isfile():
                raise ValueError(f"unsafe selected tar member: {member.name}")
            target = (temporary / member.name).resolve()
            if not target.is_relative_to(temporary.resolve()):
                raise ValueError(f"unsafe selected member path: {member.name}")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable selected tar member: {member.name}")
            digest = sha256(); size = 0
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    size += len(block); digest.update(block); output.write(block)
            expected_member = manifest_members[member.name]
            if size != expected_member.size or digest.hexdigest() != expected_member.sha256:
                raise ValueError(f"selected member digest mismatch: {member.name}")
            actual.add(member.name)
    if actual != expected:
        raise ValueError(f"selected member set mismatch: {pack.family}")


def _write_metadata(path: Path, assets: tuple[str, ...], fields: dict[str, bool]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sha256", *fields])
        writer.writeheader()
        for asset in assets:
            writer.writerow({"sha256": asset, **fields})


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _publish_no_replace(temporary: Path, final: Path) -> None:
    """Atomically publish a directory only while its lexical destination is absent."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise RuntimeError("atomic no-replace publication is unavailable") from error
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(_AT_FDCWD, os.fsencode(temporary), _AT_FDCWD, os.fsencode(final), _RENAME_NOREPLACE) != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(f"refusing to overwrite existing stage root: {final}")
        raise OSError(error_number, os.strerror(error_number), final)


def _acquire_destination_lock(final: Path) -> Path:
    if os.path.lexists(final):
        raise FileExistsError(f"refusing to overwrite existing stage root: {final}")
    lock = final.with_name(".active.materialize.lock")
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"stage materialization already active: {final}") from error
    os.close(descriptor)
    if os.path.lexists(final):
        lock.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite existing stage root: {final}")
    return lock


def materialize_stage(
    stage: str,
    catalog: Mapping[str, Sequence[FamilyPack]],
    output_root: Path,
    *,
    index_path: Path,
    expected_counts: Mapping[str, int] = EXPECTED_STAGE_COUNTS,
    expected_waiver: Mapping[str, int] = EXPECTED_WAIVER_COUNTS,
) -> Path:
    if stage not in STAGE_FAMILIES:
        raise ValueError(f"unknown production stage: {stage}")
    final = Path(output_root) / stage / "active"
    if os.path.lexists(final):
        raise FileExistsError(f"refusing to overwrite existing stage root: {final}")
    scopes = compute_stage_scopes(catalog, expected_counts)
    waiver = _validate_waiver(catalog, expected_waiver)
    assets = scopes[stage]
    final.parent.mkdir(parents=True, exist_ok=True)
    lock = _acquire_destination_lock(final)
    temporary: Path | None = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix=".materializing-", dir=final.parent))
        evidence_packs = []
        for family in STAGE_FAMILIES[stage]:
            records = catalog[family]
            for record in records:
                selected = tuple(asset for asset in assets if asset in record.included_assets)
                _copy_selected(record, selected, temporary)
                evidence_packs.append({
                    "batch_id": record.batch_id, "family": family,
                    "pack": str(record.pack), "manifest": str(record.manifest),
                    "pack_sha256": record.pack_sha256, "manifest_sha256": record.manifest_sha256,
                    "tool_commit": record.tool_commit,
                })
        _write_metadata(temporary / "renders_cond", assets, {"cond_rendered": True})
        for family in STAGE_FAMILIES[stage]:
            if family == "common":
                continue
            field_prefix = {"SS-64": "ss_latent", "shape-512": "shape_latent", "shape-1024": "shape_latent", "PBR-1024": "pbr_latent"}[family]
            _write_metadata(temporary / _FAMILY_ROOTS[family], assets, {
                f"{field_prefix}_view00_encoded": True, f"{field_prefix}_view01_encoded": True,
            })
        evidence = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": SOURCE,
            "shard_id": SHARD_ID,
            "source_index": {"path": str(Path(index_path).resolve()), "sha256": file_sha(Path(index_path))},
            "acceptance_mode": "valid_subset_user_waiver",
            "original_90_percent_gate_passed": False,
            "counts": {
                "frozen": waiver["frozen_assets"], "global_quarantine": waiver["quarantined_assets"],
                "shape512_family_exclusions": waiver["shape512_exclusions"],
                "stages": {name: len(scope) for name, scope in scopes.items()},
            },
            "asset_count": len(assets), "index_sha256": file_sha(Path(index_path)),
            "packs": sorted(evidence_packs, key=lambda value: (value["family"], value["batch_id"])),
            "stage": stage, "stage_root": str(final.resolve()), "stage_scope": list(assets),
            "stage_scope_sha256": sha256("\n".join(assets).encode()).hexdigest(),
            "tool_commits": sorted({record.tool_commit for family in STAGE_FAMILIES[stage] for record in catalog[family]}),
            "waiver": "production-valid-subset",
            **waiver,
        }
        (temporary / "materialization.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        _publish_no_replace(temporary, final)
    except Exception:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        lock.unlink(missing_ok=True)
    return final


def materialize_all(
    index_path: Path,
    prepared_root: Path,
    output_root: Path,
    *,
    expected_batches: Sequence[str] = EXPECTED_BATCHES,
    expected_counts: Mapping[str, int] = EXPECTED_STAGE_COUNTS,
) -> dict[str, Path]:
    catalog = load_production_catalog(index_path, prepared_root, SOURCE, SHARD_ID, expected_batches=expected_batches)
    return {stage: materialize_stage(stage, catalog, output_root, index_path=index_path, expected_counts=expected_counts) for stage in STAGE_FAMILIES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=tuple(STAGE_FAMILIES), action="append")
    args = parser.parse_args()
    catalog = load_production_catalog(args.index, args.prepared_root, SOURCE, SHARD_ID)
    for stage in args.stage or tuple(STAGE_FAMILIES):
        print(materialize_stage(stage, catalog, args.output_root, index_path=args.index))


if __name__ == "__main__":
    main()
