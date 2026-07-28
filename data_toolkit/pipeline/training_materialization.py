"""Source-aware, create-only production training materialization."""

from __future__ import annotations

import ctypes
import csv
from datetime import datetime, timezone
from dataclasses import dataclass
from hashlib import sha256
import json
import os
import errno
from pathlib import Path, PurePosixPath
import secrets
import shutil
import stat
import tarfile
from typing import Mapping, Sequence

from data_toolkit.pipeline.packing import (  # noqa: E402
    PACK_FAMILIES,
    PackMember,
    _load_manifest,
    file_sha,
    verify_pack,
)
from data_toolkit.pipeline.training_eligibility import (  # noqa: E402
    ABO_COUNT_CONTRACT,
    EXPECTED_CANDIDATE_STAGE_COUNTS,
    EXPECTED_FINAL_STAGE_COUNTS,
    EXPECTED_FROZEN_COUNT,
    EXPECTED_GLOBAL_QUARANTINE_COUNT,
    EXPECTED_SHAPE512_FAMILY_EXCLUSION_COUNT,
    EXPECTED_TRAINING_EXCLUSION_COUNTS,
    EligibilityExclusion,
    canonical_count_contract,
    filter_stage_scope,
    observed_count_contract,
    policy_evidence,
)
from data_toolkit.pipeline.validation import ValidationError  # noqa: E402


DEFAULT_INDEX = Path("/root/data2/pixal3d/prepared/index/ABO/ABO-00000.json")
DEFAULT_PREPARED = Path("/root/data2/pixal3d/prepared")
DEFAULT_OUTPUT = Path("/root/node17/data/pixal3d/train/production/abo")
THREED_FUTURE_INDEXES = (
    Path(
        "/root/data2/pixal3d/prepared/index/3D-FUTURE/"
        "3D-FUTURE-00000.json"
    ),
    Path(
        "/root/data2/pixal3d/prepared/index/3D-FUTURE/"
        "3D-FUTURE-00001.json"
    ),
)
THREED_FUTURE_OUTPUT = Path(
    "/root/node17/data/pixal3d/train/production/3d-future"
)
SOURCE = "ABO"
SHARD_ID = "ABO-00000"
EXPECTED_BATCHES = tuple(f"batch{index:03d}" for index in range(18))
STAGE_FAMILIES = {
    "ss64": ("common", "SS-64"),
    "shape512": ("common", "shape-512"),
    "shape1024": ("common", "shape-1024"),
    "pbr1024": ("common", "shape-1024", "PBR-1024"),
}
EXPECTED_CANDIDATE_COUNTS = EXPECTED_CANDIDATE_STAGE_COUNTS
EXPECTED_STAGE_COUNTS = EXPECTED_FINAL_STAGE_COUNTS
EXPECTED_WAIVER_COUNTS = {
    "frozen_assets": EXPECTED_FROZEN_COUNT,
    "quarantined_assets": EXPECTED_GLOBAL_QUARANTINE_COUNT,
    "shape512_exclusions": EXPECTED_SHAPE512_FAMILY_EXCLUSION_COUNT,
}
_FAMILY_ROOTS = {
    "SS-64": "ss_latents/ss_enc_conv3d_16l8_fp16_64_view",
    "shape-512": "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
    "shape-1024": "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
    "PBR-1024": "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix",
}
_FAMILY_METADATA_FIELDS = {
    "SS-64": (
        "ss_latent_view_scale00_encoded",
        "ss_latent_view_scale01_encoded",
    ),
    "shape-512": (
        "shape_latent_view00_encoded",
        "shape_latent_view01_encoded",
    ),
    "shape-1024": (
        "shape_latent_view00_encoded",
        "shape_latent_view01_encoded",
    ),
    "PBR-1024": (
        "pbr_latent_view00_encoded",
        "pbr_latent_view01_encoded",
    ),
}


@dataclass(frozen=True)
class ProductionSourceSpec:
    """Immutable source inputs and acceptance/count contracts."""

    source: str
    indexes: tuple[Path, ...]
    expected_batches: Mapping[str, tuple[str, ...]]
    expected_frozen: int
    expected_candidate_stages: Mapping[str, int]
    fixed_count_contract: Mapping[str, object] | None
    acceptance_mode: str
    original_90_percent_gate_passed: bool


ABO_SOURCE_SPEC = ProductionSourceSpec(
    source=SOURCE,
    indexes=(DEFAULT_INDEX,),
    expected_batches={SHARD_ID: EXPECTED_BATCHES},
    expected_frozen=EXPECTED_FROZEN_COUNT,
    expected_candidate_stages=EXPECTED_CANDIDATE_STAGE_COUNTS,
    fixed_count_contract=ABO_COUNT_CONTRACT,
    acceptance_mode="valid_subset_user_waiver",
    original_90_percent_gate_passed=False,
)
THREED_FUTURE_SOURCE_SPEC = ProductionSourceSpec(
    source="3D-FUTURE",
    indexes=THREED_FUTURE_INDEXES,
    expected_batches={
        "3D-FUTURE-00000": tuple(
            f"batch{index:03d}" for index in range(20)
        ),
        "3D-FUTURE-00001": tuple(
            f"batch{index:03d}" for index in range(18)
        ),
    },
    expected_frozen=9472,
    expected_candidate_stages={
        "ss64": 8495,
        "shape512": 8513,
        "shape1024": 8495,
        "pbr1024": 8495,
    },
    fixed_count_contract=None,
    acceptance_mode="valid_subset_user_waiver",
    original_90_percent_gate_passed=False,
)


@dataclass(frozen=True)
class FamilyPack:
    source: str
    shard_id: str
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
    source_index_path: Path
    source_index_sha256: str


def _read_regular_bytes(path: Path, description: str) -> bytes:
    path = Path(path)
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise ValueError(f"missing {description}: {path}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"non-regular {description}: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"non-regular {description}: {path}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"non-regular {description}: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _read_json(path: Path, description: str) -> tuple[dict, bytes]:
    raw = _read_regular_bytes(path, description)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"invalid {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {description}: {path}")
    return value, raw


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


def _load_shard_catalog(
    index_path: Path,
    prepared_root: Path,
    source: str,
    shard_id: str,
    *,
    expected_batches: Sequence[str] = EXPECTED_BATCHES,
) -> dict[str, tuple[FamilyPack, ...]]:
    """Load only verified, production-gated packs from a complete shard index."""
    index_path = Path(index_path)
    index, index_raw = _read_json(index_path, "production index")
    canonical_index_path = index_path.resolve()
    index_sha256 = sha256(index_raw).hexdigest()
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
                catalog[family].append(
                    FamilyPack(
                        source=source,
                        shard_id=shard_id,
                        batch_id=batch_id,
                        family=family,
                        pack=pack,
                        manifest=manifest_path,
                        frozen_assets=_canonical_assets(
                            manifest.asset_sha256s, "frozen scope"
                        ),
                        included_assets=_canonical_assets(
                            manifest.included_asset_sha256s,
                            "included scope",
                        ),
                        members=manifest.members,
                        config_hash=manifest.config_hash,
                        tool_commit=manifest.tool_commit,
                        pack_sha256=manifest.pack_sha256,
                        manifest_sha256=manifest_sha,
                        source_index_path=canonical_index_path,
                        source_index_sha256=index_sha256,
                    )
                )
    return {family: tuple(records) for family, records in catalog.items()}


def load_production_catalog(
    index_path: Path,
    prepared_root: Path,
    source: str,
    shard_id: str,
    *,
    expected_batches: Sequence[str] = EXPECTED_BATCHES,
) -> dict[str, tuple[FamilyPack, ...]]:
    """Compatibility loader for one verified production shard."""
    return _load_shard_catalog(
        index_path,
        prepared_root,
        source,
        shard_id,
        expected_batches=expected_batches,
    )


def _shard_frozen_scope(
    shard_id: str, catalog: Mapping[str, Sequence[FamilyPack]]
) -> set[str]:
    """Validate one shard's frozen scope is disjoint by batch and family."""
    family_scopes: dict[str, set[str]] = {}
    for family, records in catalog.items():
        scope: set[str] = set()
        for record in records:
            overlap = scope.intersection(record.frozen_assets)
            if overlap:
                raise ValueError(
                    "duplicate frozen asset across batches: "
                    f"{shard_id}: {family}: {sorted(overlap)[0]}"
                )
            scope.update(record.frozen_assets)
        family_scopes[family] = scope
    baseline = family_scopes.get("common")
    if baseline is None or any(
        scope != baseline for scope in family_scopes.values()
    ):
        raise ValueError(f"frozen populations differ across families: {shard_id}")
    return baseline


def _validate_source_spec(spec: ProductionSourceSpec) -> tuple[str, ...]:
    if not isinstance(spec, ProductionSourceSpec):
        raise TypeError("spec must be a ProductionSourceSpec")
    shard_ids = tuple(path.stem for path in spec.indexes)
    if (
        not spec.source
        or not spec.indexes
        or len(set(spec.indexes)) != len(spec.indexes)
        or len(set(shard_ids)) != len(shard_ids)
        or set(shard_ids) != set(spec.expected_batches)
    ):
        raise ValueError("source spec indexes do not match expected shards")
    if (
        isinstance(spec.expected_frozen, bool)
        or not isinstance(spec.expected_frozen, int)
        or spec.expected_frozen < 0
    ):
        raise ValueError("source spec frozen count must be non-negative")
    candidate_counts = dict(spec.expected_candidate_stages)
    if set(candidate_counts) != set(STAGE_FAMILIES) or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > spec.expected_frozen
        for value in candidate_counts.values()
    ):
        raise ValueError("invalid source spec candidate stage counts")
    return shard_ids


def load_source_catalog(
    spec: ProductionSourceSpec, prepared_root: Path
) -> dict[str, tuple[FamilyPack, ...]]:
    """Load and aggregate every verified shard in a source contract."""
    shard_ids = _validate_source_spec(spec)
    selected = {
        family for families in STAGE_FAMILIES.values() for family in families
    }
    aggregate: dict[str, list[FamilyPack]] = {
        family: [] for family in selected
    }
    frozen: set[str] = set()
    for index_path, shard_id in zip(spec.indexes, shard_ids):
        shard_catalog = _load_shard_catalog(
            index_path,
            prepared_root,
            spec.source,
            shard_id,
            expected_batches=spec.expected_batches[shard_id],
        )
        shard_frozen = _shard_frozen_scope(shard_id, shard_catalog)
        overlap = frozen.intersection(shard_frozen)
        if overlap:
            raise ValueError(
                "asset overlap across shards: "
                f"{shard_id}: {sorted(overlap)[0]}"
            )
        frozen.update(shard_frozen)
        for family in selected:
            aggregate[family].extend(shard_catalog[family])
    if len(frozen) != spec.expected_frozen:
        raise ValueError(f"unexpected frozen source count: {len(frozen)}")
    catalog = {
        family: tuple(records) for family, records in aggregate.items()
    }
    compute_stage_scopes(catalog, spec.expected_candidate_stages)
    return catalog


def _validate_catalog_identity(
    spec: ProductionSourceSpec,
    catalog: Mapping[str, Sequence[FamilyPack]],
) -> None:
    expected_shards = set(spec.expected_batches)
    expected_packs = {
        (shard_id, batch_id)
        for shard_id, batch_ids in spec.expected_batches.items()
        for batch_id in batch_ids
    }
    for family in {
        value
        for families in STAGE_FAMILIES.values()
        for value in families
    }:
        try:
            records = catalog[family]
        except KeyError as error:
            raise ValueError(
                f"missing catalog identity family: {family}"
            ) from error
        actual_packs = [
            (record.shard_id, record.batch_id) for record in records
        ]
        if (
            {record.shard_id for record in records} != expected_shards
            or set(actual_packs) != expected_packs
            or len(actual_packs) != len(expected_packs)
        ):
            raise ValueError(f"catalog identity mismatch: {family}")
        for record in records:
            if (
                record.source != spec.source
                or record.family != family
                or record.batch_id
                not in spec.expected_batches[record.shard_id]
            ):
                raise ValueError(f"catalog identity mismatch: {family}")


def _pinned_source_indexes(
    spec: ProductionSourceSpec,
    catalog: Mapping[str, Sequence[FamilyPack]],
) -> list[dict[str, str]]:
    """Recover one immutable index pin per source shard from the catalog."""
    pins = []
    for index_path in spec.indexes:
        expected_path = Path(index_path).resolve()
        path_shard = Path(index_path).stem
        if path_shard in spec.expected_batches:
            shard_id = path_shard
        elif len(spec.indexes) == len(spec.expected_batches) == 1:
            shard_id = next(iter(spec.expected_batches))
        else:
            raise ValueError(
                f"catalog index pin cannot resolve shard: {expected_path}"
            )
        observed = {
            (record.source_index_path, record.source_index_sha256)
            for records in catalog.values()
            for record in records
            if record.shard_id == shard_id
        }
        if len(observed) != 1:
            raise ValueError(
                f"catalog index pin is inconsistent: {expected_path}"
            )
        pinned_path, pinned_digest = observed.pop()
        if (
            pinned_path != expected_path
            or len(pinned_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in pinned_digest
            )
        ):
            raise ValueError(f"catalog index pin is invalid: {shard_id}")
        pins.append(
            {
                "shard_id": shard_id,
                "path": str(expected_path),
                "sha256": pinned_digest,
            }
        )
    return pins


def _validate_pinned_indexes(
    source_indexes: Sequence[Mapping[str, str]],
) -> None:
    for record in source_indexes:
        path = Path(record["path"])
        current = sha256(
            _read_regular_bytes(path, "production index")
        ).hexdigest()
        if current != record["sha256"]:
            raise ValueError(f"production index bytes changed: {path}")


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


def _source_population(
    catalog: Mapping[str, Sequence[FamilyPack]], expected_frozen: int
) -> dict[str, int]:
    """Return source-wide pack exclusions without ABO's subset assumptions."""
    required = {
        family for families in STAGE_FAMILIES.values() for family in families
    }
    try:
        frozen = {
            family: {
                asset
                for record in catalog[family]
                for asset in record.frozen_assets
            }
            for family in required
        }
        included = {
            family: {
                asset
                for record in catalog[family]
                for asset in record.included_assets
            }
            for family in required
        }
    except KeyError as error:
        raise ValueError(
            f"missing source population family: {error.args[0]}"
        ) from error
    frozen_base = frozen["common"]
    if any(scope != frozen_base for scope in frozen.values()):
        raise ValueError("source frozen populations differ across families")
    if len(frozen_base) != expected_frozen:
        raise ValueError(f"unexpected frozen source count: {len(frozen_base)}")
    baseline = included["common"]
    return {
        "frozen_assets": len(frozen_base),
        "quarantined_assets": len(frozen_base - baseline),
        "shape512_exclusions": len(baseline - included["shape-512"]),
    }


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


def _scope_sha256(assets: Sequence[str]) -> str:
    return sha256("\n".join(assets).encode()).hexdigest()


def _remove_excluded_assets(
    temporary: Path, stage: str, exclusions: Sequence[EligibilityExclusion]
) -> None:
    """Remove only validated extracted asset directories in this hidden stage."""
    components = ["renders_cond"] + [
        _FAMILY_ROOTS[family] for family in STAGE_FAMILIES[stage] if family != "common"
    ]
    temporary_root = temporary.resolve(strict=True)
    for exclusion in exclusions:
        asset = exclusion.asset
        if not isinstance(asset, str) or not asset or Path(asset).name != asset:
            raise ValueError(f"unsafe excluded asset: {asset!r}")
        for component in components:
            component_root = (temporary / component).resolve(strict=True)
            if not component_root.is_relative_to(temporary_root):
                raise ValueError(f"unsafe temporary component: {component}")
            target = component_root / asset
            try:
                target_stat = target.lstat()
            except OSError as error:
                raise ValueError(f"missing excluded asset directory: {component}/{asset}") from error
            if stat.S_ISLNK(target_stat.st_mode) or not stat.S_ISDIR(target_stat.st_mode):
                raise ValueError(f"invalid excluded asset directory: {component}/{asset}")
            if not target.resolve(strict=True).is_relative_to(component_root):
                raise ValueError(f"unsafe excluded asset directory: {component}/{asset}")
            shutil.rmtree(target)


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _open_directory_nofollow(path: Path) -> int:
    """Open and pin a directory without following any path component."""
    absolute = Path(os.path.abspath(path))
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
    )
    directory_fd = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def _rename_no_replace(
    source: Path,
    destination: Path,
    *,
    source_dir_fd: int = _AT_FDCWD,
    destination_dir_fd: int = _AT_FDCWD,
) -> None:
    """Atomically rename one path while its lexical destination is absent."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise RuntimeError("atomic no-replace publication is unavailable") from error
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(
        source_dir_fd,
        os.fsencode(source),
        destination_dir_fd,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    ) != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(
                f"refusing to overwrite existing destination: {destination}"
            )
        raise OSError(
            error_number, os.strerror(error_number), destination
        )


def _publish_no_replace(
    temporary: Path,
    final: Path,
    *,
    parent_fd: int | None = None,
) -> None:
    """Atomically publish a directory only while its lexical destination is absent."""
    try:
        _rename_no_replace(
            temporary,
            final,
            source_dir_fd=(
                parent_fd if parent_fd is not None else _AT_FDCWD
            ),
            destination_dir_fd=(
                parent_fd if parent_fd is not None else _AT_FDCWD
            ),
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing stage root: {final}"
        ) from error


def _preserve_failed_attempt(
    temporary: Path,
    output_root: Path,
    stage: str,
    *,
    source_parent_fd: int | None = None,
) -> Path:
    """Move one failed staging tree to the trusted production reject root."""
    rejected = Path(output_root).parent / "rejected"
    try:
        rejected_fd = _open_directory_nofollow(rejected)
    except OSError as error:
        raise RuntimeError(
            f"rejected directory is not a safe directory: {rejected}"
        ) from error
    owns_source_parent_fd = source_parent_fd is None
    try:
        rejected_stat = os.fstat(rejected_fd)
        if not stat.S_ISDIR(rejected_stat.st_mode):
            raise RuntimeError(
                f"rejected directory is not a directory: {rejected}"
            )
        if source_parent_fd is None:
            source_parent_fd = _open_directory_nofollow(temporary.parent)
        temporary_stat = os.stat(
            temporary.name,
            dir_fd=source_parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(temporary_stat.st_mode):
            raise RuntimeError(
                f"staging attempt is not a directory: {temporary}"
            )
        if temporary_stat.st_dev != rejected_stat.st_dev:
            raise RuntimeError(
                "rejected directory is not on the staging filesystem: "
                f"{rejected}"
            )
        suffix = temporary.name.removeprefix(".materializing-")
        destination_name = (
            f"{Path(output_root).name}-{stage}-materializing-{suffix}"
        )
        _rename_no_replace(
            Path(temporary.name),
            Path(destination_name),
            source_dir_fd=source_parent_fd,
            destination_dir_fd=rejected_fd,
        )
        return rejected / destination_name
    finally:
        if owns_source_parent_fd and source_parent_fd is not None:
            os.close(source_parent_fd)
        os.close(rejected_fd)


def _entry_exists(parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _create_staging_directory(parent_fd: int) -> str:
    for _attempt in range(100):
        name = f".materializing-{secrets.token_hex(8)}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        return name
    raise FileExistsError("could not allocate a unique staging directory")


def _acquire_destination_lock(
    final: Path,
    *,
    parent_fd: int | None = None,
) -> Path:
    if (
        _entry_exists(parent_fd, final.name)
        if parent_fd is not None
        else os.path.lexists(final)
    ):
        raise FileExistsError(f"refusing to overwrite existing stage root: {final}")
    lock = final.with_name(".active.materialize.lock")
    try:
        descriptor = os.open(
            lock.name if parent_fd is not None else lock,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
            dir_fd=parent_fd,
        )
    except FileExistsError as error:
        raise FileExistsError(f"stage materialization already active: {final}") from error
    os.close(descriptor)
    if (
        _entry_exists(parent_fd, final.name)
        if parent_fd is not None
        else os.path.lexists(final)
    ):
        if parent_fd is None:
            lock.unlink(missing_ok=True)
        else:
            os.unlink(lock.name, dir_fd=parent_fd)
        raise FileExistsError(f"refusing to overwrite existing stage root: {final}")
    return lock


def _materialize_stage(
    stage: str,
    catalog: Mapping[str, Sequence[FamilyPack]],
    output_root: Path,
    *,
    index_path: Path,
    spec: ProductionSourceSpec = ABO_SOURCE_SPEC,
    expected_counts: Mapping[str, int] = EXPECTED_CANDIDATE_COUNTS,
    expected_waiver: Mapping[str, int] | None = None,
    expected_stage_counts: Mapping[str, int] | None = None,
    expected_training_exclusion_counts: Mapping[str, int] | None = None,
) -> Path:
    if stage not in STAGE_FAMILIES:
        raise ValueError(f"unknown production stage: {stage}")
    final = Path(output_root) / stage / "active"
    if os.path.lexists(final):
        raise FileExistsError(f"refusing to overwrite existing stage root: {final}")
    candidate_scopes = compute_stage_scopes(catalog, expected_counts)
    if expected_waiver is None and spec.fixed_count_contract is not None:
        expected_waiver = {
            "frozen_assets": spec.fixed_count_contract["frozen"],
            "quarantined_assets": spec.fixed_count_contract[
                "global_quarantine"
            ],
            "shape512_exclusions": spec.fixed_count_contract[
                "shape512_family_exclusions"
            ],
        }
    waiver = (
        _validate_waiver(catalog, expected_waiver)
        if expected_waiver is not None
        else _source_population(catalog, spec.expected_frozen)
    )
    candidate_assets = candidate_scopes[stage]
    using_default_stage_counts = expected_stage_counts is None
    if using_default_stage_counts and spec.fixed_count_contract is not None:
        expected_stage_counts = spec.fixed_count_contract["stages"]
    using_default_exclusion_counts = expected_training_exclusion_counts is None
    if (
        using_default_exclusion_counts
        and spec.fixed_count_contract is not None
    ):
        expected_training_exclusion_counts = spec.fixed_count_contract[
            "training_exclusions"
        ]
    source_indexes = _pinned_source_indexes(spec, catalog)
    if (
        Path(index_path).resolve()
        != Path(source_indexes[0]["path"])
    ):
        raise ValueError("materialization index path does not match catalog")
    _validate_pinned_indexes(source_indexes)
    final.parent.mkdir(parents=True, exist_ok=True)
    stage_parent_fd = _open_directory_nofollow(final.parent)
    lock: Path | None = None
    temporary: Path | None = None
    temporary_name: str | None = None
    temporary_fd: int | None = None
    try:
        lock = _acquire_destination_lock(
            final,
            parent_fd=stage_parent_fd,
        )
        temporary_name = _create_staging_directory(stage_parent_fd)
        temporary_fd = os.open(
            temporary_name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | os.O_CLOEXEC,
            dir_fd=stage_parent_fd,
        )
        temporary = Path(f"/proc/self/fd/{temporary_fd}")
        evidence_packs = []
        for family in STAGE_FAMILIES[stage]:
            records = catalog[family]
            for record in records:
                selected = tuple(asset for asset in candidate_assets if asset in record.included_assets)
                _copy_selected(record, selected, temporary)
                evidence_pack = {
                    "batch_id": record.batch_id, "family": family,
                    "pack": str(record.pack), "manifest": str(record.manifest),
                    "pack_sha256": record.pack_sha256, "manifest_sha256": record.manifest_sha256,
                    "tool_commit": record.tool_commit,
                }
                if len(spec.indexes) > 1 or spec.source != SOURCE:
                    evidence_pack = {
                        "source": record.source,
                        "shard_id": record.shard_id,
                        **evidence_pack,
                    }
                evidence_packs.append(evidence_pack)
        assets, exclusions = filter_stage_scope(stage, temporary, candidate_assets)
        _remove_excluded_assets(temporary, stage, exclusions)
        if (
            expected_stage_counts is not None
            and len(assets) != expected_stage_counts[stage]
        ):
            raise ValueError(f"unexpected {stage} final scope count: {len(assets)}")
        if (
            expected_training_exclusion_counts is not None
            and len(exclusions)
            != expected_training_exclusion_counts[stage]
        ):
            raise ValueError(f"unexpected {stage} training exclusion count: {len(exclusions)}")
        candidate_count_contract = {
            name: len(scope) for name, scope in candidate_scopes.items()
        }
        if expected_stage_counts is None:
            final_count_contract = {
                **candidate_count_contract,
                stage: len(assets),
            }
        elif using_default_stage_counts:
            final_count_contract = dict(expected_stage_counts)
        else:
            final_count_contract = {
                **candidate_count_contract,
                **expected_stage_counts,
            }
        if expected_training_exclusion_counts is None:
            exclusion_count_contract = {
                name: (len(exclusions) if name == stage else 0)
                for name in candidate_count_contract
            }
        elif using_default_exclusion_counts:
            exclusion_count_contract = dict(
                expected_training_exclusion_counts
            )
        else:
            exclusion_count_contract = {
                **{name: 0 for name in candidate_count_contract},
                **expected_training_exclusion_counts,
            }
        _write_metadata(temporary / "renders_cond", assets, {"cond_rendered": True})
        for family in STAGE_FAMILIES[stage]:
            if family == "common":
                continue
            _write_metadata(
                temporary / _FAMILY_ROOTS[family],
                assets,
                {field: True for field in _FAMILY_METADATA_FIELDS[family]},
            )
        if spec.fixed_count_contract is None:
            observed_counts = observed_count_contract(
                frozen=waiver["frozen_assets"],
                candidate_stages=candidate_count_contract,
                training_exclusions=exclusion_count_contract,
            )
            count_contract = {
                "global_quarantine": waiver["quarantined_assets"],
                "shape512_family_exclusions": waiver[
                    "shape512_exclusions"
                ],
                "frozen": observed_counts["frozen"],
                **{
                    field: {stage: observed_counts[field][stage]}
                    for field in (
                        "candidate_stages",
                        "pack_exclusions",
                        "training_exclusions",
                        "stages",
                    )
                },
            }
        else:
            count_contract = canonical_count_contract(
                frozen=waiver["frozen_assets"],
                global_quarantine=waiver["quarantined_assets"],
                shape512_family_exclusions=waiver[
                    "shape512_exclusions"
                ],
                candidate_stages=candidate_count_contract,
                training_exclusions=exclusion_count_contract,
                stages=final_count_contract,
            )
        evidence = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": spec.source,
            "acceptance_mode": spec.acceptance_mode,
            "original_90_percent_gate_passed":
                spec.original_90_percent_gate_passed,
            "counts": count_contract,
            "candidate_asset_count": len(candidate_assets),
            "candidate_stage_scope": list(candidate_assets),
            "candidate_stage_scope_sha256": _scope_sha256(candidate_assets),
            "asset_count": len(assets),
            "packs": sorted(
                evidence_packs,
                key=lambda value: (
                    value.get("shard_id", ""),
                    value["family"],
                    value["batch_id"],
                ),
            ),
            "stage": stage, "stage_root": str(final.resolve()), "stage_scope": list(assets),
            "stage_scope_sha256": _scope_sha256(assets),
            "training_exclusion_count": len(exclusions),
            "training_exclusions": [
                {"asset": exclusion.asset, "reasons": list(exclusion.reasons)}
                for exclusion in exclusions
            ],
            "training_exclusion_reason_counts": {
                reason: sum(reason in exclusion.reasons for exclusion in exclusions)
                for reason in sorted({reason for exclusion in exclusions for reason in exclusion.reasons})
            },
            "eligibility_policy": policy_evidence(),
            "tool_commits": sorted({record.tool_commit for family in STAGE_FAMILIES[stage] for record in catalog[family]}),
            "waiver": "production-valid-subset",
            **waiver,
        }
        if len(source_indexes) == 1:
            evidence.update(
                {
                    "shard_id": source_indexes[0]["shard_id"],
                    "source_index": {
                        "path": source_indexes[0]["path"],
                        "sha256": source_indexes[0]["sha256"],
                    },
                    "index_sha256": source_indexes[0]["sha256"],
                }
            )
        else:
            evidence["source_indexes"] = source_indexes
        (temporary / "materialization.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        _validate_pinned_indexes(source_indexes)
        _publish_no_replace(
            Path(temporary_name),
            Path(final.name),
            parent_fd=stage_parent_fd,
        )
    except BaseException as error:
        if (
            temporary_name is not None
            and _entry_exists(stage_parent_fd, temporary_name)
        ):
            try:
                rejected_attempt = _preserve_failed_attempt(
                    Path(temporary_name),
                    output_root,
                    stage,
                    source_parent_fd=stage_parent_fd,
                )
            except BaseException as preservation_error:
                raise RuntimeError(
                    "failed to preserve materialization attempt; "
                    "staging retained in pinned stage parent as "
                    f"{temporary_name}"
                ) from preservation_error
            error.add_note(
                f"failed materialization preserved at {rejected_attempt}"
            )
        raise
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if lock is not None:
            try:
                os.unlink(lock.name, dir_fd=stage_parent_fd)
            except FileNotFoundError:
                pass
        os.close(stage_parent_fd)
    return final


def materialize_stage(
    spec: ProductionSourceSpec,
    stage: str,
    catalog: Mapping[str, Sequence[FamilyPack]],
    output_root: Path,
) -> Path:
    """Materialize one source stage under the source's count contract."""
    _validate_source_spec(spec)
    _validate_catalog_identity(spec, catalog)
    return _materialize_stage(
        stage,
        catalog,
        output_root,
        index_path=spec.indexes[0],
        spec=spec,
        expected_counts=spec.expected_candidate_stages,
    )


def materialize_all(
    spec: ProductionSourceSpec,
    prepared_root: Path,
    output_root: Path,
) -> dict[str, Path]:
    """Materialize all stages from one fully verified source catalog."""
    catalog = load_source_catalog(spec, prepared_root)
    return {
        stage: materialize_stage(spec, stage, catalog, output_root)
        for stage in STAGE_FAMILIES
    }
