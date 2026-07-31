# Paper-Faithful Family Eligibility Design

## Goal

Preserve every geometry-usable asset for Pixal3D shape and sparse-structure
training while restricting material training to assets whose materials can be
parsed by the standard metallic-roughness PBR path used by TRELLIS.2.

This is a data-contract correction. It does not change the Pixal3D model,
camera policy, variable-view policy, latent encoders, material representation,
or Blender material graph.

## Evidence and root cause

Pixal3D's maintainer states that the TRELLIS.2 version uses the same training
data as TRELLIS.2:

- https://github.com/TencentARC/Pixal3D/issues/30#issuecomment-4777141844

The TRELLIS.2 supplemental states that all curated assets are used for the
geometry/shape SC-VAE, while material SC-VAE training retains only assets that
use a standard metallic-roughness PBR workflow, producing a smaller material
subset:

- https://openaccess.thecvf.com/content/CVPR2026/supplemental/Xiang_Native_and_Compact_CVPR_2026_supplemental.pdf

The official toolkit keeps mesh and PBR preprocessing as separate steps, and
its PBR dumper deliberately raises `Material is not supported` for material
node graphs outside its supported workflow:

- https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md
- https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/blender_script/dump_pbr.py

The frozen 20-asset ObjaverseXL GitHub smoke shard exposed three independent
classes of failure:

1. Three OBJ condition renders failed only under the historical Blender-3
   importer call. Direct and integrated rerenders with the pinned Blender
   4.5.1 importer succeeded. These are implementation failures, not bad assets.
2. Three assets reach the official PBR parser but have unsupported material
   node graphs. These are valid geometry assets and expected material-filter
   outcomes.
3. Two GitHub repositories are unavailable. Their source assets cannot be
   reconstructed and remain global quarantines after the existing retry
   budget.

The current orchestrator records category 2 as a terminal asset failure. Its
single global eligible list then removes the asset from rendering, shape,
sparse-structure, and PBR processing. This is stricter than the cited training
method and discards valid geometry data.

## Considered approaches

### A. Family-scoped eligibility (selected)

Keep the frozen asset scope immutable, but track the exact assets eligible for
each prepared family. Unsupported PBR assets remain eligible for common,
shape, and sparse-structure families and are excluded only from PBR families.

This matches the paper's geometry/material split and the existing Pixal3D
dataset loaders, which independently require the metadata and files for the
training stage being loaded. The PBR loader additionally requires the matching
shape latent, so each PBR family is a subset of its corresponding shape family.

### B. Global all-or-nothing eligibility (rejected)

This is the current implementation. It is operationally simple but removes a
geometry-valid asset because its material graph is outside the standard PBR
subset. It conflicts with the cited data methodology.

### C. Convert unsupported material graphs (rejected)

Baking or rewriting arbitrary Blender node graphs could increase material
retention, but it would introduce a new preprocessing method not described by
Pixal3D or TRELLIS.2. It is explicitly outside the current paper-faithful goal.

## Immutable invariants

- Frozen batch SHA order and source identity do not change.
- The configured roots remain `/root/data2/pixal3d`,
  `/root/data3/pixal3d`, and `/root/node17/data/pixal3d`.
- Blender remains pinned to 4.5.1 and uses the Blender-4 OBJ importer.
- Render count, resolution, cameras, anchor views, latent models, resolutions,
  and dtype do not change.
- The pipeline still publishes all eight families atomically and audits every
  pack and raw archive.
- Provider-wide, process-control, resource, checksum, checkpoint, and
  infrastructure failures are never converted into asset exclusions.
- No unsupported shader is silently accepted or converted.

## Family membership

Every pack manifest retains the full ordered frozen scope in
`asset_sha256s`. A versioned manifest additionally records the exact ordered
`included_asset_sha256s` for that family.

| Family | Inclusion rule |
| --- | --- |
| `shape-R` | Valid condition render and valid shape latent at resolution `R` |
| `SS-64` | Valid condition render and valid SS latent derived from the configured highest-resolution shape latent |
| `PBR-R` | Included in `shape-R` and has a valid PBR latent at resolution `R` |
| `common` | Union of the assets included in at least one non-common prepared family |
| `raw` | Source package was downloaded and checksum-validated; unavailable provider assets are absent |

Consequently, `PBR-R` is always a subset of `shape-R`. A common pack may be a
superset of a particular stage pack; after extraction, the existing dataset
metadata filters select the required intersection for that stage.

For backward compatibility, `completed_count` becomes the number of included
assets for that manifest and `quarantined_count` is the frozen-scope count
minus the included count. The exact identities, rather than those scalar
counts, are authoritative. Old manifests without `included_asset_sha256s`
remain readable as schema-v1 publications in their producing commits; all new
publications use the new schema.

## Durable eligibility state

The quality ledger gains a schema-versioned `family_exclusions` mapping:

```json
{
  "<asset-sha>": {
    "PBR-256": {
      "category": "unsupported_shader",
      "stage": "dump_pbr",
      "reason": "Material is not supported",
      "attempts": 1
    }
  }
}
```

One root-cause event may exclude several dependent families. The stored record
is idempotent and preserves category, stage, reason, and attempts. The existing
`quarantine` mapping remains the global record for assets that have no usable
training family or whose source is unavailable.

Family exclusion is written atomically before a command validator returns
success. Resume reloads the ledger before constructing any command-specific
instance list, so a completed command cannot lose its exclusion state.

The global terminal `quality_outcomes` contract remains:

- `completed`: the asset is included in at least one prepared training family.
- `failure` or `schema_failure`: the asset is included in no prepared training
  family and is globally quarantined.

Expected PBR filtering is reported separately and does not count as a global
end-to-end failure. Schema and infrastructure failures retain their existing
gate behavior.

## Command routing and failure propagation

The runner continues to pass `--instances` files to the existing leaf workers,
but each file is derived from the command's family prerequisites instead of
one global all-or-nothing list.

| Stage | Input eligibility | Asset-level failure effect |
| --- | --- | --- |
| download / stage raw | Frozen assets not globally terminal | Provider-unavailable assets become global quarantine after retries |
| dump mesh / asset stats / condition render | Source-valid assets | Exclude every derived training family; final state is global quarantine |
| dump PBR | Source-valid assets not already excluded from all PBR families | Exclude all PBR resolutions; geometry families continue |
| dual grid `R` / shape encode `R` | Eligible for `shape-R` | Exclude `shape-R` and dependent `PBR-R`; at 1024 also exclude dependent `SS-64` |
| PBR voxel / PBR encode `R` | Eligible for `PBR-R` | Exclude only `PBR-R` |
| SS encode | Eligible for `SS-64` | Exclude only `SS-64` |
| final validation | Remaining family candidates | Validate each family independently, then derive common and global terminal state |

`asset_stats.py` already falls back to the mesh dump when `pbr_dumped` is not
true. It therefore continues to receive geometry candidates after PBR
exclusion without a new model or conversion path.

The PBR leaf stage must persist an asset-level failure record rather than only
printing an exception from a worker. The orchestrator classifies the official
`Material is not supported` condition as `unsupported_shader`; timeouts and
malformed outputs remain distinct and auditable. Missing output alone is not
used to invent a shader category.

## Packing, audit, and handoff

Pack construction receives a per-family SHA mapping. It builds members only
for that family's included identities and writes both frozen scope and included
scope into the manifest. Publication and reuse checks compare both scopes.

Audit verifies:

- every included identity belongs to the frozen scope;
- the included list is sorted, unique, and consistent with member paths;
- every `PBR-R` identity is also included in `shape-R`;
- `common` equals the union of non-common family identities;
- family counts agree with exact included identities;
- exclusions in the quality ledger explain omitted identities;
- raw archive identities and checksums remain independently valid.

Reports expose global quarantine counts and per-family included/excluded counts
separately. The training handoff retains the existing Stage 1/2/3 extraction
layout, but records exact family memberships so loaders cannot assume that all
eight packs have identical asset sets.

## Recovery and migration

The interrupted old-contract ObjaverseXL GitHub rerun must not be resumed with
the new code. Its watcher was stopped and its checkpoint-compatible partial
outputs remain preserved. Before the new smoke run, move its checkpoint,
ledger, partial local outputs, and any partial publications into the existing
timestamped recovery area with a checksum inventory. Do not alter the frozen
batch text files.

Rerun the same 20 frozen SHA values from clean execution state under one new
tool commit. Based on the diagnosed sample, the expected result is:

- all three historical OBJ-render false failures pass geometry processing;
- the three unsupported-material assets remain in geometry families but are
  absent from PBR families with `unsupported_shader` records;
- the two unavailable repositories remain globally quarantined after retries;
- no pack combines artifacts produced by different tool commits.

These counts are expectations, not hard-coded acceptance criteria. The new
smoke and audit outputs are authoritative.

## Test and verification plan

Implementation follows test-driven development.

1. Add failing leaf-worker tests proving that PBR errors persist the asset SHA,
   category evidence, and reason without accepting a bad pickle.
2. Add failing runner tests proving that an unsupported PBR asset remains in
   shape and SS command lists and is removed from PBR command lists after
   resume.
3. Add failing dependency tests for per-resolution shape, PBR, and SS
   exclusions.
4. Add failing pack tests for frozen scope, included scope, family counts,
   backward-compatible manifest reads, and publication reuse checks.
5. Add failing integration tests with one full-PBR asset and one
   geometry-only asset. Both must appear in geometry packs and only the first
   may appear in PBR packs.
6. Add audit/report tests for family subset and exclusion-ledger invariants.
7. Run focused tests, then the complete `tests/data_toolkit` suite.
8. Preserve the interrupted old-contract state, rerun the frozen 20-asset
   ObjaverseXL GitHub smoke, and run its exact-commit audit.

## Out of scope

- Material baking, node rewriting, or fallback shader conversion.
- Model architecture or feature-fusion changes.
- Camera, variable-K, anchor, view-alignment, or latent-format changes.
- Lowering checksum, schema, infrastructure, or resource safety checks.
- Changing the frozen sample identities to improve the smoke pass rate.
