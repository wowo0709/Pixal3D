# Asset Quarantine Ledger Design

## Goal

Record asset-intrinsic failures separately from infrastructure failures so that
only validated assets enter preprocessing outputs and training handoff.

## Policy

- `completed` assets remain eligible for downstream stages.
- Asset-intrinsic failures become `quarantined` and are excluded from all later
  stages in the batch and from training handoff.
- Infrastructure, provider-wide, resource, checkpoint, and process-control
  failures are never silently quarantined; they retain retry/stop behavior.
- Frozen manifests preserve the original sample scope. Pack members contain
  only completed assets, while quarantine records preserve excluded identities.

## Durable record

Extend the quality ledger with a per-asset quarantine record containing the
asset SHA-256, category, stage, reason, attempt count, source identifier, and
timestamp. Existing `failure` outcomes remain backward-compatible and are
interpreted as quarantined when no richer record exists.

## Categories

Initial categories are `unsupported_format`, `unsupported_shader`,
`missing_required_metadata`, `missing_render_transform`, `missing_source`, and
`provider_asset_unavailable`. The runner only writes these records from
asset-level validation paths; command/infrastructure exceptions do not create
them.

## Verification

Tests must prove that quarantine records survive checkpoint resume, are excluded
from staging and pack members, retain frozen scope identity, and do not convert
infrastructure failures into asset quarantine.
