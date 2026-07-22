# Work-Conserving Source Priority Design

## Goal

Production preprocessing claims batches in this priority order:

1. `ABO`
2. `3D-FUTURE`
3. `HSSD`
4. `ObjaverseXL_sketchfab`

The priority is work-conserving rather than a strict source barrier. A worker
may claim from the next source when every nonterminal batch in a higher-priority
source is already leased by another worker. Existing leases are never revoked
or interrupted when the priority changes.

## Constraints

- Preserve the frozen `units.json` manifest and its config hash.
- Preserve completed, failed, history, checkpoint, archive, and pack records.
- Keep source-local batch ordering identical to the frozen manifest.
- Keep node16 and node17 on one atomic shared queue without duplicate claims.
- Treat both completed and terminally quarantined batches as terminal.
- Reclaim a stale higher-priority lease before considering a lower-priority
  source, subject to the existing attempt budget.
- Do not introduce a strict barrier that idles a worker while a lower-priority
  batch is claimable.

## Queue Priority Artifact

The queue root gains `priority.json`, separate from immutable `units.json`.
Its schema is:

```json
{
  "schema_version": 1,
  "sources": ["ABO", "3D-FUTURE", "HSSD", "ObjaverseXL_sketchfab"],
  "updated_at": "2026-07-22T12:00:00+00:00"
}
```

The source list must contain every source present in `units.json` exactly once,
with no unknown source, omission, duplicate, or empty value. The artifact is
written atomically using the queue's existing no-follow JSON persistence. A
missing artifact preserves legacy behavior by using first-source appearance in
the frozen manifest. A malformed artifact stops a claim with an infrastructure
error instead of silently reverting to another order.

Priority is operational scheduling state, not dataset identity, so it does not
participate in the frozen config hash. It can be changed while production is
running without rebuilding shards or invalidating evidence.

## Claim Algorithm

Each `claim()` call reads the current priority artifact, creates a stable view
of frozen units ordered by `(source priority, original manifest position)`, and
then applies the existing terminal, lease, stale-lease, and attempt-budget logic
in that order.

For two workers and one remaining ABO batch:

- The first worker atomically leases that ABO batch.
- The second worker sees no claimable ABO batch and continues scanning
  `3D-FUTURE`.
- Atomic lease-directory creation remains the collision boundary, so concurrent
  scans cannot duplicate a claim.

When a higher-priority source has a pending or stale-reclaimable batch, no lower
source is selected. When all of its nonterminal batches have live leases, the
scan continues to the next source. This is the requested priority behavior and
not a completion barrier.

## CLI and Status

The queue CLI adds:

```bash
python -m data_toolkit.pipeline.cli queue \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --action prioritize \
  --sources ABO,3D-FUTURE,HSSD,ObjaverseXL_sketchfab
```

`queue --action status` includes the effective source priority so the live
policy can be audited. `init` remains responsible only for freezing units;
`prioritize` requires an already initialized, config-matching queue.

## Deployment and Cutover

1. Implement and verify the queue and CLI changes locally.
2. Deploy the same commit to node16 and node17.
3. Write `priority.json` once on the shared queue.
4. Mark each node draining without terminating its current lease.
5. At each batch boundary, restart or reactivate its supervised worker on the
   new commit.
6. Verify that the next two claims use the highest-priority available units and
   have distinct lease tokens.

The currently running node16 ABO batch and node17 ObjaverseXL batch remain
owned by their current workers. No local scratch or checkpoint is reset.

## Failure Handling

- Atomic file creation continues to resolve simultaneous claims.
- A corrupt or incomplete priority artifact prevents new claims and leaves
  existing leases untouched.
- Worker failure releases only its owned batch under the existing three-attempt
  policy; the next claim applies the same source priority again.
- Draining, cordoning, removal, and later activation retain their existing
  supervisor semantics.
- A source containing unusable assets can terminate batches as failed after the
  existing attempt budget; those batches do not permanently block lower
  sources.

## Testing

Automated tests cover:

- priority validation and atomic round-trip;
- legacy manifest order when `priority.json` is absent;
- stable ABO-first ordering independent of manifest source order;
- a second worker advancing to `3D-FUTURE` when the final ABO batch is live;
- stale ABO lease recovery before a lower-priority claim;
- completed and terminally failed units being skipped;
- CLI priority persistence and status reporting;
- unchanged duplicate-claim and retry behavior.

The full `tests/data_toolkit` suite must pass before deployment.

## ETA Model

At design time the remaining frozen scope is:

| Source | Remaining batches | Remaining assets |
|---|---:|---:|
| ABO | 16 | 3,973 |
| 3D-FUTURE | 36 | 8,960 |
| HSSD | 25 | 6,158 |
| ObjaverseXL_sketchfab | 672 | 168,051 |

Observed valid 256-asset batches are dominated by 512px eight-view Blender
rendering. A planning range of 1-2 hours on node17 and 1.5-3 hours on node16 is
used until at least three valid completed batches exist per source. With both
nodes available and work-conserving overlap at source tails, the current
planning windows are:

| Completion scope | Planning window |
|---|---:|
| ABO | 10-20 hours |
| Through 3D-FUTURE | 30-61 hours |
| Through HSSD | 44-89 hours |
| Entire scope through ObjaverseXL | 19-37 days |

These are wall-clock ranges, not guarantees. Early quarantine shortens them;
repeated 900-second Blender timeouts move them toward the upper bound. ETA
reporting should be recalculated from rolling per-source terminal batch
durations after deployment rather than treating zero-valid-asset batches as
representative throughput.
