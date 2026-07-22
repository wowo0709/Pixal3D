# Full-VRAM GPU Runtime Policy Design

## Goal

Keep the encoder GPU-memory target at 80% while changing the internal GPU
admission hard ceiling to the physical device capacity, 100%. Apply the policy
to node16, node17, and later registered workers without changing the frozen
production queue, canonical config hash, smoke/pilot evidence, completed work,
or live leases.

The scheduler may admit internal reservations whose sum is exactly 100%. It
does not reserve emergency headroom and does not pre-subtract memory owned by
unrelated processes. CUDA, Blender, and the existing batch retry/quarantine
path remain the final enforcement when actual allocations exceed physical
VRAM.

## Constraints

- Preserve `units.json` byte-for-byte and preserve its current config hash.
- Preserve `priority.json`, all leases, history, terminal records, checkpoints,
  packs, archives, and quality ledgers.
- Keep `gpu_memory_target_percent` at exactly 80 for this production scope.
- Set the effective `gpu_memory_hard_percent` to exactly 100.
- Permit an internal reservation sum equal to the hard ceiling.
- Do not interrupt a live batch merely to change the policy.
- Make the policy shared and auditable so dynamically added nodes cannot
  silently use a different ceiling.
- Keep existing CUDA OOM adaptive micro-batch reduction and batch-level retry
  behavior unchanged.

## Shared Runtime Policy Artifact

The shared runtime root gains
`control/runtime/gpu_policy.json`, separate from the canonical YAML and the
frozen work-queue manifest:

```json
{
  "schema_version": 1,
  "target_percent": 80,
  "hard_percent": 100,
  "updated_at": "2026-07-22T13:00:00+00:00"
}
```

The artifact is written atomically using the existing no-follow JSON
persistence utilities. Both percentages must be integers, the target must
equal the canonical config target, and `target_percent < hard_percent <= 100`
must hold. For the current production scope this admits only target 80 with a
hard value from 81 through 100; the operator command will write 100.

A missing artifact preserves legacy behavior by using the canonical config
values, currently target 80 and hard 90. A malformed, unsupported, or
canonical-target-mismatched artifact stops a new worker before it claims a
batch. It does not revoke an existing lease.

The artifact is operational scheduling state rather than dataset identity. It
does not participate in `PipelineConfig.config_hash()` and does not require
queue reinitialization or smoke/pilot evidence regeneration.

## Runtime Application

Worker startup reads the shared artifact after resolving its registration and
before constructing the node-local `WorkerExecutionConfig`. The execution
config replaces only the two effective parallelism percentages with the
validated runtime values; `config_hash()` continues to delegate to the
unchanged canonical config.

Every scheduler created by that worker receives target 80 and hard 100.
Encoder commands continue receiving
`--gpu_memory_target_percent 80`. `NodeResourceBroker` and
`DynamicResourceBroker` reject a request only when the resulting internal
reservation would be greater than the hard ceiling. An exact total of 100 is
therefore admitted.

`NodeResourceBroker`, which is used inside the current production worker, does
not add new per-process attribution or pre-subtract unrelated allocations from
its internal reservation sum. `DynamicResourceBroker` retains its existing
externally observed memory accounting and may admit a request whose observed
plus reserved total is exactly 100%. This matches the approved ceiling while
preserving both brokers' established responsibilities. Existing encoder
CUDA-OOM handling halves a micro-batch; other command failures release only
the owned batch and use the existing three-attempt terminal policy.

## CLI and Status

The workers CLI adds:

```bash
python -m data_toolkit.pipeline.cli workers \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --action set-gpu-policy \
  --gpu-target-percent 80 \
  --gpu-hard-percent 100
```

`set-gpu-policy` writes one shared artifact and does not require a node ID.
Other worker actions reject the GPU-policy-only arguments. `workers --action
status` adds these fields to every registered node entry:

```json
{
  "gpu_memory_target_percent": 80,
  "gpu_memory_hard_percent": 100
}
```

The repeated per-node fields make it explicit that all active and dynamically
registered nodes consume the same shared effective policy while retaining the
existing node-keyed status shape.

## Cutover

1. Implement and run the complete `tests/data_toolkit` suite locally.
2. Drain each node without terminating its current lease.
3. Deploy identical runtime source and commit identity to node16 and node17.
4. Write `gpu_policy.json` once on the shared data2 root.
5. At each node's batch boundary, restart or allow its supervisor to launch a
   new worker on the deployed code and reactivate that node.
6. Verify worker status reports target 80/hard 100 and the next lease has a
   post-deployment worker process.
7. Observe `nvidia-smi` during rendering and encoding. OOM/retry is accepted by
   this policy; duplicate claims, silent lease loss, or a worker using hard 90
   are not.

Nodes cut over independently. A node that finishes first resumes immediately;
it does not wait for the other node. The existing source priority remains
`ABO`, `3D-FUTURE`, `HSSD`, `ObjaverseXL_sketchfab`.

## Failure Handling

- Atomic replacement prevents readers from seeing a partial policy artifact.
- Invalid policy prevents new claims and leaves current leases untouched.
- A worker started without the artifact uses legacy target 80/hard 90; status
  exposes that fallback, so deployment can verify before activation.
- CUDA OOM retains encoder micro-batch halving. If a command still fails, the
  worker releases its lease using the current retry/terminal-failure rules.
- Removing or draining one node does not change the shared policy or stop the
  other node.
- Newly registered nodes read the policy at worker startup and therefore need
  no per-node environment override.

## Testing

Automated tests cover:

- atomic policy round-trip and legacy fallback;
- rejection of non-integer, target-mismatched, target-not-below-hard, and
  above-100 values;
- malformed policy preventing worker execution before claim;
- canonical config hash remaining unchanged under the runtime override;
- node-local execution config exposing target 80/hard 100;
- exact 100% reservation being admitted and greater-than-100 being rejected in
  both resource brokers;
- CLI persistence, argument validation, and per-node status reporting;
- unchanged atomic claims, drain/activate behavior, and OOM retry behavior.

The full `tests/data_toolkit` suite must pass before either node starts a new
worker with this policy.
