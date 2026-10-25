# Final qualitative example protocol

Declared before TEST model-quality evaluation on 2026-10-25. This policy
does not select a model, alter an evaluation population, or establish a
success rate. Final images remain pending.

## Outcome-blind examples

Use only `sample_id`, `source_uid`, `source_cluster_id`, and `split` from
the verified native TEST508 manifest (508 examples, 134 source clusters,
SHA256 `5e67564a299eb9ef400c504fae00c9ebb8478a6571b97254c1098736c728eb05`).

1. Order distinct source clusters by the hexadecimal SHA256 of UTF-8
   `paper-qualitative-source-v1:` followed by `source_cluster_id`. Break ties
   by the original cluster string. Retain the first two clusters.
2. Within each cluster, retain the first example ordered by SHA256 of
   `paper-qualitative-case-v1:` followed by `sample_id`, then the original
   sample ID for ties. Keep these examples even if final performance is poor.

The actual selected sample IDs are:

- `shard003_6e4005b2a9ac445c8fb4295bc5d29f9a_d2_fast300fpssupplyv1_PA_target08`
- `shard001_b5e62e1a306e4cb7846bd51a75ac0d9a_d1_fast300fpssupplyv1_balanced_JN_L_target08`

Both source identities and the full metadata-only population are retained
in `final-qualitative-outcome-blind-plan-v355/selection.json` under the
canonical unified-module-pipeline result root. Original remote/local manifest
SHA256 and size agree. No prediction, model-quality metric or geometry array
was inspected for this selection.

## Conditional diagnostic examples

After the frozen TEST evaluation, compare the selected alias with `RETARGET`.
If the selected alias itself is `RETARGET`, compare with `V1`. This choice
depends on frozen model identity, not on TEST rankings.

For every example use selected-minus-comparator differences. CD and F-Score
are surface metrics; LPIPS and DreamSim average source cameras9/10. Preserve
the original metric values. Eligibility and ordering are:

| Category | Eligibility | Ascending ordering key |
|---|---|---|
| Joint improvement | ΔCD<0, ΔFScore>0, ΔLPIPS<0, ΔDreamSim<0 | ΔFScore |
| Geometry loss | ΔCD>0, ΔFScore<0 | ΔFScore |
| Appearance loss | ΔLPIPS>0, ΔDreamSim>0 | ΔLPIPS |

Within each category sort by its numeric key, then the case hash defined
above and sample ID. Select index `(n-1)//2`, the lower median. Count all
eligible cases before selection for an audit, but do not report the selected
examples as a population success rate. If a category is empty, state that no
example meets that specific condition; do not change the condition or claim
universal improvement. Merge labels when the same case is selected more
than once. Do not substitute cases merely to fill a panel.

The outcome-blind panel and conditional panels may be placed in the main
paper and supplement for space. Both selected IDs and any conditional
eligibility/choice must be disclosed; they are distinct selection scopes.

## Common-image comparison

Show identical GT and every deduplicated physical model: source-only zero
shot4, edited-target zero shot4, V1, retargeting and the selected model when
distinct. Use existing closed scored-image archives at target8/source9/10,
with the same camera, full-frame image conversion, scale and background.
Join actual scored prediction/reference image identities before display.
Do not independently render meshes, align an output, or crop each model
differently. The diagnostic figures do not substitute for ten-metric main
results, cluster intervals, held-out confirmation, or genuine failure analysis.
