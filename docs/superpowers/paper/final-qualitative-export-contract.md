# Final qualitative export implementation

The original [selection protocol](qualitative-selection-protocol.md) is unchanged.
`experiments.deformnext_retargeting.export_final_qualitative` exports its fixed
examples and conditional lower medians from the actual frozen ten-metric run.
It performs no neural forward, model choice, mesh rendering, or metric recomputation.

Before TEST, merge the actual `final-paper-export-readiness-v424/freeze-requirements.json`
packet (extending v420) into the final evaluation protocol. Assign `qualitative_selection_plan`
and **update the existing `source_SHA256` mapping** with the packet's entries;
retain every original generation/render/metric source pin. The exporter rejects
a freeze that did not already contain its code and the original closed v355 declaration.
The packet also supplies a fresh read-only local/remote TEST manifest receipt.
It does not create a model freeze, final context, or TEST job.

After actual final metrics are complete, create a closed context with schema
`final-qualitative-closed-TEST508-inputs-v1` and the following actual inputs:

| Field | Required value |
|---|---|
| `repo`, `canonical_local_root` | Actual checkout and canonical result root |
| `CPU_affinity` | Assigned metadata cores; this campaign uses 92/93 |
| `source_SHA256` | Current `required_sources()` entries, already bound by the pre-TEST freeze |
| `source_inputs` | Closed specs for `parent_freeze`, `control_freeze`, `selection_plan`, `test_manifest`, `eight_metrics`, `ten_metrics` |
| Each closed spec | Explicit `path`, `SHA256`, and actual producer `receipt` |
| `output` | New owned canonical directory; existing outputs are not replaced |
| `name`, `manifest_directory`, `closure_command` | Actual unique closure identity, owned manifest directory, and existing file4 SHA verifier command |

Run on the metadata CPU cores with `CUDA_VISIBLE_DEVICES=''`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  taskset -c 92,93 nice -n 10 python -m experiments.deformnext_retargeting.export_final_qualitative \
  --plan /actual/closed/context.json
```

The exporter checks the freeze before opening quality reports, all 508/134-source
identities, all 1,524 scored camera observations per physical model, and exact
prediction/reference float and official uint8 image hashes. It reuses existing
closed images at target8/source9/10, displays every deduplicated model and common
M1 original-material reference, and annotates the already computed source
LPIPS/DreamSim and surface CD/F-Score. PDF/PNG outputs use a common 7.2-inch
canvas at 300dpi. Labels must fit; no model-specific crop, alignment or rerender is used.

Outputs are `selection.json`, PDF/PNG per distinct selected case, `figure-captions.txt`,
`audit.json`, and an actual file4 closure. Empty categories stay empty and overlapping
categories share one example. Counts describe eligibility, not population success rates.

Thirty synthetic metadata/pixel contract tests passed. The changed full six-column
layout/image join also passed its targeted check and was visually reviewed.
The actual 508 metadata, original policy and two outcome-blind IDs exactly match
v355, and the original manifest's current remote/local SHA and size agree.
**Actual final TEST images and the complete exporter run remain pending.**

## Main table from the same final reports

`experiments.deformnext_retargeting.export_final_main_table` uses schema
`final-main-paper-table-closed-inputs-v1`. It uses the same context fields above,
with closed specs for `parent_freeze`, `control_freeze`, `selection_plan`,
`eight_metrics`, and `ten_metrics`. Its `required_sources()` pins are included
in the expanded v424 packet; retain all existing numerical source pins when merging.

The entry accepts only a complete frozen TEST508 report. It checks all ten fields,
the original camera-role means against the complete per-asset rows, raw signed KID,
the official 1,524-image pools, and the exact adjusted interval record from the
eight-metric producer. Model rows follow the frozen physical order; a selected
baseline is not duplicated and cells are not bolded by favorable point estimates.

Outputs are `main-table.tex`, `comparison-evidence.json`, unrounded/formatted values
in `audit.json`, and writing-scope notes. Eight pairwise endpoints keep their
predeclared family; FID/KID remain descriptive. An interval crossing zero is
`uncertain_not_equivalent`, independently of its point estimate. These files help
write Results/Discussion and do not automatically establish module necessity or
complete manuscript writing. The existing v312 SS/cascade ablation exporter is reused.

Fifteen synthetic metadata contract tests passed, including early rejection before
TEST quality when the source was not frozen. The strengthened immutability checks
also passed. A clearly labelled synthetic Main table was compiled and visually
reviewed only for typesetting. **No actual TEST Main table has been exported.**
