"""Write guarded result prose for both final papers from audited TEST evidence.

The generator intentionally describes only comparisons certified by the claim
audit.  It never promotes FID/KID ranks to inferential claims and never treats
the bundled Selected-versus-V1 contrast as an isolated module effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METRICS = ('LPIPS', 'DreamSim', 'CLIP', 'PSNR', 'SSIM', 'M_IoU', 'CD', 'FScore')
LABEL = {'LPIPS': 'LPIPS', 'DreamSim': 'DreamSim', 'CLIP': 'CLIP',
         'PSNR': 'PSNR', 'SSIM': 'SSIM', 'M_IoU': 'mask IoU',
         'CD': 'Chamfer distance', 'FScore': 'F@0.02'}
V1 = 'ours_v1_source_m0'
SELECTED = 'ours_selected_source_m0'
OFFICIAL = 'pixal3d_official_rgb1_source_m0'


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_list(names: list[str]) -> str:
    labels = [LABEL[name] for name in names]
    if not labels:
        return 'none'
    if len(labels) == 1:
        return labels[0]
    return ', '.join(labels[:-1]) + ', and ' + labels[-1]


def groups(rows: dict) -> tuple[list[str], list[str], list[str]]:
    output = []
    for state in ('supported_favorable', 'supported_unfavorable', 'unresolved'):
        output.append([name for name in METRICS
                       if rows[name]['classification'] == state])
    return tuple(output)  # type: ignore[return-value]


def fmt(metric: str, value: float) -> str:
    if metric == 'PSNR':
        return f'{value:.3f}'
    if metric in {'LPIPS', 'DreamSim', 'CLIP', 'SSIM', 'M_IoU', 'FScore'}:
        return f'{value:.4f}'
    if metric == 'CD':
        return f'{value:.5f}'
    if metric == 'FID':
        return f'{value:.2f}'
    if metric == 'KID':
        return f'{1000 * value:.3f}'
    raise KeyError(metric)


def outcome_sentence(label: str, rows: dict) -> str:
    favorable, unfavorable, unresolved = groups(rows)
    parts = []
    if favorable:
        parts.append('supports a favorable difference in ' + metric_list(favorable))
    else:
        parts.append('supports no favorable endpoint')
    if unfavorable:
        parts.append('supports an unfavorable difference in ' + metric_list(unfavorable))
    if unresolved:
        parts.append('leaves ' + metric_list(unresolved) + ' unresolved')
    return label + ' ' + '; '.join(parts) + '.'


def favorable_clause(condition: str, names: list[str]) -> str:
    if names:
        return condition + ' supports favorable differences in ' + metric_list(names)
    return condition + ' supports no favorable predeclared endpoint'


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError('result slot must be new: ' + str(path))
    path.write_text(text.strip() + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-table-json', type=Path, required=True)
    parser.add_argument('--raw-table-json', type=Path, required=True)
    parser.add_argument('--paired-json', type=Path, required=True)
    parser.add_argument('--raw-paired-json', type=Path, required=True)
    parser.add_argument('--official-contrasts-json', type=Path, required=True)
    parser.add_argument('--claim-audit-json', type=Path, required=True)
    parser.add_argument('--common-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    table = read(args.main_table_json)
    raw = read(args.raw_table_json)
    paired = read(args.paired_json)
    raw_paired = read(args.raw_paired_json)
    official = read(args.official_contrasts_json)
    audit = read(args.claim_audit_json)
    common = read(args.common_manifest)
    n = common['samples']
    if (audit['status'] != 'PASS_FAIR_TEST_CLAIM_AUDIT'
            or audit['samples'] != n or table['samples'] != n or raw['samples'] != n
            or paired['samples'] != n or raw_paired['samples'] != n
            or official['samples'] != n or len(table['methods']) != 10
            or len(table['metrics']) != 10
            or set(audit['comparisons']['selected_vs_v1']) != set(METRICS)
            or set(audit['comparisons']['raw_selected_vs_v1']) != set(METRICS)
            or set(audit['comparisons']['ours_vs_official']) != {V1, SELECTED}):
        raise ValueError('complete, audited, common-population inputs required')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reg_rows = audit['comparisons']['selected_vs_v1']
    raw_rows = audit['comparisons']['raw_selected_vs_v1']
    selected_official = audit['comparisons']['ours_vs_official'][SELECTED]
    raw_favorable, _, _ = groups(raw_rows)
    reg_favorable, _, _ = groups(reg_rows)
    raw_favorable_clause = favorable_clause('raw evaluation', raw_favorable)
    reg_favorable_clause = favorable_clause(
        'source-M0-registered evaluation', reg_favorable)
    raw_sentence = outcome_sentence('For raw outputs, Selected versus V1', raw_rows)
    reg_sentence = outcome_sentence(
        'After source-M0-only Sim(3), Selected versus V1', reg_rows)
    official_sentence = outcome_sentence(
        'In the input-asymmetric system comparison, Selected versus official RGB1',
        selected_official)
    supported = raw_favorable + [name for name in reg_favorable
                                if name not in raw_favorable]

    selected_raw = raw['values'][SELECTED]
    selected_reg = table['values'][SELECTED]
    official_raw = raw['values'][OFFICIAL]
    official_reg = table['values'][OFFICIAL]
    registration = (
        f"For Selected, registration changes mask IoU from "
        f"{fmt('M_IoU', selected_raw['M_IoU'])} to "
        f"{fmt('M_IoU', selected_reg['M_IoU'])} and Chamfer distance from "
        f"{fmt('CD', selected_raw['CD'])} to {fmt('CD', selected_reg['CD'])}; "
        f"for official RGB1, mask IoU changes from "
        f"{fmt('M_IoU', official_raw['M_IoU'])} to "
        f"{fmt('M_IoU', official_reg['M_IoU'])}."
    )
    distributions = (
        f"The registered Selected distribution has FID "
        f"{fmt('FID', selected_reg['FID'])} and KID$\\times 10^3$ "
        f"{fmt('KID', selected_reg['KID'])}; these are descriptive point "
        f"estimates rather than paired inferential endpoints."
    )

    abstract = f"""
On the frozen common-success set of {n} TEST edits, family-wise adjusted
paired Selected-versus-V1 intervals show that {raw_favorable_clause}, while
{reg_favorable_clause}. Every condition--endpoint direction is reported
separately as favorable, unfavorable, or unresolved.
The official RGB1 comparison is a complete-system comparison with different
available image evidence, not an isolated module test.
"""
    results = f"""
The two ten-method tables evaluate the same {n} edits. {raw_sentence}
{reg_sentence} These paired conclusions use source-cluster resampling and one
16-endpoint family spanning the two alignment conditions. {official_sentence}
That comparison is explicitly system-level: official RGB1 receives the edited
target and camera, whereas Ours additionally receives three source views.
{registration} {distributions} Native success counts remain separate from the
common-success denominator. The Selected--V1 test bundles correspondence,
ray sharing, attention, and retargeting, so it does not identify an isolated
component effect.
"""
    discussion = f"""
Raw and source-M0-registered outputs lead to different point estimates, and
neither alignment is selected after observing the outcome. {registration}
Consequently, registration is treated as an evaluation factor rather than a
hidden correction. Adjusted intervals, rather than rank or mean direction,
govern the words ``improves'' and ``worsens.'' FID/KID and cross-method ranks
remain descriptive. The official-control contrast measures systems with
different supported inputs, while the matched V1 contrast measures the bundled
Selected composition. It therefore cannot establish that semantic matching,
score weighting, ray sharing, or retargeting is individually necessary.
"""
    conclusion = f"""
We formulate ray-shared semantic evidence retargeting inside a pretrained
projective carrier and freeze the geometry and appearance choices before
TEST. On {n} common-success edits, the audited Selected--V1 comparison
shows that {raw_favorable_clause}, while {reg_favorable_clause}; unsupported
endpoint claims remain explicitly unresolved. Reporting both alignments exposes
the effect of post-generation
registration, while the official RGB1 comparison is retained as a labeled
input-asymmetric system comparison.
"""

    cvpr_results = f"""
The final comparison uses the same {n} common-success edits for every row.
{raw_sentence}
{reg_sentence} The family-wise adjustment spans both alignment conditions.
{official_sentence} Official RGB1 uses T1+C and Ours uses T1+S3+C, so this is
a complete-system comparison rather than a module effect. {registration}
{distributions} The bundled Selected--V1 contrast cannot assign its outcome to
one correspondence, attention, ray-sharing, or retargeting component.
"""

    slots = {
        'unified-abstract-findings.tex': abstract,
        'unified-results.tex': results,
        'unified-discussion.tex': discussion,
        'unified-conclusion.tex': conclusion,
        'cvpr2027-abstract-findings.tex': abstract,
        'cvpr2027-results.tex': cvpr_results,
        'cvpr2027-conclusion.tex': conclusion,
    }
    for name, text in slots.items():
        write_new(args.output_dir / name, text)
    provenance = {
        'status': 'PASS_AUDITED_FAIR_FINDINGS_WRITTEN',
        'samples': n,
        'inputs': {
            'main_table_json': digest(args.main_table_json),
            'raw_table_json': digest(args.raw_table_json),
            'paired_json': digest(args.paired_json),
            'raw_paired_json': digest(args.raw_paired_json),
            'official_contrasts_json': digest(args.official_contrasts_json),
            'claim_audit_json': digest(args.claim_audit_json),
            'common_manifest': digest(args.common_manifest),
        },
        'files': {name: digest(args.output_dir / name) for name in slots},
        'supported_selected_vs_v1': supported,
    }
    write_new(args.output_dir / 'fair-findings-provenance.json',
              json.dumps(provenance, indent=2))
    print(json.dumps({'status': 'PASS_AUDITED_FAIR_FINDINGS_WRITTEN',
                      'samples': n, 'files': sorted(slots),
                      'supported_selected_vs_v1': supported}))


if __name__ == '__main__':
    main()
