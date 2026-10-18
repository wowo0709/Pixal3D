"""Bind completed FP32 feature diagnostics to the actual Full model and panel.

These inference component swaps never establish a generated-SS quality gain.
The original experiment can omit this later supplement; partial or conflicting
supplements are errors. No model binaries are copied to the reporting artifact.
"""
from pathlib import Path
import hashlib
import json
import math


PREFIX = 'diagnostics/matching_prior_drift/development128'
DISTRIBUTIONS = ('cached_static_T005', 'cached_static_T05', 'initializer_prior_T05',
                 'trained_prior_T05', 'trained_total_T05',
                 'trained_decoder_initial_head_T05', 'initial_decoder_trained_head_T05')
FIELDS = ('candidate_argmax_changed_fraction', 'mean_KL_candidate_vs_reference',
          'mean_total_variation', 'value_aggregate_relative_RMS_vs_reference',
          'value_aggregate_mean_cosine')
CONTRASTS = tuple(a+'-vs-'+b for a, b in [
    ('trained_prior_T05', 'initializer_prior_T05'),
    ('trained_total_T05', 'trained_prior_T05'),
    ('trained_total_T05', 'cached_static_T05'),
    ('initializer_prior_T05', 'cached_static_T05'),
    ('cached_static_T05', 'cached_static_T005'),
    ('trained_decoder_initial_head_T05', 'initializer_prior_T05'),
    ('initial_decoder_trained_head_T05', 'initializer_prior_T05'),
    ('trained_prior_T05', 'trained_decoder_initial_head_T05'),
    ('trained_prior_T05', 'initial_decoder_trained_head_T05')])
SCOPE = ('FP32 CPU inference on development128/93sources/9883 active queries. '
         'Saved-component swaps and feature distributions;argmax disagreement '
         'is not accuracy. These are not independently trained ablations or '
         'generated-SS quality effects. No additional121 or optimizer updates.')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_reports(base, hybrid, expected_rows, full_qa):
    """Validate identity and statistic scope without loading any neural model."""
    identity = [(r['sample_id'], r['source_cluster_id']) for r in expected_rows]
    if len(identity) != 128 or len({r[0] for r in identity}) != 128 or len({r[1] for r in identity}) != 93:
        raise ValueError('actual complete development panel required')
    statuses = ['PASS_FIXED128_SAVED_MATCHING_PRIOR_DIAGNOSIS',
                'PASS_FIXED128_SAVED_MATCHING_PRIOR_HYBRID_DIAGNOSIS']
    for report, status in zip([base, hybrid], statuses):
        if (report['status'] != status or report['cases'] != 128 or report['source_clusters'] != 93
            or report['active_queries'] != 9883 or report['device'] != 'CPU_FP32'
            or report['GPU_used'] is not False or report['optimizer_updates'] != 0):
            raise ValueError('completed inference-only diagnostic scope differs')
        if [(r['sample_id'], r['source_cluster_id']) for r in report['inputs']] != identity:
            raise ValueError('diagnostic case/source order differs from actual Full evaluation')
        counts = report['per_case_context_and_score_changes']
        if ([r['sample_id'] for r in counts] != [r[0] for r in identity]
            or any(not 1 <= r['active_queries'] <= 1024 for r in counts)
            or sum(r['active_queries'] for r in counts) != 9883):
            raise ValueError('diagnostic active query accounting differs')
        if (report['weights']['initializer'] != full_qa['selected_head']
            or report['weights']['trained_conditioner'] != full_qa['trained_conditioner']
            or report['recipe'] != full_qa['source_recipe']):
            raise ValueError('diagnostic uses a different saved Full model or recipe')
    for key in ['inputs', 'weights', 'recipe', 'original_saved_QA', 'panel', 'runtime_source']:
        if base[key] != hybrid[key]:
            raise ValueError('hybrid diagnostic changed inputs or saved weights')
    if (set(base['attention_distributions']) != set(DISTRIBUTIONS[:5])
        or set(hybrid['attention_distributions']) != set(DISTRIBUTIONS)
        or set(base['contrasts']) != set(CONTRASTS[:5])
        or set(hybrid['contrasts']) != set(CONTRASTS)):
        raise ValueError('required saved-component contrasts are incomplete')
    replay = hybrid['full128_original_diagnostic_replay']
    if replay['status'] != 'PASS' or not 0 <= replay['max_absolute_statistic_difference'] <= 2e-6:
        raise ValueError('hybrid original-statistics replay is incomplete')
    for name, stats in base['attention_distributions'].items():
        if any(abs(v-hybrid['attention_distributions'][name][key]) > 2e-6 for key, v in stats.items()):
            raise ValueError('hybrid changed original attention statistics')
    for name, values in base['contrasts'].items():
        if any(abs(values[key]-hybrid['contrasts'][name][key]) > 2e-6 for key in FIELDS):
            raise ValueError('hybrid changed original contrast statistics')
    metrics = {}
    distribution_rows = []
    for name in DISTRIBUTIONS:
        values = hybrid['attention_distributions'][name]
        bounds = {'mean_max_weight': (1/64-1e-6, 1+1e-6),
                  'mean_entropy_nats': (-1e-6, math.log(64)+1e-6),
                  'mean_effective_candidates': (1-1e-6, 64+1e-6)}
        for key, (lower, upper) in bounds.items():
            value = values[key]
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError('invalid attention distribution statistic')
            metrics[f'{PREFIX}/distributions/{name}/{key}'] = value
        distribution_rows.append([name, values['mean_effective_candidates'],
                                  values['mean_max_weight'], values['mean_entropy_nats']])
    contrast_rows = []
    for name, values in hybrid['contrasts'].items():
        for key in FIELDS:
            value = values[key]
            upper = 1+1e-6 if key in ['candidate_argmax_changed_fraction', 'mean_total_variation',
                                    'value_aggregate_mean_cosine'] else math.inf
            lower = -1-1e-6 if key == 'value_aggregate_mean_cosine' else -1e-6
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError('invalid component-swap diagnostic statistic')
            metrics[f'{PREFIX}/contrasts/{name}/{key}'] = value
        contrast_rows.append([name, *[values[key] for key in FIELDS]])
    return {'metrics': metrics, 'tables': {
        PREFIX+'/attention_distributions': {
            'columns': ['saved_weight_configuration', 'effective_candidates', 'max_weight', 'entropy_nats'],
            'data': distribution_rows},
        PREFIX+'/component_swaps': {'columns': ['inference_swap', *FIELDS], 'data': contrast_rows}},
        'scope': SCOPE}


def collect_matching_drift(root, expected_rows, full_qa):
    root = Path(root)
    base_path = root/'matching-prior-drift-panel128-v1/summary.json'
    hybrid_path = root/'matching-prior-drift-panel128-hybrid-v1/summary.json'
    if not base_path.exists() and not hybrid_path.exists():
        return None
    base, hybrid = read(base_path), read(hybrid_path)
    result = validate_reports(base, hybrid, expected_rows, full_qa)
    receipt_path = root/'provenance/matching-prior-drift-panel128-and-hybrid-v1-file4.json'
    receipt = read(receipt_path)
    if receipt['status'] != 'PASS' or not receipt['local_equals_actual_file4_SHA256']:
        raise ValueError('actual file4 diagnostic closure required')
    files = [receipt_path]
    for ref in receipt['files']:
        path = root.parent/ref['relative_path']
        if not path.resolve().is_relative_to(root.parent.resolve()) or path.suffix == '.pt':
            raise ValueError('diagnostic artifact scope includes an invalid path or model binary')
        if sha(path) != ref['sha256'] or path.stat().st_size != ref['bytes']:
            raise ValueError('actual file4 diagnostic artifact changed')
        files.append(path)
    for path in [base_path, hybrid_path]:
        if path not in files:
            raise ValueError('diagnostic summary absent from actual file4 closure')
    qa_path = root/'attention-statistics-v1/task-full/summary.json'
    if base['original_saved_QA'] != {'path': str(qa_path), 'sha256': sha(qa_path), 'bytes': qa_path.stat().st_size}:
        raise ValueError('diagnostic saved-QA binding differs')
    if hybrid['full128_original_diagnostic_replay']['reference']['sha256'] != sha(base_path):
        raise ValueError('hybrid replay bound a different original report')
    proof_path = base_path.parent/'external-inputs-actual-file4.json'
    proof = read(proof_path)
    binding = read(hybrid_path.parent/'external-inputs-binding.json')
    if (proof['status'] != 'PASS_ACTUAL_FILE4_INPUT_HASHES' or len(proof['files']) != 387
        or proof['diagnostic_SHA256'] != sha(base_path)
        or binding['status'] != 'PASS_HYBRID_SAME_INPUTS_BOUND_TO_ACTUAL_FILE4'
        or binding['actual_canonical_file4_inputs'] != 387
        or not binding['all128_same_ordered_inputs_and_weights']
        or binding['base_diagnostic_SHA256'] != sha(base_path)
        or binding['hybrid_diagnostic_SHA256'] != sha(hybrid_path)
        or binding['actual_file4_input_receipt_SHA256'] != sha(proof_path)):
        raise ValueError('canonical input hashes or hybrid input binding differ')
    cfg = read(root/'configs/task-full.json')
    mirror = Path(cfg['validation']['dataset_args']['path_map']['/file4'])
    manifest_ref = full_qa['manifest'];manifest_path = Path(manifest_ref['path'])
    if sha(manifest_path) != manifest_ref['sha256'] or manifest_path.stat().st_size != manifest_ref['bytes']:
        raise ValueError('actual Full input manifest changed')
    rows = {r['sample_id']: r for r in map(json.loads, manifest_path.read_text().splitlines())}
    for case in base['inputs']:
        row = rows[case['sample_id']]
        paths = [mirror/Path(row['transport_cache']).relative_to('/file4'),
                 Path(cfg['validation']['dataset_args']['match_root'])/(case['sample_id']+'.npz'),
                 mirror/Path(row['target_anchor']).relative_to('/file4')]
        if ([ref['path'] for ref in case['files']] != [str(p) for p in paths]
            or case['conditioning_input_sha256'] != row['conditioning_input_sha256']):
            raise ValueError('diagnostic source features or edited RGB differ from manifest')
    inputs = [*base['weights'].values(), manifest_ref]
    inputs += [ref for case in base['inputs'] for ref in case['files']]
    expected = {}
    for ref in inputs:
        canonical = str(Path('/file4')/Path(ref['path']).relative_to(mirror))
        expected[canonical] = {'path': canonical, 'sha256': ref['sha256'], 'bytes': ref['bytes']}
    actual = {ref['path']: ref for ref in proof['files']}
    if len(actual) != 387 or actual != expected:
        raise ValueError('canonical file4 receipt covers different diagnostic inputs')
    archive = root/'provenance/matching-prior-drift-panel128-and-hybrid-v1-source'
    for ref in base['runtime_source']:
        path = archive/Path(ref['path']).name
        if path not in files or sha(path) != ref['sha256']:
            raise ValueError('diagnostic runtime source archive differs')
    result['files'] = files
    return result
