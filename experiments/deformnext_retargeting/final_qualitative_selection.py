"""Prespecified final qualitative cases, with no image generation or model choice.

Metadata-only examples are retained regardless of quality. Conditional cases
illustrate fixed outcome categories; their counts are not success-rate claims.
"""
from copy import deepcopy
import hashlib
import math


POLICY_SCHEMA = 'paper-qualitative-case-selection-v1'
PLAN_STATUS = 'PREDECLARED_ACTUAL_METADATA_ONLY_TEST_QUALITATIVE_EXAMPLES'
SELECTION_STATUS = 'SELECTED_PRESPECIFIED_FINAL_QUALITATIVE_EXAMPLES'
METRICS = ('CD', 'FScore', 'LPIPS', 'DreamSim')
CATEGORIES = ('joint-improvement', 'geometry-loss', 'appearance-loss')


def require(value, message):
    if not value:
        raise ValueError(message)


def policy():
    """The existing v355 declaration, without changing conditions or tie rules."""
    return dict(schema=POLICY_SCHEMA, metadata_only_examples=2,
        outcome_blind_namespaces=dict(source='paper-qualitative-source-v1', case='paper-qualitative-case-v1'),
        outcome_blind_algorithm='first two distinct source clusters by SHA256 ordering; first case by independent SHA256 ordering within each source',
        diagnostic_comparator='RETARGET unless selected_alias is RETARGET, then V1; fixed by model identity, never TEST scores',
        primary_diagnostic_metrics=dict(CD='lower', FScore='higher', LPIPS='lower-source9_10', DreamSim='lower-source9_10'),
        diagnostic_categories={
            'joint-improvement': dict(eligibility='delta_CD<0 and delta_FScore>0 and delta_LPIPS<0 and delta_DreamSim<0', order='delta_FScore ascending'),
            'geometry-loss': dict(eligibility='delta_CD>0 and delta_FScore<0', order='delta_FScore ascending'),
            'appearance-loss': dict(eligibility='delta_LPIPS>0 and delta_DreamSim>0', order='delta_LPIPS ascending')},
        diagnostic_choice='lower median index (n-1)//2 in eligible category, tie by case SHA256 then sample_id',
        same_asset_multiple_categories='merge labels; do not substitute another case to fill a panel',
        empty_category='report no eligible example under the stated condition; no fallback or changed metric',
        diagnostic_case_counts_not_population_success_rates=True, examples_not_evidence_of_average_effect=True,
        actual_frozen_models='all deduplicated zero-source4,zero-edited4,V1,RETARGET,selected aliases plus the identical GT',
        actual_views=[8, 9, 10],
        display='full-frame identical camera and RGB conversion; no model-specific crop, alignment or re-render',
        image_source='existing closed scored-image archives and exact prediction/reference image joins',
        input_views_are_not_novel_view_generalization=True, new_model_or_sampling_seed_selection=False)


def case_hash(sample_id):
    return hashlib.sha256(('paper-qualitative-case-v1:' + sample_id).encode()).hexdigest()


def outcome_blind(rows):
    """Use only the declared metadata fields; retain two distinct sources."""
    require(isinstance(rows, list) and len(rows) >= 2, 'nonempty metadata population with two sources required')
    seen, groups = set(), {}
    for row in rows:
        require(set(row) == {'sample_id', 'source_uid', 'source_cluster_id', 'split'}
            and all(isinstance(v, str) and bool(v) for v in row.values()) and row['split'] == 'test',
            'only the four prescribed nonempty TEST metadata fields are allowed')
        sid, cluster = row['sample_id'], row['source_cluster_id']
        require(sid not in seen, 'qualitative population must have distinct sample IDs')
        seen.add(sid); groups.setdefault(cluster, []).append(row)
    require(len(groups) >= 2, 'two distinct source clusters required')
    source_hash = lambda cluster: hashlib.sha256(('paper-qualitative-source-v1:' + cluster).encode()).hexdigest()
    chosen = []
    for cluster in sorted(groups, key=lambda c: (source_hash(c), c))[:2]:
        row = min(groups[cluster], key=lambda r: (case_hash(r['sample_id']), r['sample_id']))
        chosen.append(dict(row, source_rank_SHA256=source_hash(cluster), case_rank_SHA256=case_hash(row['sample_id'])))
    return chosen


def validate_declaration(declared, population):
    """Admit the original closed metadata-only declaration, not new choices."""
    require(declared.get('status') == PLAN_STATUS and declared.get('policy') == policy(),
        'the unchanged predeclared qualitative policy is required')
    rows = declared['all_metadata']; ids = population['sample_ids']; clusters = population['source_clusters_by_sample']
    require(len(rows) == len(ids) == len(set(ids)) == declared.get('assets') == 508
        and len(set(clusters.values())) == declared.get('source_clusters') == 134
        and set(clusters) == set(ids) and [r['sample_id'] for r in rows] == ids
        and all(r['source_cluster_id'] == clusters[r['sample_id']] for r in rows),
        'complete ordered TEST508/134-source metadata and frozen population required')
    require(all(declared['manifest'].get(k) == population['manifest'].get(k) for k in ['path', 'SHA256'])
        and declared.get('no_candidate_predictions_or_TEST_quality_values_read') is True
        and declared.get('no_candidate_NN_forward_or_training_or_new_render') is True
        and declared.get('actual_final_freeze_created') is False
        and declared.get('actual_final_qualitative_images_created') is False
        and declared.get('final_model_choice_not_made') is True,
        'retain the original manifest and quality-blind pre-freeze declaration')
    selected = outcome_blind(rows)
    require(declared.get('selected_outcome_blind_examples') == selected,
        'outcome-blind sample identities must match the original declaration')
    return selected


def select_cases(*, declared, population, selected_alias, models, statistics):
    """Select fixed cases and conditional lower medians from actual role scores.

    The closed-file and actual pre-TEST freeze checks belong to the exporter.
    This function never reads a filesystem, changes a model, or computes scores.
    """
    blind = validate_declaration(declared, population)
    require(selected_alias in models and {'zero-source4', 'zero-edited4', 'V1', 'RETARGET'} <= set(models)
        and len(models) in (4, 5), 'every deduplicated frozen model and both original controls required')
    comparator = 'V1' if selected_alias == 'RETARGET' else 'RETARGET'
    ids = population['sample_ids']; clusters = population['source_clusters_by_sample']
    per = statistics['per_asset']; lookup = {}
    for role in ['source9_10', 'all8_9_10']:
        require(set(per[role]) == set(models), 'all physical models must retain their actual camera-role scores')
        lookup[role] = {}
        for alias, rows in per[role].items():
            require(len(rows) == 508 and [r['sample_id'] for r in rows] == ids
                and all(r['source_cluster_id'] == clusters[r['sample_id']] for r in rows),
                'complete ordered per-asset source identities and role metrics required')
            lookup[role][alias] = {row['sample_id']: row for row in rows}
    scores, deltas = {}, {}
    for sid in ids:
        scores[sid] = {}
        for alias in models:
            geo, src = lookup['all8_9_10'][alias][sid], lookup['source9_10'][alias][sid]
            values = dict(CD=geo['CD'], FScore=geo['FScore'], LPIPS=src['LPIPS'], DreamSim=src['DreamSim'])
            require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values.values())
                and values['CD'] >= 0 and 0 <= values['FScore'] <= 1
                and values['LPIPS'] >= 0 and 0 <= values['DreamSim'] <= 2,
                'finite valid geometry and source-preservation metrics required')
            scores[sid][alias] = values
        deltas[sid] = {m: scores[sid][selected_alias][m] - scores[sid][comparator][m] for m in METRICS}
    conditions = {
        'joint-improvement': lambda d: d['CD'] < 0 and d['FScore'] > 0 and d['LPIPS'] < 0 and d['DreamSim'] < 0,
        'geometry-loss': lambda d: d['CD'] > 0 and d['FScore'] < 0,
        'appearance-loss': lambda d: d['LPIPS'] > 0 and d['DreamSim'] > 0}
    category_results, chosen = {}, {}
    def retain(sid, category):
        if sid not in chosen:
            chosen[sid] = dict(sample_id=sid, source_cluster_id=clusters[sid], categories=[],
                case_rank_SHA256=case_hash(sid), selected_minus_comparator=deepcopy(deltas[sid]),
                per_model_metrics=deepcopy(scores[sid]))
        chosen[sid]['categories'].append(category)
    for row in blind:
        retain(row['sample_id'], 'outcome-blind')
    for category in CATEGORIES:
        metric = 'LPIPS' if category == 'appearance-loss' else 'FScore'
        eligible = sorted((sid for sid in ids if conditions[category](deltas[sid])),
            key=lambda sid: (deltas[sid][metric], case_hash(sid), sid))
        selected = eligible[(len(eligible) - 1) // 2] if eligible else None
        category_results[category] = dict(eligible_cases=len(eligible), eligible_sample_ids_in_order=eligible,
            ordering_metric=metric, selected_sample_id=selected, empty_category_not_replaced=selected is None)
        if selected is not None:
            retain(selected, category)
    return dict(status=SELECTION_STATUS, assets=508, source_clusters=134,
        selected_alias=selected_alias, comparator=comparator, models=list(models), cameras=[8, 9, 10],
        outcome_blind_examples=deepcopy(blind), categories=category_results, cases=list(chosen.values()),
        all_per_asset_deltas=[dict(sample_id=sid, source_cluster_id=clusters[sid], **deltas[sid]) for sid in ids],
        conditional_examples_not_success_rates=True, no_model_or_noise_selection=True,
        empty_categories_not_filled=True, duplicate_examples_merged=True, full_goal_complete=False)
