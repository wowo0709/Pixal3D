"""Evaluator-only probability/transfer statistics for saved ray observations.

Relations are deduplicated (source-camera ray, target patch) pairs. Visibility
defines the evaluation population; it never changes conditioning or weights.
"""
import numpy as np


def utilization(*, ray_ids, patch_ids, candidates, candidate_valid, prior,
                weights, source_valid, positive_indices, positive_visible,
                target_foreground, use_probability, correction_rms,
                relative_correction_rms):
    """Measure exact-GT mass, especially prior-Top1-miss/support-hit relations."""
    arrays = [ray_ids, patch_ids, candidates, candidate_valid, prior, weights,
              source_valid, positive_indices, positive_visible, target_foreground,
              use_probability, correction_rms, relative_correction_rms]
    if any(not isinstance(x, np.ndarray) for x in arrays):
        raise ValueError('saved arrays, with no model forward, required')
    if (ray_ids.ndim != 2 or ray_ids.shape[0] != 3
            or patch_ids.shape != ray_ids.shape[1:]
            or candidates.shape != (3072, 64)
            or any(x.shape != candidates.shape for x in [candidate_valid, prior, weights])
            or source_valid.shape != (3072,)
            or positive_indices.shape != (1024, 3)
            or positive_visible.shape != positive_indices.shape
            or target_foreground.shape != (1024,)
            or any(x.shape != (3072,) for x in
                   [use_probability, correction_rms, relative_correction_rms])):
        raise ValueError('native three-camera ray/patch/support shapes required')
    for x, low, high in [(ray_ids, -1, 1023), (patch_ids, 0, 1023),
                         (candidates, 0, 3071), (positive_indices, 0, 3071)]:
        if not np.issubdtype(x.dtype, np.integer) or np.any((x < low) | (x > high)):
            raise ValueError('integer camera/patch addresses in native range required')
    for x in [candidate_valid, source_valid, positive_visible, target_foreground]:
        if x.dtype != np.bool_:
            raise ValueError('explicit boolean support/observation masks required')
    numeric = [prior, weights, use_probability, correction_rms, relative_correction_rms]
    if any(not np.isfinite(x).all() or (x < 0).any() for x in numeric):
        raise ValueError('finite nonnegative actual probabilities and transfer norms required')
    if np.any(use_probability > 1):
        raise ValueError('source-use probability outside [0,1]')
    active = candidate_valid.any(-1)
    if np.any(candidate_valid & ~source_valid[candidates]):
        raise ValueError('candidate support includes invalid source patches')
    for p in [prior, weights]:
        if (np.any(p[~candidate_valid] != 0)
                or not np.allclose(p[active].sum(-1), 1., atol=1e-5, rtol=0)):
            raise ValueError('normalized distributions on actual valid candidates required')
    encoded = np.unique(np.concatenate([
        (ray_ids[v, ray_ids[v] >= 0] + v * 1024) * 1024
        + patch_ids[ray_ids[v] >= 0] for v in range(3)]))
    rays, patches = encoded // 1024, encoded % 1024
    visible = positive_visible & source_valid[positive_indices] & target_foreground[:, None]
    observed = visible[patches].any(-1)
    rays, patches = rays[observed], patches[observed]
    if np.any(~active[rays]):
        raise ValueError('observable relation lacks native64 candidate support; do not filter silently')
    uniform = candidate_valid / candidate_valid.sum(-1, keepdims=True).clip(1)
    distributions = {'matching_prior': prior, 'actual_attention': weights, 'uniform_K': uniform}
    subsets = {key: dict(relations=0, exact_mass_sum={name: 0. for name in distributions},
                         use_probability_sum=0., correction_rms_sum=0.,
                         relative_correction_rms_sum=0.) for key in
               ['all_observable', 'prior_top1_miss_support_hit', 'prior_top1_hit', 'support_miss']}
    for start in range(0, len(rays), 512):
        r, p = rays[start:start + 512], patches[start:start + 512]
        hit = ((candidates[r, :, None] == positive_indices[p, None, :])
               & visible[p, None, :]).any(-1) & candidate_valid[r]
        support_hit = hit.any(-1)
        top1_hit = hit[np.arange(len(r)), prior[r].argmax(-1)]
        masks = dict(all_observable=np.ones(len(r), dtype=bool),
                     prior_top1_miss_support_hit=~top1_hit & support_hit,
                     prior_top1_hit=top1_hit, support_miss=~support_hit)
        for name, mask in masks.items():
            row = subsets[name]
            row['relations'] += int(mask.sum())
            for key, distribution in distributions.items():
                row['exact_mass_sum'][key] += float((distribution[r[mask]] * hit[mask]).sum(dtype=np.float64))
            for key, values in [('use_probability', use_probability), ('correction_rms', correction_rms),
                                ('relative_correction_rms', relative_correction_rms)]:
                row[key + '_sum'] += float(values[r[mask]].sum(dtype=np.float64))
    return dict(distinct_relations=len(encoded), observable_relations=len(rays),
                unobservable_relations_excluded=int((~observed).sum()), subsets=subsets,
                GT_visibility_used_only_for_evaluation=True,
                uniform_K_is_diagnostic_not_matched_training=True)


def summarize_counts(records):
    """Pool relation counts; retain per-asset results for source-cluster inference."""
    if not records or len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('unique nonempty actual asset records required')
    subsets = {}
    for name in records[0]['counts']['subsets']:
        rows = [r['counts']['subsets'][name] for r in records]
        count = sum(r['relations'] for r in rows)
        subsets[name] = dict(relations=count,
            exact_mass_mean={key: sum(r['exact_mass_sum'][key] for r in rows) / count if count else None
                             for key in rows[0]['exact_mass_sum']},
            **{key + '_mean': sum(r[key + '_sum'] for r in rows) / count if count else None
               for key in ['use_probability', 'correction_rms', 'relative_correction_rms']})
    return dict(assets=len(records), source_clusters=len({r['source_cluster_id'] for r in records}),
                observable_relations=sum(r['counts']['observable_relations'] for r in records),
                subsets=subsets)


def mass_intervals(records, *, draws=4000, seed=20261024):
    """Exploratory cluster bootstrap of mass on prior-Top1-miss/support-hit pairs."""
    summarize_counts(records)
    if type(draws) is not int or draws < 1:
        raise ValueError('positive bootstrap count required')
    clusters = sorted({r['source_cluster_id'] for r in records})
    groups = [[r['counts']['subsets']['prior_top1_miss_support_hit'] for r in records
               if r['source_cluster_id'] == cluster] for cluster in clusters]
    den = np.array([sum(r['relations'] for r in group) for group in groups])
    resamples = np.random.default_rng(seed).integers(0,len(groups),size=(draws,len(groups)))
    bootden = den[resamples].sum(1)
    nonempty = bootden > 0
    contrasts = {}
    for reference in ['matching_prior', 'uniform_K']:
        delta = np.array([sum(r['exact_mass_sum']['actual_attention']
                              - r['exact_mass_sum'][reference] for r in group) for group in groups])
        values = delta[resamples].sum(1)[nonempty] / bootden[nonempty]
        contrasts['actual_attention_minus_' + reference] = dict(
            relation_weighted_delta=float(delta.sum()/den.sum()) if den.sum() else None,
            source_cluster_CI95=np.quantile(values,[.025,.975]).tolist() if len(values) else None)
    return dict(contrasts=contrasts,source_clusters=len(groups),bootstrap_draws=draws,
                nonempty_bootstrap_draws=int(nonempty.sum()),
                excluded_empty_subset_draws=int((~nonempty).sum()),bootstrap_seed=seed,
                exploratory_not_multiple_comparison_corrected=True)


def validate_native_population(records, historical, address_evidence):
    """Require actual address parity instead of equating CPU and native counts."""
    arms = ['E_FIXED_RAW', 'E_LEARNED_RAW']
    if (list(records) != arms or address_evidence.get('status') !=
            'PASS_ACTUAL_OLD_CPU_AND_SAVED_NATIVE_CAMERA_ADDRESS_DIAGNOSIS'
            or address_evidence.get('assets') != 32
            or address_evidence.get('old_CPU_diagnostic_all32_relation_counts_exact') is not True
            or address_evidence.get('original_native_code_unmodified') is not True
            or address_evidence.get('NN_forward_or_training') != 0
            or 'GPU_BF16' not in address_evidence.get('all_assets_exact_native_address_modes', [])):
        raise ValueError('actual no-NN native-BF16/historical-FP32 address parity proof required')
    ids = [r['sample_id'] for r in historical['records']]
    if (len(ids) != 32 or len(set(ids)) != 32
            or [r['sample_id'] for r in address_evidence['records']] != ids
            or any([r['sample_id'] for r in records[a]] != ids for a in arms)):
        raise ValueError('the same ordered32 actual assets required')
    old_total = native_total = 0
    for i, (old, proof) in enumerate(zip(historical['records'], address_evidence['records'])):
        fixed, learned = (records[a][i] for a in arms)
        old_count = old['modes']['member-point-D1-mixture64']['counts']['observable_ray_target_patch_pairs']
        current = fixed['counts']['observable_relations']
        if (learned['counts']['observable_relations'] != current
                or fixed['source_cluster_id'] != learned['source_cluster_id'] or fixed['source_cluster_id'] != old['source_cluster_id']
                or fixed['GT_fingerprint'] != learned['GT_fingerprint'] or fixed['GT_fingerprint'] != old['GT_fingerprint']
                or fixed['input_tensor_SHA256'] != learned['input_tensor_SHA256']
                or proof['source_array_SHA256'] != fixed['SHA256']
                or proof['observable_relations']['CPU_FP32'] != old_count
                or proof['observable_relations']['GPU_BF16'] != current
                or proof['observable_relations']['actual_saved_native'] != current
                or not proof['ray_address_equality_to_saved_native']['GPU_BF16']):
            raise ValueError('per-asset actual native address/GT/model population differs')
        old_total += old_count; native_total += current
    return dict(historical_CPU_FP32_relations=old_total,actual_native_BF16_relations=native_total,
                native_pair_relation_population_equal=True,
                historical_and_native_ray_addresses_identical=address_evidence['CPU_to_native_ray_address_differences']==0,
                historical_to_native_relation_symmetric_difference=address_evidence['CPU_to_native_relation_symmetric_difference'])
