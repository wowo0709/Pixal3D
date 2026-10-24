import numpy as np
import pytest

from experiments.deformnext_retargeting.ray_attention_utilization import utilization, summarize_counts, mass_intervals, validate_native_population


def example():
    # Two unique target patches on ray0; a repeated voxel must not get another vote.
    rays = np.array([[0, 0, 0], [-1, -1, -1], [-1, -1, -1]])
    candidates = np.tile(np.arange(64), (3072, 1))
    valid = np.zeros_like(candidates, dtype=bool); valid[0] = True
    prior = np.zeros((3072, 64), dtype=np.float32); prior[0, 0] = .75; prior[0, 1] = .25
    attention = prior.copy(); attention[0, 0] = .4; attention[0, 1] = .6
    truth = np.zeros((1024, 3), dtype=np.int64); truth[0, 0] = 1; truth[1, 0] = 0
    seen = np.zeros((1024, 3), dtype=bool); seen[:2, 0] = True
    return dict(ray_ids=rays, patch_ids=np.array([0, 0, 1]), candidates=candidates,
                candidate_valid=valid, prior=prior, weights=attention,
                source_valid=np.ones(3072, dtype=bool), positive_indices=truth,
                positive_visible=seen, target_foreground=np.ones(1024, dtype=bool),
                use_probability=np.full(3072, .5), correction_rms=np.full(3072, 2.),
                relative_correction_rms=np.full(3072, .1))


def test_recoverable_subset_mass_deduplicates_and_does_not_reweight_model():
    data = example(); before = data['weights'].copy(); out = utilization(**data)
    assert out['distinct_relations'] == out['observable_relations'] == 2
    recoverable = out['subsets']['prior_top1_miss_support_hit']
    assert recoverable['relations'] == 1
    assert recoverable['exact_mass_sum']['matching_prior'] == .25
    assert recoverable['exact_mass_sum']['actual_attention'] == pytest.approx(.6)
    assert recoverable['exact_mass_sum']['uniform_K'] == 1 / 64
    assert recoverable['correction_rms_sum'] == 2.
    np.testing.assert_array_equal(data['weights'], before)


def test_invisible_truth_is_excluded_not_used_to_change_weights():
    data = example(); data['positive_visible'][0] = False; out = utilization(**data)
    assert out['observable_relations'] == 1
    assert out['unobservable_relations_excluded'] == 1
    assert out['subsets']['prior_top1_miss_support_hit']['relations'] == 0


def test_absent_GT_support_is_separate_from_top1_ambiguity():
    data = example(); data['positive_indices'][0, 0] = 100; out = utilization(**data)
    assert out['subsets']['support_miss']['relations'] == 1
    assert out['subsets']['prior_top1_miss_support_hit']['relations'] == 0


@pytest.mark.parametrize('damage', ['weight', 'support', 'use', 'missing_candidates'])
def test_corrupt_or_silently_filtered_observations_are_rejected(damage):
    data = example()
    if damage == 'weight': data['weights'][0] *= .5
    elif damage == 'support': data['source_valid'][1] = False
    elif damage == 'use': data['use_probability'][0] = 1.01
    else: data['candidate_valid'][0] = False; data['prior'][0] = 0; data['weights'][0] = 0
    with pytest.raises(ValueError): utilization(**data)


def test_empty_subset_remains_unmeasured_and_duplicate_assets_are_rejected():
    data = example(); data['positive_visible'][:] = False
    record = dict(sample_id='one', source_cluster_id='source', counts=utilization(**data))
    result = summarize_counts([record])
    assert result['subsets']['prior_top1_miss_support_hit']['exact_mass_mean']['actual_attention'] is None
    with pytest.raises(ValueError): summarize_counts([record, record])


def test_mass_intervals_cluster_all_deformations_and_expose_empty_resamples():
    counts = utilization(**example())
    rows = [dict(sample_id='a1',source_cluster_id='A',counts=counts),
            dict(sample_id='a2',source_cluster_id='A',counts=counts)]
    out = mass_intervals(rows,draws=100)
    assert out['source_clusters'] == 1
    assert out['nonempty_bootstrap_draws'] == 100
    contrast = out['contrasts']['actual_attention_minus_matching_prior']
    assert contrast['relation_weighted_delta'] == pytest.approx(.35)
    assert contrast['source_cluster_CI95'] == pytest.approx([.35,.35])
    data = example(); data['positive_visible'][:] = False
    rows.append(dict(sample_id='b',source_cluster_id='B',counts=utilization(**data)))
    out = mass_intervals(rows,draws=100)
    assert out['source_clusters'] == 2
    assert 0 < out['excluded_empty_subset_draws'] < 100


def native_population():
    records = {a: [] for a in ['E_FIXED_RAW','E_LEARNED_RAW']}; old = []; proofs = []
    for i in range(32):
        row = dict(sample_id=str(i),source_cluster_id=str(i),GT_fingerprint='label',
                   input_tensor_SHA256={'rays':'same'},SHA256='actual_observation',counts={'observable_relations':3})
        for arm in records: records[arm].append(dict(row))
        old.append(dict(sample_id=str(i),source_cluster_id=str(i),GT_fingerprint='label',
                        modes={'member-point-D1-mixture64':{'counts':{'observable_ray_target_patch_pairs':2}}}))
        proofs.append(dict(sample_id=str(i),source_array_SHA256='actual_observation',
                           observable_relations={'CPU_FP32':2,'GPU_BF16':3,'actual_saved_native':3},
                           ray_address_equality_to_saved_native={'GPU_BF16':True}))
    evidence = dict(status='PASS_ACTUAL_OLD_CPU_AND_SAVED_NATIVE_CAMERA_ADDRESS_DIAGNOSIS',assets=32,
                    old_CPU_diagnostic_all32_relation_counts_exact=True,original_native_code_unmodified=True,
                    NN_forward_or_training=0,all_assets_exact_native_address_modes=['GPU_BF16'],records=proofs,
                    CPU_to_native_ray_address_differences=32,CPU_to_native_relation_symmetric_difference=32)
    return records, {'records':old}, evidence


def test_native_population_change_requires_explicit_actual_address_evidence():
    records, old, proof = native_population(); result = validate_native_population(records,old,proof)
    assert result['historical_CPU_FP32_relations'] == 64
    assert result['actual_native_BF16_relations'] == 96
    assert result['historical_and_native_ray_addresses_identical'] is False
    with pytest.raises(ValueError): validate_native_population(records,old,{})


@pytest.mark.parametrize('damage',['pair','proof','GT','address','duplicate'])
def test_native_pair_and_provenance_cannot_be_relaxed_to_match_only_total(damage):
    records, old, proof = native_population()
    if damage=='pair': records['E_LEARNED_RAW'][0]['counts']={'observable_relations':2}
    elif damage=='proof': proof['records'][0]['source_array_SHA256']='wrong'
    elif damage=='GT': records['E_LEARNED_RAW'][0]['GT_fingerprint']='wrong'
    elif damage=='address': proof['records'][0]['ray_address_equality_to_saved_native']['GPU_BF16']=False
    else:
        for arm in records: records[arm][1]=dict(records[arm][0])
    with pytest.raises(ValueError): validate_native_population(records,old,proof)
