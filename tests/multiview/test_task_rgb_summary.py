import pytest

from experiments.deformnext_retargeting.summarize_task_rgb_transfer import audit_sampling, paired_ci


def test_pilot_ci_resamples_source_cluster_means_once_after_seed_aggregation():
    result = paired_ci([.005] * 61)
    assert result['independent_source_clusters'] == 61
    assert result['mean'] == pytest.approx(.005)
    assert result['ci95'] == pytest.approx([.005, .005])
    with pytest.raises(ValueError, match='pair-cluster'):
        paired_ci([[.005] * 61] * 3)


def test_matched_sampling_audit_rejects_a_single_different_query_or_initial_loss():
    rows = [dict(update=i+1, sample_id='pair', sampled_query_sha256=str(i), loss=.2) for i in range(1200)]
    audit_sampling(rows, [dict(r) for r in rows])
    changed = [dict(r) for r in rows]
    changed[937]['sampled_query_sha256'] = 'different'
    with pytest.raises(ValueError, match='sequence differs'):
        audit_sampling(rows, changed)
    changed = [dict(r) for r in rows]
    changed[0]['loss'] = .19
    with pytest.raises(ValueError, match='initial matched'):
        audit_sampling(rows, changed)
