import pytest
import torch

from experiments.deformnext_retargeting.summarize_current_geometry_endpoint import cluster_contrast, audit_history, SEEDS
from experiments.deformnext_retargeting.train_current_geometry_endpoint import training_order, condition


def test_cluster_intervals_count_sources_after_averaging_all_model_noise_repeats():
    ids = ['empty', 'source1', 'source2']
    left = {seed: {(sid, noise): {'metric': None if sid == 'empty' else .4}
            for sid in ids for noise in [0, 1, 2]} for seed in SEEDS}
    right = {seed: {(sid, noise): {'metric': None if sid == 'empty' else .41}
             for sid in ids for noise in [0, 1, 2]} for seed in SEEDS}
    result = cluster_contrast(left, right, 'metric', ids)
    assert result['source_clusters'] == 2
    assert result['mean'] == pytest.approx(.01)
    assert result['ci95'] == pytest.approx([.01, .01])
    assert list(result['seed_means']) == ['101', '202', '303']
    right[101]['source1', 2]['metric'] = None
    with pytest.raises(ValueError, match='availability'):
        cluster_contrast(left, right, 'metric', ids)


def test_full_audit_detects_one_mismatched_update_noise():
    rows = [{'sample_id': f'p{i}'} for i in range(32)]
    history = []
    for update in range(128):
        row, epoch, noise = training_order(rows, 101, update)
        history.append(dict(update=update+1, epoch=epoch, sample_id=row['sample_id'], noise_seed=noise,
                            loss=.1, endpoint_loss=.1, gradient_norm=1.))
    audit_history(history, [r['sample_id'] for r in rows], 101)
    history[97]['noise_seed'] += 1
    with pytest.raises(ValueError, match='fixed pair/noise'):
        audit_history(history, [r['sample_id'] for r in rows], 101)


def test_source_off_and_query_shuffle_keep_global_negative_and_information_contract():
    class Recorder:
        def __call__(self, target, source, geometry):
            self.source, self.geometry = source, geometry
            return source, {}
    adapter = Recorder()
    native = {'cond': {'global': torch.randn(1, 2, 4), 'proj': torch.ones(1, 4096, 4)},
              'neg_cond': {'global': torch.zeros(1, 2, 4), 'proj': torch.zeros(1, 4096, 4)}}
    pair = dict(sample_id='fixed', native=native, target_proj=torch.zeros(1, 4096, 4),
                geometry=torch.arange(4096*9).reshape(1, 4096, 9).float())
    off, _ = condition(pair, adapter, source_off=True)
    assert adapter.source is pair['target_proj']
    assert off['cond']['global'] is native['cond']['global']
    assert off['neg_cond'] is native['neg_cond']
    assert native['cond']['proj'].eq(1).all()
    condition(pair, adapter, shuffled=True)
    assert torch.equal(adapter.geometry[0].sort(0).values, pair['geometry'][0].sort(0).values)
    assert not torch.equal(adapter.geometry, pair['geometry'])
    with pytest.raises(ValueError, match='physical query channels'):
        condition({**pair, 'geometry': None}, adapter, shuffled=True)
