import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.deformnext_retargeting.matching_v2_setup import freeze
from experiments.deformnext_retargeting.quick_metrics import panel_indices, panel_seeds
from experiments.deformnext_retargeting.diagnose_matching_v2 import OracleDataset


def test_panel_order_changes_do_not_change_sample_noise(tmp_path):
    dataset = SimpleNamespace(rows=[{'sample_id': 'b'}, {'sample_id': 'a'}])
    panel = tmp_path / 'panel.json'; panel.write_text(json.dumps({'sample_ids': ['a', 'b']}))
    seeds = tmp_path / 'seeds.json'; seeds.write_text(json.dumps({'a': 100, 'b': 200}))
    quick = {'panel_file': str(panel), 'seed_map_file': str(seeds)}
    assert panel_indices(dataset, quick) == [1, 0]
    assert panel_seeds(dataset, [1, 0], quick) == [100, 200]
    assert panel_seeds(dataset, [0, 1], quick) == [200, 100]
    panel.write_text(json.dumps({'sample_ids': ['a', 'a']}))
    with pytest.raises(ValueError, match='duplicate'):
        panel_indices(dataset, quick)


def test_head_calibration_is_source_cluster_disjoint_and_panels_nested(tmp_path):
    manifests = tmp_path / 'manifests'; manifests.mkdir()
    labels = tmp_path / 'labels'; labels.mkdir()
    def rows(prefix, count):
        return [{'sample_id': f'{prefix}{i}', 'source_uid': f'{prefix}uid{i // 2}',
                 'source_cluster_id': f'{prefix}cluster{i // 2}', 'category': str(i % 3)}
                for i in range(count)]
    train, val = rows('train', 40), rows('val', 40)
    for row in train: (labels / (row['sample_id'] + '.npz')).touch()
    for name, data in [('train', train), ('validation', val)]:
        (manifests / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in data))
    out = tmp_path / 'output'; freeze(manifests, labels, out)
    load = lambda name: [json.loads(l) for l in (out / (name + '.jsonl')).read_text().splitlines()]
    a, b = load('head_train'), load('head_calibration')
    assert len(a) + len(b) == len(train)
    assert not {r['source_cluster_id'] for r in a} & {r['source_cluster_id'] for r in b}
    small = json.loads((out / 'panel-32.json').read_text())['sample_ids']
    large = json.loads((out / 'panel-128.json').read_text())['sample_ids']
    assert large[:32] == small
    previous = (out / 'seed-map.json').read_text(); freeze(manifests, labels, out)
    assert (out / 'seed-map.json').read_text() == previous


def test_oracle_uses_only_visible_gt_and_rejects_unobserved_queries(tmp_path):
    row = {'sample_id': 'sample', 'split': 'validation'}
    positive = np.zeros((1024, 3), dtype=np.int32); positive[0] = [7, 1040, 2200]
    visible = np.zeros((1024, 3), dtype=bool); visible[0, 1] = True
    np.savez(tmp_path / 'sample.npz', sample_id='sample', split='validation',
             positive_indices=positive, positive_visible=visible)
    class Dataset:
        rows = [row]
        def __len__(self): return 1
        def __getitem__(self, index): return {'transport_source_valid': torch.zeros(3, 1024, dtype=torch.bool)}
    item = OracleDataset(Dataset(), tmp_path)[0]
    assert item['transport_target_valid'].sum() == 1
    assert item['transport_candidate_valid'].sum() == 1
    assert item['transport_semantic_indices'][0, 1] == 1040
    assert item['transport_source_valid'].flatten()[1040]
    assert torch.isfinite(item['transport_matching_logits']).sum() == 1


@pytest.mark.parametrize('k', [64, 128, 256])
def test_flexible_head_ignores_padding_and_learns_ranking_and_reliability(k):
    from experiments.deformnext_retargeting.matching_v2_head import MatchingV2Head, ranking_loss
    torch.manual_seed(8)
    head = MatchingV2Head()
    q, candidates = torch.randn(2, 1024), torch.randn(2, k, 1024)
    raw = torch.nn.functional.cosine_similarity(q[:, None], candidates, dim=-1)
    valid = torch.ones(2, k, dtype=torch.bool); valid[:, -4:] = False
    scores, confidence = head(q, candidates, raw, valid)
    changed = candidates.clone(); changed[:, -4:] *= 10000
    other, other_confidence = head(q, changed, raw, valid)
    torch.testing.assert_close(scores[valid], other[valid])
    torch.testing.assert_close(confidence, other_confidence)
    assert confidence.shape == (2, 3)
    positive = torch.zeros_like(valid); positive[:, 0] = True
    loss = ranking_loss(scores, positive, positive.any(-1)) + torch.nn.functional.binary_cross_entropy_with_logits(confidence, torch.ones_like(confidence))
    loss.backward()
    assert head.scorer.output.weight.grad.abs().sum() > 0
    assert head.confidence[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('arm', ['b2', 'b3', 'c1', 'c2'])
def test_v2_native_start_and_rejected_query_preserve_entire_native_condition(arm):
    from pixal3d.modules.matching_v2_conditioner import MatchingV2Conditioner
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    bridge = object.__new__(ImageConditionedProjMixin)
    module = MatchingV2Conditioner(arm=arm, reject=True, reject_threshold=.5,
        **({'deformation_dim': 2048} if arm == 'c2' else {}))
    bridge.deform_conditioner=module; bridge.training_models={}
    pack = dict(source_values=torch.randn(1, 3, 24, 1024), source_valid=torch.ones(1, 3, 24, dtype=torch.bool),
        evidence_top1=torch.randn(1, 3, 1024), evidence_soft=torch.randn(1, 3, 1024),
        p_use=torch.tensor([[.9, .1, .8]]), cached_valid=torch.ones(1, 3, dtype=torch.bool),
        target_valid=torch.tensor([[True, True, False]]),
        patch_grid=torch.tensor([[[[0, 1], [2, 0]], [[1, 2], [0, 1]]]]))
    if arm in {'c1','c2'}: pack['target_query']=torch.randn(1,3,1024)
    if arm == 'c2': pack['deformation']=torch.randn(1,3,2048)
    native, anchor=torch.randn(1,8,1024),torch.randn(1,8,1024)
    forward=lambda: bridge._inject_deform_transport({'proj':native.clone()},pack,None,anchor_condition={'proj':anchor})['proj']
    torch.testing.assert_close(forward(),native,rtol=0,atol=0)
    forward().square().sum().backward(); assert module.fusion_gate.grad.abs()>0
    if module.retargeter is not None:
        assert module.retargeter.net[-1].weight.grad.abs().sum()>0
        with torch.no_grad(): module.retargeter.net[-1].weight.normal_(std=.01)
    module.fusion_gate.data.fill_(.5)
    result=forward(); rejected=pack['patch_grid'].flatten(1)!=0
    torch.testing.assert_close(result[rejected],native[rejected],rtol=0,atol=0)
    assert (result[~rejected]-native[~rejected]).abs().sum()>0


def test_batched_dense_evaluation_keeps_per_asset_noise_and_counts(monkeypatch):
    from experiments.deformnext_retargeting.quick_metrics import _ss_metrics
    monkeypatch.setattr(torch.cuda,'manual_seed',lambda seed:None)
    class Dataset:
        rows=[{'sample_id':f'asset{i}','source_cluster_id':f'cluster{i}'} for i in range(3)]
        def __getitem__(self,index):return {'x_0':torch.ones(1,2,2,2),'cond':torch.zeros(4,3,2,2)}
        def collate_fn(self,items):return {k:torch.stack([r[k] for r in items]) for k in items[0]}
        def decode_latent(self,latents,batch_size):return latents
    sampler=SimpleNamespace(sample=lambda model,noise,**kw:SimpleNamespace(samples=noise))
    trainer=SimpleNamespace(device='cpu',models={'denoiser':None},get_sampler=lambda:sampler,get_inference_cond=lambda **kw:{})
    expected,_=_ss_metrics(trainer,Dataset(),[0,1,2],{'sampling_steps':12,'generation_batch_size':1})
    per_asset=trainer._quick_per_asset
    actual,_=_ss_metrics(trainer,Dataset(),[0,1,2],{'sampling_steps':12,'generation_batch_size':2})
    assert actual==expected
    assert trainer._quick_per_asset==per_asset


def test_queue_never_shares_ss_training_and_restricts_physical_devices():
    from experiments.deformnext_retargeting.matching_v2_queue import can_run,gpu_ids,choose_gpu
    assert gpu_ids('0,1,2,3')==['0','1','2','3']
    for invalid in ['0,0','-1','bad']:
        with pytest.raises(ValueError):gpu_ids(invalid)
    with pytest.raises(ValueError,match='not authorized'):
        choose_gpu({'gpu':'7'},gpu_ids('0,1,2,3'),{})
    assert choose_gpu({'resource':'cpu','gpu':'7'},gpu_ids('0,1,2,3'),{})==''
    state={'head':{'status':'running','gpu':'0','resource':'light'}}
    assert can_run('0','evaluation',state)
    assert not can_run('0','light',state)
    assert not can_run('0','exclusive',state)
    state['ss']={'status':'running','gpu':'1','resource':'exclusive'}
    assert not can_run('1','evaluation',state)
    assert not can_run('1','light',state)
