import pytest
import torch

from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from experiments.deformnext_retargeting.matched_top1_retargeter import configuration
from tests.multiview.test_task_matching_attention import inputs


def model(aggregation):
    torch.manual_seed(19)
    return DeformationAwareConditioner(arm='c1',key_dim=1024,value_dim=1024,query_dim=1024,
        score_temperature=.05,candidate_aggregation=aggregation)


def test_top1_retargeter_selects_matching_argmax_not_uniform_and_masks_empty_queries():
    m=model('top1');p=inputs();out=m(**p)
    chosen=p['matching_logits'].argmax(-1)
    flat=p['source_values'].flatten(1,2)
    expected=torch.stack([flat[0,chosen[0,q]] for q in range(3)])[None];expected[:,2]=0
    torch.testing.assert_close(out.evidence,expected,rtol=0,atol=0)
    assert out.candidate_weights[0,:2].sum()==2 and out.candidate_weights[0,:2].count_nonzero()==2
    assert out.candidate_weights[0,2].count_nonzero()==0
    out.evidence.square().mean().backward()
    assert m.retargeter.net[-1].weight.grad.abs().sum()>0
    assert all(not x.requires_grad for x in m.q2c.parameters())


def test_top1_retargeter_initial_weights_and_caller_rng_equal_weighted_C1():
    a=model('weighted');rng_a=torch.random.get_rng_state();b=model('top1');rng_b=torch.random.get_rng_state()
    assert torch.equal(rng_a,rng_b)
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(x,b.state_dict()[k]) for k,x in a.state_dict().items())
    p=inputs();assert (a(**p).evidence-b(**p).evidence).abs().max()>0


def test_top1_retargeter_temperature_cannot_change_online_or_cached_correspondence():
    m=model('top1');p=inputs()
    with torch.no_grad():
        m.fusion_gate.fill_(.1)
        m.retargeter.net[-1].weight.normal_(std=.002)
    for online in [False, True]:
        pack=dict(p)
        if online:
            pack.pop('matching_logits',None)
            pack.pop('semantic_indices',None)
        m.score_temperature=.05
        left=m(**pack)
        m.score_temperature=.005
        right=m(**pack)
        for key in ['evidence','candidate_weights','retarget_residual','candidate_indices']:
            torch.testing.assert_close(getattr(left,key),getattr(right,key),rtol=0,atol=0)


@pytest.mark.parametrize('extra',[{'arm':'b3'},{'task_attention':{'decoder':True,'score_prior':True,'chunk_size':64}}])
def test_top1_mode_cannot_silently_replace_existing_or_task_models(extra):
    args=dict(arm='c1',key_dim=1024,value_dim=1024,query_dim=1024,candidate_aggregation='top1');args.update(extra)
    with pytest.raises(ValueError,match='static R9 C1'):DeformationAwareConditioner(**args)


def test_matched_recipe_preserves_every_training_data_and_retargeter_setting(tmp_path):
    source={'trainer':{'args':{'deform_conditioner':{'arm':'c1','score_temperature':.05,'hidden_dim':256},
        'max_steps':480,'batch_size_per_gpu':48,'batch_split':1}},'training_seed':101,'dataset':{'proof':'fixed'},'experiment':{'arm':'static-c1'}}
    c=configuration(source,tmp_path)
    assert c['training_seed']==source['training_seed'] and c['dataset']==source['dataset']
    a=c['trainer']['args']['deform_conditioner'];b=source['trainer']['args']['deform_conditioner']
    assert {k:v for k,v in a.items() if k not in {'candidate_aggregation','freeze_q2c'}}==b
    assert a['candidate_aggregation']=='top1' and a['freeze_q2c']
    assert c['trainer']['args']['max_steps']==480 and 'candidate_aggregation' not in b
