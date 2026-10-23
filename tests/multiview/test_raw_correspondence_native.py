"""Structure/configuration tests; no CPU neural forwards."""
import copy

import pytest
import torch

from experiments.deformnext_retargeting.raw_correspondence_native_recipe import NAME,SCHEMA,VARIANTS,MATCHING,validate_config
from pixal3d.modules.raw_correspondence_top1_source import RawCorrespondenceTop1SourceConditioner
from test_r_mediated_native_recipe import recipe


def control(arm='C_RAW',phase='science'):
    reference=recipe('R_raw_topk');cfg=copy.deepcopy(reference)
    cfg['trainer']['name']=NAME;cfg['trainer']['args'].pop('ray_feature_utility')
    cfg['use_wandb']=phase=='science'
    cfg['experiment'].update(native_variant=arm,conditioning_schema=SCHEMA,
        module_pipeline_schema='unified-module-pipeline-v2',readout_selection='top1',
        correspondence_mode=VARIANTS[arm],actual_source_tensors=10,learned_projection_tensors=0,
        matching_decoder_physically_absent=MATCHING[arm]=='RAW-cosine',matching_scalar_score_physically_absent=True,matching_mode=MATCHING[arm],physical_deformation_query_channels=0,
        appearance_mode='single-correspondence-Top1-feature',residual_logit_bound=None,
        engineering_only=phase!='science')
    return cfg,reference


@pytest.mark.parametrize('arm',VARIANTS)
@pytest.mark.parametrize('phase',['proof','capacity','science'])
def test_raw_and_spatial_share_native_model_training_data_and_r_contract(arm,phase):
    cfg,reference=control(arm,phase)
    assert validate_config(cfg,phase,reference)==arm


@pytest.mark.parametrize('change',['correspondence','decoder','head','batch','source-lr','teacher'])
def test_comparison_cannot_silently_change_another_module_or_training_setting(change):
    cfg,reference=control();e=cfg['experiment'];t=cfg['trainer']['args']
    if change=='correspondence':e['correspondence_mode']='native-spatial'
    elif change=='decoder':e['matching_decoder_physically_absent']=False
    elif change=='head':e['learned_projection_tensors']=9
    elif change=='batch':t['batch_size_per_gpu']=48
    elif change=='source-lr':t['model_learning_rates']['deform_conditioner']=.001
    else:e['source_teacher_updates']=112
    with pytest.raises(ValueError):validate_config(cfg,'science',reference)


def test_decoder_and_attention_are_physically_absent_and_mode_checkpoints_are_distinct():
    kwargs=dict(arm='c1',implementation='r9-native-v5',key_dim=1024,value_dim=1024,
        query_dim=1024,hidden_dim=256,topk=64,score_temperature=.05,
        projection_mode='native-bilinear',ray_mode='semantic-top1',ray_chunk_size=64,
        candidate_aggregation='top1')
    modules=[]
    for mode in ['semantic','native-spatial']:
        with torch.random.fork_rng():
            torch.manual_seed(101)
            module=RawCorrespondenceTop1SourceConditioner(correspondence=mode,**kwargs)
        assert module.q2c is None and not hasattr(module,'ray_readout')
        assert sum(p.requires_grad for p in module.parameters())==10
        assert not any(n.startswith('q2c.') for n in module.state_dict())
        module.train();module.eval();modules.append(module)
    left={n:p for n,p in modules[0].named_parameters() if p.requires_grad}
    right={n:p for n,p in modules[1].named_parameters() if p.requires_grad}
    assert left.keys()==right.keys() and all(torch.equal(p,right[n]) for n,p in left.items())
    with pytest.raises(RuntimeError):modules[0].load_state_dict(modules[1].state_dict(),strict=True)


def pure_feature_fixture():
    """Onehot feature algebra, without constructing or forwarding a NN."""
    target=torch.zeros(1,1024,1024);target[0,0,0]=1.;target[0,1,1]=1.
    source=torch.zeros(1,3,1024,1024)
    source[0,0,:64,:64]=torch.eye(64)
    valid=torch.zeros(1,3,1024,dtype=torch.bool);valid[0,0,:64]=True
    rays=torch.full((1,3,4096),-1,dtype=torch.long);rays[0,0,:3]=0
    patches=torch.zeros(1,4096,dtype=torch.long);patches[0,:3]=torch.tensor([0,1,1])
    observed=torch.zeros(1,4096,dtype=torch.bool);observed[0,:3]=True
    return target,source,valid,rays,patches,observed


def test_raw_feature_algebra_keeps_distinct_member_votes_and_original_spatial_candidates():
    from pixal3d.modules.raw_member_ray_posterior import raw_member_ray_posteriors
    from pixal3d.modules.raw_spatial_member_posterior import raw_spatial_member_posteriors
    target,source,valid,rays,patches,observed=pure_feature_fixture()
    kwargs=dict(target_keys=target,source_valid=valid,ray_ids=rays,patch_ids=patches,
                observed_query=observed,temperature=.05,chunk_size=64)
    semantic=raw_member_ray_posteriors(source_keys=source,**kwargs)
    spatial_keys=torch.zeros(1,3072,3,1024)
    spatial_keys[0,0,:,:3]=torch.eye(3)
    spatial=raw_spatial_member_posteriors(spatial_keys=spatial_keys,**kwargs)
    for posterior in [semantic,spatial]:
        assert posterior.distinct_member_count[0,0]==2  # Repeated patch1 votes once.
        torch.testing.assert_close(posterior.contextual_query[0,0,:2],torch.tensor([.5,.5]))
        assert posterior.query_valid.sum()==1 and not posterior.query_valid[0,1:].any()
        torch.testing.assert_close(posterior.probabilities[0,0].sum(),torch.tensor(1.))
    selected=set(semantic.candidate_indices[0,0,:2].tolist())
    assert selected=={0,1}
    assert spatial.candidate_indices.shape[-1]==3
    torch.testing.assert_close(spatial.probabilities[0,0,:2],torch.tensor([.5,.5]),atol=1e-6,rtol=0)


def test_incomplete_source_support_abstains_for_native_prior_fallback():
    from pixal3d.modules.raw_member_ray_posterior import raw_member_ray_posteriors
    target,source,valid,rays,patches,observed=pure_feature_fixture();valid[0,0,63]=False
    posterior=raw_member_ray_posteriors(target_keys=target,source_keys=source,source_valid=valid,
        ray_ids=rays,patch_ids=patches,observed_query=observed)
    assert not posterior.query_valid.any() and not posterior.probabilities.any()


def test_D1_cosine_keeps_frozen_decoder_but_physically_removes_scalar_head():
    from pixal3d.modules.d1_cosine_top1_source import D1CosineTop1SourceConditioner
    kwargs=dict(arm='c1',implementation='r9-native-v5',key_dim=1024,value_dim=1024,
        query_dim=1024,hidden_dim=256,topk=64,score_temperature=.05,
        projection_mode='native-bilinear',ray_mode='semantic-top1',ray_chunk_size=64,
        candidate_aggregation='top1')
    module=D1CosineTop1SourceConditioner(**kwargs)
    assert module.q2c is not None and module.q2c.fusion_head is None
    assert not any(p.requires_grad for p in module.q2c.parameters())
    assert sum(p.requires_grad for p in module.parameters())==10
    module.train();assert module.training and not module.q2c.training


def test_module_protocol_separates_decoder_from_scalar_score():
    from experiments.deformnext_retargeting.module_pipeline_protocol_v2 import protocol
    value=protocol();comparisons={r['effect']:r for r in value['module_contrasts']}
    assert len(value['arms'])==12 and value['evaluation']['module_family_size']==10
    assert comparisons['matching_decoder']['candidate']=='C_D1_COS'
    assert comparisons['matching_decoder']['reference']=='C_RAW'
    assert comparisons['matching_scalar_score']['candidate']=='C_DECODER'
    assert comparisons['matching_scalar_score']['reference']=='C_D1_COS'
