import json
import sys

import pytest
import torch

from experiments.deformnext_retargeting import archive_bounded_member_feature_ss as entry
from experiments.deformnext_retargeting import evaluate_member_feature_ss_artifact_subset as producer
from experiments.deformnext_retargeting import member_feature_value_control as value_module
from pixal3d.trainers.flow_matching import member_feature_source_ray as trainer_module
from test_member_feature_source_ray import pair


@pytest.mark.parametrize('fails', [False, True])
def test_archive_binds_correct_model_and_restores_after_success_or_error(tmp_path, monkeypatch, fails):
    original_class, original_control = trainer_module.MemberFeatureSourceRayConditioner, value_module.feature_value_control
    _, _, data = pair()
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(trainer=dict(name='MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer'),
        experiment=dict(attention_parameterization='bounded-matching-prior-functional-residual-v1',residual_logit_bound=1.))))
    remaining = ['--evaluation-module','member_feature_runtime','--arm-config',str(config),'--output',str(tmp_path)]
    def native():
        assert sys.argv[1:] == remaining
        assert trainer_module.MemberFeatureSourceRayConditioner is not original_class
        assert value_module.feature_value_control is not original_control
        if fails:raise RuntimeError('native archive failed')
        model = trainer_module.MemberFeatureSourceRayConditioner(arm='c1',implementation='r9-native-v5',
            key_dim=1024,value_dim=1024,query_dim=1024,hidden_dim=256,topk=64,score_temperature=.5,
            projection_mode='native-bilinear',ray_mode='semantic-top1',ray_chunk_size=4,candidate_aggregation='top1')
        with torch.no_grad():model(**data)
        (tmp_path/'evaluation.json').write_text(json.dumps(dict(step=480,optimizer_updates_during_evaluation=0,
            native_SS_artifacts=dict(archived_asset_count=32))))
    monkeypatch.setattr(producer,'main',native)
    argv=['bounded-archive','--bounded-prior-logit-bound','1.0',*remaining];monkeypatch.setattr(sys,'argv',argv)
    if fails:
        with pytest.raises(RuntimeError,match='native archive failed'):entry.main()
    else:
        entry.main();result=json.loads((tmp_path/'evaluation.json').read_text())
        assert result['bounded_prior_model_binding']['optimizer_updates']==0
        assert (tmp_path/'bounded-prior-archive-binding.json').exists()
    assert sys.argv is argv and trainer_module.MemberFeatureSourceRayConditioner is original_class
    assert value_module.feature_value_control is original_control
