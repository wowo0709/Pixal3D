"""All module controls share the ray query/carrier/R/data and fixed training budget."""
import copy

import pytest

from experiments.deformnext_retargeting.ray_group_study import ARMS, recipe, validate_profile, validate_runtime_backend


def source():
    spec = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
        query_dim=1024, topk=64, hidden_dim=256, g_max=.25, freeze_q2c=True,
        q2c_checkpoint='/external/selected-frozen-head.pt')
    data = dict(transport_implementation='r9-native-v5', transport_arm='c1',
                match_root='/external/old-2d-match-cache', task_matching_features=True)
    return dict(num_gpus=2, training_seed=101, config='/old/config.json',
        trainer=dict(name='OldTrainer', args=dict(deform_conditioner=spec, max_steps=480,
                    batch_size_per_gpu=48, batch_split=1, num_workers=8,
                    continuation_ckpt={'denoiser': '/external/originalSS4210.pt'})),
        dataset=dict(name='DeformNeXtSS64', args=copy.deepcopy(data)),
        validation=dict(dataset_args=copy.deepcopy(data), roots_file='/external/validation508.jsonl'),
        experiment=dict(training_seed=101, arm='old', required_microbatch_per_gpu=48),
        default_output_dir='/old/run', output_dir='/old/run', ckpt='none', load_dir='')


def test_every_arm_preserves_continuation_data_ray_unit_and_global96_budget(tmp_path):
    original = source()
    before = copy.deepcopy(original)
    root = tmp_path / 'ray-group-study-v1'
    configs = {name: recipe(original, root, name) for name in ARMS}
    assert original == before
    for name, cfg in configs.items():
        validate_profile(cfg, root)
        args = cfg['trainer']['args']
        spec = args['deform_conditioner']
        assert args['continuation_ckpt'] == original['trainer']['args']['continuation_ckpt']
        assert cfg['validation']['roots_file'] == original['validation']['roots_file']
        assert spec['projection_mode'] == 'native-bilinear' and spec['score_temperature'] == .05
        assert spec['g_max'] == .25 and spec['query_dim'] == 1024 and spec['arm'] == 'c1'
        assert spec.get('deformation_dim') is None
        assert cfg['num_gpus'] * args['batch_size_per_gpu'] == 96
        assert args['batch_split'] == 1 and args['max_steps'] == 480
        assert cfg['experiment']['inference_gt_inputs'] is False
        assert cfg['dataset']['args']['match_root'] is None
    assert len({cfg['master_port'] for cfg in configs.values()}) == len(ARMS)
    assert configs['ray-top1']['trainer']['args']['deform_conditioner']['task_attention'] is None
    assert configs['ray-top1']['trainer']['args']['deform_conditioner']['candidate_aggregation'] == 'top1'
    assert configs['ray-weighted']['trainer']['args']['deform_conditioner']['task_attention'] is None
    assert configs['ray-no-decoder']['trainer']['args']['deform_conditioner']['task_attention']['decoder'] is False


@pytest.mark.parametrize('change', ['budget', 'old_matches', 'retargeter', 'DQ', 'uniform_score'])
def test_module_control_cannot_silently_change_the_common_baseline(tmp_path, change):
    root = tmp_path / 'ray-group-study-v1'
    cfg = recipe(source(), root, 'ray-full')
    if change == 'budget':
        cfg['trainer']['args']['batch_size_per_gpu'] = 24
    elif change == 'old_matches':
        cfg['validation']['dataset_args']['match_root'] = '/old/matches'
    elif change == 'retargeter':
        cfg['trainer']['args']['deform_conditioner']['g_max'] = .5
    elif change == 'DQ':
        cfg['trainer']['args']['deform_conditioner']['deformation_dim'] = 2048
    else:
        cfg['trainer']['args']['deform_conditioner']['task_attention']['score_prior'] = False
    with pytest.raises(ValueError):
        validate_profile(cfg, root)


def test_gpu_entry_requires_the_observed_reference_backend_before_cuda_allocation(monkeypatch):
    monkeypatch.delenv('ATTN_BACKEND', raising=False)
    monkeypatch.delenv('PYTORCH_CUDA_ALLOC_CONF', raising=False)
    with pytest.raises(ValueError, match='reference SS runtime'):
        validate_runtime_backend()
    monkeypatch.setenv('ATTN_BACKEND', 'sdpa')
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    validate_runtime_backend()
