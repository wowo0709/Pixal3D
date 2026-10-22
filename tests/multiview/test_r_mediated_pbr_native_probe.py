from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.probe_r_mediated_pbr_native import validate_config
from test_r_mediated_pbr_trainer import source_spec


def fixture(mode):
    original = dict(models={'denoiser': dict(name='ElasticSLatFlowModel', args=dict(resolution=64,
        in_channels=64, out_channels=32, proj_in_channels=2048))},
        trainer=dict(args=dict(lora=dict(rank=16, alpha=16, dropout=0.))))
    t = dict(batch_size_per_gpu=1, batch_split=1, max_steps=1900, num_workers=0,
        prefetch_data=False, microbatch_to_device=True, ema_rate=[], p_uncond=.1,
        optimizer=dict(name='AdamW', args=dict(lr=5e-6, weight_decay=.01, betas=[.9, .95], eps=1e-8)),
        lr_scheduler=dict(name='LinearWarmupLRScheduler', args=dict(warmup_steps=25)),
        mix_precision_mode='amp', mix_precision_dtype='bfloat16',
        lora=deepcopy(original['trainer']['args']['lora']),
        continuation_ckpt={'denoiser': '/declared/full-original-PBR16850.pt'}, finetune_ckpt=None)
    d = dict(name='DeformNeXtPBR1024', args=dict(max_tokens=32768, expected_split='train', transport_arm='v1'))
    trainer = 'ImageConditionedProjSparseFlowMatchingCFGTrainer'
    if mode != 'vanilla':
        trainer = 'RMediatedPBRNativeTrainer'
        t.update(deform_conditioner=source_spec(mode), model_learning_rates={'deform_conditioner': 1e-4})
        d.update(name='DeformNeXtFeaturePBR1024')
        d['args'].update(transport_arm='c1', task_matching_features=True)
    return dict(models=deepcopy(original['models']), training_seed=101, use_wandb=False,
        experiment=dict(scientific_training_admitted=False, source_multiview_step=16850, PBR_source_mode=mode),
        trainer=dict(name=trainer, args=t), dataset=d), original


@pytest.mark.parametrize('mode', ['vanilla', 'top1-only', 'top1-plus-feature', 'raw-topk-plus-feature'])
def test_native_PBR_proof_preserves_original_model_and_rejects_warm_or_wrong_scope(mode):
    cfg, original = fixture(mode)
    validate_config(cfg, original, mode)
    for key, value in [('batch_size_per_gpu', 6), ('batch_split', 2), ('p_uncond', 0.),
                       ('lora', dict(rank=8, alpha=8, dropout=0.))]:
        bad = deepcopy(cfg); bad['trainer']['args'][key] = value
        with pytest.raises(ValueError): validate_config(bad, original, mode)
    for key, value in [('source_multiview_step', 762), ('scientific_training_admitted', True)]:
        bad = deepcopy(cfg); bad['experiment'][key] = value
        with pytest.raises(ValueError): validate_config(bad, original, mode)
    bad = deepcopy(cfg); bad['models']['denoiser']['args']['proj_in_channels'] = 1024
    with pytest.raises(ValueError): validate_config(bad, original, mode)
