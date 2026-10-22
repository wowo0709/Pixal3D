"""A matched native-task campaign, separate from source-feature proxy training."""
import copy
from contextlib import contextmanager


NAME = 'RMediatedNativeFlowTrainer'
SCHEMA = 'r-mediated-native-flow-cold-v1'
VARIANTS = {'vanilla': None, 'R_feature': 'top1-plus-feature', 'R_raw_topk': 'raw-topk-plus-feature'}
INITIAL = 'Exact native MV4 global/projection/negative; fresh cold source; no teacher initializer'


def validate_config(cfg, phase):
    if phase not in {'proof', 'capacity', 'science'}:
        raise ValueError('explicit native proof, capacity or scientific phase required')
    e, t = cfg['experiment'], cfg['trainer']['args']
    variant = e.get('native_variant')
    if (variant not in VARIANTS or cfg['trainer']['name'] != NAME
            or e.get('conditioning_schema') != SCHEMA or e.get('initial_function') != INITIAL
            or e.get('additional_source_teacher_supervision') is not False
            or e.get('source_teacher_updates') != 0 or e.get('native_task_objective') != 'SS_flow_mse'
            or e.get('engineering_only') is not (phase != 'science')
            or cfg.get('use_wandb') is not (phase == 'science')
            or cfg.get('training_seed') not in (101, 202)
            or cfg.get('load_dir') != '' or cfg.get('load_ckpt') is not None or cfg.get('ckpt') != 'none'
            or cfg.get('auto_retry') != 0 or cfg.get('num_gpus') != 1 or cfg.get('num_nodes') != 1
            or cfg.get('tryrun') is not False or cfg.get('profile') is not False):
        raise ValueError('fresh honest cold native-task identity, phase and world1 required')
    if (t['batch_size_per_gpu'] != 96 or t['batch_split'] != 1 or t['max_steps'] != 480
            or t.get('ffn_chunk_assets') not in (12,24) or t.get('prefetch_data') is not False
            or t.get('microbatch_to_device') is not True or t.get('num_workers') != 2
            or t.get('mix_precision_mode') != 'amp' or t.get('mix_precision_dtype') != 'bfloat16'
            or t.get('ema_rate') != [] or t.get('lora') != dict(rank=16, alpha=16, dropout=0.)
            or set(t.get('continuation_ckpt', {})) != {'denoiser'}
            or t.get('finetune_ckpt') is not None
            or t.get('optimizer') != dict(name='AdamW', args=dict(lr=1e-5, weight_decay=.01, betas=[.9, .95], eps=1e-8))
            or t.get('lr_scheduler') != dict(name='LinearWarmupLRScheduler', args=dict(warmup_steps=25))
            or t.get('p_uncond') != .1 or t.get('i_save') != 96
            or e.get('training_assets') != 9149 or e.get('epochs') != 5
            or e.get('updates_per_epoch') != 96 or e.get('target_modality') != 'rgb'
            or (e.get('global_batch'),e.get('world_size'),e.get('microbatch_per_gpu'))!=(96,1,96)):
        raise ValueError('common native SS96/micro96/accum1/480/AdamW/BF16/LoRA/bounded-FFN required')
    d, v = cfg['dataset'], cfg['validation']
    if (d['name'] != 'DeformNeXtSS64' or d['args']['expected_split'] != 'train'
            or d['args']['image_size'] != 512 or d['args']['expected_ss_schema'] != 'deformnext-ss64-m1-anchor-v3'
            or v['dataset_name'] != d['name'] or v['dataset_args']['expected_split'] != 'validation'
            or v['max_assets'] != 32 or v['every_steps'] != 96 or e.get('validate_at_start') is not True
            or v['quick_metrics'].get('panel_size') != 32 or v['quick_metrics'].get('generation_batch_size') != 8
            or v['quick_metrics'].get('sampling_steps') != 12
            or not v['quick_metrics'].get('panel_file') or not v['quick_metrics'].get('seed_map_file')):
        raise ValueError('aligned TRAIN9149 and fixed native initial/epoch32 RGB/GT/noise protocol required')
    if variant == 'vanilla':
        if (t.get('deform_conditioner') is not None or t.get('model_learning_rates', {}) != {}
                or t.get('ray_feature_utility') is not None
                or d['args']['transport_arm'] != 'v1' or v['dataset_args']['transport_arm'] != 'v1'):
            raise ValueError('vanilla must physically contain no source conditioner or utility')
    else:
        s = t['deform_conditioner']
        if (e.get('appearance_mode') != VARIANTS[variant] or e.get('residual_logit_bound') != 1.
                or t.get('model_learning_rates') != dict(deform_conditioner=1e-4)
                or set(t.get('ray_feature_utility', {})) != {'checkpoint', 'SHA256', 'completed'}
                or (s.get('arm'), s.get('ray_mode'), s.get('candidate_aggregation')) != ('c1', 'semantic-top1', 'top1')
                or s.get('projection_mode') != 'native-bilinear' or s.get('freeze_q2c') is not True
                or s.get('topk') != 64 or s.get('score_temperature') != .05 or s.get('ray_chunk_size') != 64
                or (s.get('key_dim'), s.get('query_dim'), s.get('value_dim')) != (1024, 1024, 1024)
                or s.get('task_attention') is not None or s.get('g_max')!=.25 or 'deformation_dim' in s
                or d['args']['transport_arm'] != 'c1' or v['dataset_args']['transport_arm'] != 'c1'):
            raise ValueError('explicit cold19 R-only source-camera query, bound1 and physical NoDQ required')
    return variant


def common_contract(cfg):
    """Keep genuine numerical/data fields, excluding only declared source differences."""
    c = copy.deepcopy(cfg)
    t = c['trainer']['args']
    for field in ['deform_conditioner', 'ray_feature_utility', 'model_learning_rates']:
        t.pop(field, None)
    for dataset in [c['dataset']['args'], c['validation']['dataset_args']]:
        for key in ['transport_arm', 'transport_implementation', 'expected_q2c_sha256',
                    'projection_root', 'match_root', 'task_matching_features']:
            dataset.pop(key, None)
    return dict(models=c['models'], trainer=t, dataset=c['dataset'], data_dir=c['data_dir'],
        validation=c['validation'], training_seed=c['training_seed'])


@contextmanager
def cold_source_binding(variant):
    """Extra module construction must not shift native data/noise RNG streams."""
    import torch
    from pixal3d.trainers.flow_matching import member_feature_source_ray as trainer
    from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
    from pixal3d.modules.r_mediated_source_contrast import RMediatedSourceContrastConditioner
    from .r_mediated_source_pilot import assert_R_only
    if variant not in VARIANTS:
        raise ValueError('declared native variant required')
    instances = []
    if variant == 'vanilla':
        yield instances
        return
    previous = trainer.MemberFeatureSourceRayConditioner
    if previous is not MemberFeatureSourceRayConditioner:
        raise ValueError('isolated uncombined source constructor binding required')

    class Configured(RMediatedSourceContrastConditioner):
        def __init__(self, **kwargs):
            devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                super().__init__(appearance_mode=VARIANTS[variant], residual_logit_bound=1., **kwargs)
            self.q2c.eval()
            self._native_cold_initialization = dict(variant=variant, constructor_RNG_preserved=True,
                source_teacher_updates=0, additional_source_teacher_supervision=False,
                strict_own_schema=True, R_only_delivery=True)
            def check_delivery(module, args, output):
                assert_R_only(module, output)
            self.register_forward_hook(check_delivery)
            instances.append(self)

        def train(self, mode=True):
            super().train(mode)
            self.q2c.eval()
            return self

    try:
        trainer.MemberFeatureSourceRayConditioner = Configured
        yield instances
    finally:
        trainer.MemberFeatureSourceRayConditioner = previous
