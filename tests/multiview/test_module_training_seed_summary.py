"""Synthetic record tests for replication accounting; no neural forwards."""
from copy import deepcopy
import hashlib

import pytest

from experiments.deformnext_retargeting.module_training_seed_summary import (
    ALIASES, SOURCE_COUNTS, models_for_selected, summarize, validate_training,
)
from experiments.deformnext_retargeting.module_pipeline_protocol_v2 import protocol
from test_module_confirmation_recipe import inputs as recipe_inputs


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def inputs(tmp_path, selected='FULL'):
    arms = models_for_selected(selected)
    configs, results, traces, starts = ({s: {} for s in (101, 202)} for _ in range(4))
    for arm in arms:
        folder = tmp_path/arm
        folder.mkdir()
        original, confirmation = recipe_inputs(folder, arm)
        for seed, cfg in [(101, original), (202, confirmation)]:
            configs[seed][arm] = cfg
            starts[seed][arm] = dict(status='PASS_ACTUAL_COLD_NATIVE_TASK_SS_STARTUP',
                variant=ALIASES.get(arm, arm), source_tensors=SOURCE_COUNTS[arm], LoRA_tensors=360,
                global_batch=96, microbatch=96, accumulation=1, actual_optimizer_states=0,
                post_constructor_RNG={k: sha((seed, k)) for k in ['CPU', 'CUDA', 'numpy', 'python']},
                initial_LoRA_SHA256=sha('same-original-LoRA4210'), TRAIN_order_SHA256=sha('TRAIN9149'))
            traces[seed][arm] = [dict(step=i+1,
                asset_ids=[['train'+str((i*96+j)%9149) for j in range(96)]],
                x_t_SHA256=sha((seed, i, 'x_t')), t_SHA256=sha((seed, i, 't')),
                global_SHA256=sha((i, 'same-conditioned-RGB'))) for i in range(480)]
            root = cfg['output_dir']
            results[seed][arm] = []
            for noise in range(3):
                rows = [dict(sample_id=str(i), source_cluster_id='source'+str(i//2), seed=500+i+noise*10000,
                    target_voxels_sha256=sha(('GT', i)), voxel_iou=.4 if arm == 'V1' else .41 if arm == 'C_DECODER' else .42,
                    voxel_precision=.5, voxel_recall=.6, voxel_f1=.55) for i in range(128)]
                results[seed][arm].append(dict(status='PASS', stage='ss64', step=480,
                    native_variant=ALIASES.get(arm, arm), actual_source_tensors=SOURCE_COUNTS[arm],
                    actual_LoRA_tensors=360, evaluation_mode='final128', optimizer_updates_during_evaluation=0,
                    weights='raw', generation_batch_size=8, sampling_steps=12, guidance=3.,
                    deterministic_algorithms=True, checkpoint=root+'/ckpts/denoiser_step0000480.pt',
                    conditioner_checkpoint=None if arm == 'V1' else root+'/ckpts/deform_conditioner_step0000480.pt',
                    arm_config=root+'/config.json', input_states=['M1', 'M0', 'M0', 'M0'],
                    input_camera_ids=[8, 9, 10, 11], per_asset=rows,
                    verified_inputs={root+'/'+name: sha((seed, arm, name)) for name in
                        ['config.json', 'ckpts/denoiser_step0000480.pt', 'ckpts/deform_conditioner_step0000480.pt']}))
    return results, configs, traces, starts


@pytest.mark.parametrize('selected', protocol()['arms'])
def test_only_selected_candidate_and_deduplicated_strong_controls_are_compared(tmp_path, selected):
    data = inputs(tmp_path, selected)
    result = summarize(*data, selected=selected, draws=100)
    assert result['measured_models'] == list(models_for_selected(selected))
    assert len(result['measured_models']) == (2 if selected in ('V1', 'C_DECODER') else 3)
    assert result['training_seeds'] == [101, 202] and result['generation_noises'] == 3
    assert result['assets'] == 128 and result['source_clusters'] == 64
    assert result['source_cluster_CIs_condition_on_two_fixed_training_seeds_not_seed_population']
    assert not result['automatic_model_adoption']
    if selected not in ('V1', 'C_DECODER'):
        c = result['fixed_two_seed_average'][selected+'_minus_C_DECODER']
        assert c['mean_asset_delta']['voxel_iou'] == pytest.approx(.01)
        assert c['IoU_family_size'] == 3
    assert result['actual_frozen_base_and_checkpoint_byte_evidence_required_separately']


def test_positive_average_cannot_hide_negative_independent_seed(tmp_path):
    r, c, t, s = inputs(tmp_path)
    for value in r[202]['FULL']:
        for row in value['per_asset']:
            row['voxel_iou'] = .39
    out = summarize(r, c, t, s, selected='FULL', draws=100)
    effect = out['fixed_two_seed_average']['FULL_minus_V1']
    assert effect['mean_asset_delta']['voxel_iou'] > 0
    assert not effect['mean_IoU_direction_positive_in_both_training_seeds']


@pytest.mark.parametrize('damage', ['missing-train-seed', 'missing-model', 'missing-noise',
    'noise-repeated', 'GT', 'cluster', 'row-order', 'wrong-model', 'wrong-checkpoint',
    'source-count', 'float-source-count', 'LoRA-count', 'trained-during-eval', 'nondeterministic',
    'nonfinite', 'out-of-range', 'seed-config', 'FFN', 'initializer', 'teacher',
    'trace-noise', 'trace-incomplete', 'trace-batch', 'same-training-stream',
    'same-training-RNG', 'startup-LoRA', 'startup-batch', 'constructor-RNG',
    'wrong-sampler', 'EMA', 'missing-weight-proof', 'changed-weights-between-noises'])
def test_unpaired_models_or_changed_training_cannot_be_called_replication(tmp_path, damage):
    r, c, t, s = inputs(tmp_path)
    value = r[202]['FULL'][0]
    if damage == 'missing-train-seed':r.pop(202)
    elif damage == 'missing-model':r[202].pop('C_DECODER')
    elif damage == 'missing-noise':r[202]['FULL'].pop()
    elif damage == 'noise-repeated':r[202]['FULL'][1] = deepcopy(value)
    elif damage == 'GT':value['per_asset'][0]['target_voxels_sha256'] = sha('different-GT')
    elif damage == 'cluster':value['per_asset'][0]['source_cluster_id'] = 'different-source'
    elif damage == 'row-order':value['per_asset'].reverse()
    elif damage == 'wrong-model':value['native_variant'] = 'C_FIXED_TOPK'
    elif damage == 'wrong-checkpoint':value['checkpoint'] = '/another/model.pt'
    elif damage == 'source-count':value['actual_source_tensors'] = 12
    elif damage == 'float-source-count':value['actual_source_tensors'] = 19.0
    elif damage == 'LoRA-count':value['actual_LoRA_tensors'] = 0
    elif damage == 'trained-during-eval':value['optimizer_updates_during_evaluation'] = 1
    elif damage == 'nondeterministic':value['deterministic_algorithms'] = False
    elif damage == 'nonfinite':value['per_asset'][0]['voxel_iou'] = float('nan')
    elif damage == 'out-of-range':value['per_asset'][0]['voxel_iou'] = 1.2
    elif damage == 'seed-config':c[202]['FULL']['training_seed'] = 101
    elif damage == 'FFN':c[202]['FULL']['trainer']['args']['ffn_chunk_assets'] = 12
    elif damage == 'initializer':c[202]['FULL']['trainer']['args']['continuation_ckpt']['denoiser'] = '/trained480.pt'
    elif damage == 'teacher':c[202]['FULL']['experiment']['source_teacher_updates'] = 112
    elif damage == 'trace-noise':t[202]['FULL'][0]['x_t_SHA256'] = sha('different-forward')
    elif damage == 'trace-incomplete':t[202]['FULL'].pop()
    elif damage == 'trace-batch':t[202]['FULL'][0]['asset_ids'][0].pop()
    elif damage == 'same-training-stream':t[202] = deepcopy(t[101])
    elif damage == 'same-training-RNG':
        for arm in s[202]:s[202][arm]['post_constructor_RNG'] = deepcopy(s[101][arm]['post_constructor_RNG'])
    elif damage == 'startup-LoRA':s[202]['FULL']['initial_LoRA_SHA256'] = sha('warm-LoRA')
    elif damage == 'startup-batch':s[202]['FULL']['microbatch'] = 48
    elif damage == 'constructor-RNG':s[202]['FULL']['post_constructor_RNG']['CUDA'] = sha('shifted-RNG')
    elif damage == 'wrong-sampler':value['sampling_steps'] = 18
    elif damage == 'EMA':value['weights'] = 'ema'
    elif damage == 'missing-weight-proof':value.pop('verified_inputs')
    elif damage == 'changed-weights-between-noises':value['verified_inputs'][value['checkpoint']] = sha('different-model')
    with pytest.raises(ValueError):
        summarize(r, c, t, s, selected='FULL', draws=100)


def test_only_documented_device_metadata_is_exempted_from_cross_seed_recipe_equality(tmp_path):
    data = inputs(tmp_path, 'C_FIXED_TOPK')
    data[1][101]['C_FIXED_TOPK']['experiment'].update(assigned_GPUs=['0', '1', '2', '3'],
        CPU_affinity_per_GPU={'0': [84, 85], '1': [86, 87], '2': [88, 89], '3': [90, 91]})
    out = summarize(*data, selected='C_FIXED_TOPK', draws=100)
    assert out['execution_assignments_recorded']['C_FIXED_TOPK']['assigned_GPUs']['101'] == ['0', '1', '2', '3']
    assert not out['automatic_model_adoption']
