from copy import deepcopy
import hashlib

import pytest

from experiments.deformnext_retargeting.common_geometry_pbr_contract import (
    CASE_STATUS, GEOMETRY_FIELDS, MODES, POLICY, summarize, validate_replay, validate_stage,
)
from test_r_mediated_pbr_science import recipe
from test_r_pbr_generated_summary import inputs as summary_inputs


def sha(x):return hashlib.sha256(str(x).encode()).hexdigest()


def stage(mode):
    cfg, _, _ = recipe(mode, world=4)
    n = 0 if mode == 'vanilla' else 10 if mode == 'top1-only' else 19
    rows = [dict(status='PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE', rank=i, mode=mode,
        training_seed=101, source_multiview_step=16850, world_size=4, global_batch=24, microbatch=6,
        accumulation=1, actual_optimizer_updates=1900, actual_optimizer_states=360+n,
        actual_training_assets=9139, actual_dataloader_batches=380, source_tensors=n,
        fresh_optimizer=True, initial4_identity_before_first_update=True,
        frozen_base_matcher_and_anchors_exact=True, post_training_DDP_parameters_exact=True,
        stage_training_complete=True) for i in range(4)]
    return cfg, rows


@pytest.mark.parametrize('mode', MODES)
def test_each_actual_stage_requires_own_complete_native1900_and_source_family(mode):
    cfg, rows = stage(mode)
    assert validate_stage(cfg, mode, rows) == (0 if mode == 'vanilla' else 10 if mode == 'top1-only' else 19)


@pytest.mark.parametrize('damage', ['rank', 'step', 'seed', 'mode', 'source', 'optimizer', 'DDP', 'frozen', 'teacher'])
def test_engineering_or_other_model_is_not_scientific_replay_admission(damage):
    cfg, rows = stage('top1-only')
    if damage == 'rank':rows.pop()
    elif damage == 'step':rows[0]['actual_optimizer_updates'] = 3
    elif damage == 'seed':rows[0]['training_seed'] = 202
    elif damage == 'mode':cfg['experiment']['PBR_source_mode'] = 'vanilla'
    elif damage == 'source':rows[0]['source_tensors'] = 19
    elif damage == 'optimizer':rows[0]['fresh_optimizer'] = False
    elif damage == 'DDP':rows[0]['post_training_DDP_parameters_exact'] = False
    elif damage == 'frozen':rows[0]['frozen_base_matcher_and_anchors_exact'] = False
    else:cfg['experiment']['source_teacher_updates'] = 112
    with pytest.raises(ValueError):validate_stage(cfg, 'top1-only', rows)


def replay():
    g, _, _, _ = summary_inputs()
    r = g['top1-only']['0']
    r.update(PBR_checkpoint_SHA256=sha('model'), original_prediction_SHA256=sha('prediction'),
        SS_prediction_SHA256=sha('SS'), SS_source_evaluation='/V1/ss-evaluation.json',
        geometry_array_byte_identity={k: True for k in GEOMETRY_FIELDS},
        actual_initial_texture_noise_SHA256=sha('noise'), GT_sparse_latents_or_projection_cache_used=False,
        per_output_alignment=False, texture_support_policy=POLICY, shape_decoder_recomputed=False,
        actual_native_guide_archive_SHA256=sha('guide'),
        actual_native_guide_logits_order_scale_and_dead_branches_preserved=True,
        numerical_entry_AST_device_literal_only=dict(all_other_AST_statements_unchanged=True))
    base = dict(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION',
        sample_id='0', SS_prediction_SHA256=sha('SS'), SS_source_evaluation='/V1/ss-evaluation.json',
        metrics=deepcopy(r['metrics']), GT_geometry_or_coords_used_in_generation=False,
        per_output_alignment=False, texture_support_policy=POLICY,
        actual_native_guide_archive_SHA256=sha('guide'),
        actual_native_guide_logits_order_scale_and_dead_branches_preserved=True)
    return r, base


@pytest.mark.parametrize('damage', ['none', 'GT', 'guide', 'reconstruct', 'geometry', 'different-SS', 'different-model', 'changed-math'])
def test_same_real_native_guides_and_image_only_sampling_are_required(damage):
    r, base = replay()
    if damage == 'GT':r['GT_sparse_latents_or_projection_cache_used'] = True
    elif damage == 'guide':r['actual_native_guide_archive_SHA256'] = sha('different-guide')
    elif damage == 'reconstruct':r['shape_decoder_recomputed'] = True
    elif damage == 'geometry':r['geometry_array_byte_identity']['vertices'] = False
    elif damage == 'different-SS':r['SS_prediction_SHA256'] = sha('other-SS')
    elif damage == 'different-model':r['PBR_checkpoint_SHA256'] = sha('other-model')
    elif damage == 'changed-math':r['numerical_entry_AST_device_literal_only']['all_other_AST_statements_unchanged'] = False
    if damage == 'none':
        validate_replay(r, base, mode='top1-only', model_SHA256=sha('model'), prediction_SHA256=sha('prediction'))
    else:
        with pytest.raises(ValueError):
            validate_replay(r, base, mode='top1-only', model_SHA256=sha('model'), prediction_SHA256=sha('prediction'))


def joined():
    generations, shared, official, kw = summary_inputs()
    cases = {}
    for mode in MODES:
        cases[mode] = {}
        for sid, g in generations[mode].items():
            g.update(PBR_checkpoint_SHA256=sha(mode), actual_native_guide_archive_SHA256=sha(('guide',sid)),
                texture_support_policy=POLICY, shape_decoder_recomputed=False,
                actual_native_guide_logits_order_scale_and_dead_branches_preserved=True)
            cases[mode][sid] = dict(status=CASE_STATUS, mode=mode, sample_id=sid,
                source_cluster_id=kw['clusters'][sid], new_SS_forwards=0, new_Shape_forwards=0,
                new_PBR_sampling_calls=1, PBR_checkpoint_SHA256=sha(mode), common_prediction_SHA256=g['original_prediction_SHA256'],
                native_guide_SHA256=g['actual_native_guide_archive_SHA256'],
                actual_owned_terminals=[dict(returncode=0,actual_session_drained=True) for _ in range(3)])
    return cases, generations, shared, official, kw


def test_four_modes_reuse_geometry_and_still_report_all_eight_role_metrics():
    c, g, r, p, kw = joined()
    out = summarize(c, g, r, p, **kw)
    assert out['status'] == 'PASS_ACTUAL_CORE4_PBR1900_FIXED_NATIVE_GUIDE32_EIGHT_METRICS'
    assert out['actual_same_generated_geometry_and_native_guides'] and out['new_SS_or_Shape_forwards'] == 0
    assert not out['selected_model_adopted']
    assert 'LPIPS' in out['roles']['source9_10']['means']['vanilla']['deformation_weighted']


@pytest.mark.parametrize('damage', ['guide', 'child-failed', 'not-drained', 'extra-Shape', 'different-noise', 'different-GT'])
def test_complete_join_rejects_geometry_sampler_process_and_reference_confound(damage):
    c, g, r, p, kw = joined()
    case = c['top1-only']['0']
    if damage == 'guide':case['native_guide_SHA256'] = sha('different-guide')
    elif damage == 'child-failed':case['actual_owned_terminals'][0]['returncode'] = 1
    elif damage == 'not-drained':case['actual_owned_terminals'][0]['actual_session_drained'] = False
    elif damage == 'extra-Shape':case['new_Shape_forwards'] = 1
    elif damage == 'different-noise':g['top1-only']['0']['actual_initial_texture_noise_SHA256'] = 'different-noise'
    elif damage == 'different-GT':r['top1-only']['0']['per_view'][0]['fixed_GT_RGB_SHA256'] = 'different-GT'
    with pytest.raises(ValueError):summarize(c, g, r, p, **kw)
