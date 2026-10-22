"""Protect the join against lineage, image identity and isolation confounds."""
import hashlib
import json

import numpy as np
import pytest

from experiments.deformnext_retargeting.generated_cascade_comparison import (
    REPLAY_MODELS, ZERO_MODES, evaluate_plan, validate_generation,
    validate_perceptual_reports,
)
from experiments.deformnext_retargeting.cascade_view_roles import summarize_view_roles


def generation(name):
    g = dict(sample_id='asset', actual_resolution=1024, source_defined_normalization=True,
             per_output_alignment=False, GT_geometry_or_coords_used_in_generation=False,
             fixed_shape_models=True, metrics=dict(chamfer_L1=.01, surface_FScore=.8,
             threshold=.02, prediction_samples=20000, target_samples=20000, per_output_alignment=False))
    if name in ZERO_MODES:
        g.update(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION',
            original_zero_shot_full_stack=True, conditioning_mode=ZERO_MODES[name],
            input_states=['M0'] * 4 if name == 'zero-source4' else ['M1', 'M0', 'M0', 'M0'],
            input_camera_ids=[8, 9, 10, 11], GT_loaded_only_after_generation=True,
            additional_PBR_conditioner=False, visibility_mask_used=False, deformation_finetuning_updates=0,
            original_stage_configs_and_weights_unchanged=True,
            original_multiview_steps=dict(ss64=4210, shape512=4210, shape1024=16850, pbr1024=16850),
            actual_RGB_input_files=[dict(path=f'{i}.png', sha256='a' * 64) for i in range(4)])
    else:
        arm = 'v1' if name == 'full-v1' else ('direct' if name.endswith('Direct') else 'retargeting')
        g.update(status='PASS_ACTUAL_NATIVE_PBR_REPLAY_ON_GENERATED_SHAPE',
            GT_sparse_latents_or_projection_cache_used=False,
            GT_evaluation_artifact_copied_only_after_prediction_saved=True,
            raw_surface_metrics_unchanged=True, geometry_array_byte_identity=dict(vertices=True),
            conditioning_preserves_CPU_and_CUDA_RNG=True, training_updates=0,
            original_cascade_texture_noise_reused=False, PBR_checkpoint_SHA256='checkpoint',
            PBR_arm=arm, additional_PBR_conditioner=(arm != 'v1'), sampler_class='native',
            sampler_params=dict(steps=12), PBR_replay_seed=17, PBR_noise_protocol='reset',
            geometry_evaluation_SHA256='geometry', SS_prediction_SHA256='ss',
            actual_initial_texture_noise_SHA256='noise', generated_shape_source='shape')
    return g


def official():
    return dict(status='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION',
        full_panel=True, fullframe_primary=True, separate_GT_only_crop_supplementary=True,
        GPU_sanity=dict(status='PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY'), no_model_training=True,
        global_environment_changed=False, metric_registry_SHA256='registry', model_weights={'CLIP': 'weights'},
        model_precision='float32', per_view=[])


@pytest.mark.parametrize('name', ['zero-source4', 'zero-edited-source3'])
def test_four_rgb_states_and_original_lora_lineage_required(name):
    g = generation(name)
    validate_generation(name, g, 'asset')
    g['input_states'] = ['M0'] * 3
    with pytest.raises(ValueError, match='four-RGB'):
        validate_generation(name, g, 'asset')
    g = generation(name)
    g['original_multiview_steps']['pbr1024'] = 762
    with pytest.raises(ValueError, match='four-RGB'):
        validate_generation(name, g, 'asset')


def test_replay_requires_actual_bound_checkpoint_and_no_gt_support():
    g = generation('iso-R')
    validate_generation('iso-R', g, 'asset', 'checkpoint')
    with pytest.raises(ValueError, match='replay contract'):
        validate_generation('iso-R', g, 'asset', 'other-checkpoint')
    g['GT_sparse_latents_or_projection_cache_used'] = True
    with pytest.raises(ValueError, match='replay contract'):
        validate_generation('iso-R', g, 'asset', 'checkpoint')


def test_official_reports_cannot_mix_weights_or_precision():
    a, b = official(), official()
    assert validate_perceptual_reports([a, b]) == {}
    b['model_precision'] = 'float16'
    with pytest.raises(ValueError, match='weights/registry/precision'):
        validate_perceptual_reports([a, b])


def fixture_plan(tmp_path, fault=None):
    """Synthetic receipt records test binding logic, not actual remote closure."""
    files = []

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        files.append(path)
        return str(path)

    receipt = tmp_path / 'file4-closure.json'
    models, reports, records = {}, [official(), official()], {}
    gt = np.zeros((3, 2, 2), dtype=np.float32)
    alpha = np.ones((1, 2, 2), dtype=np.float32)
    h = lambda x: hashlib.sha256(x.tobytes()).hexdigest()
    for index, name in enumerate([*REPLAY_MODELS, *ZERO_MODES]):
        folder = tmp_path / name
        g = generation(name)
        if name == 'iso-R' and fault == 'isolation':
            g['actual_initial_texture_noise_SHA256'] = 'different-noise'
        gp = save(folder / 'generation.json', g)
        views, values = [], {}
        for cid in [8, 9, 10]:
            pixels = folder / f'camera{cid:02d}.npz'
            current_gt = gt + 1 if fault == 'GT' and name == 'zero-source4' else gt
            np.savez(pixels, GT_shared_renderer=current_gt, GT_alpha=alpha)
            files.append(pixels)
            rgb_sha = h(current_gt)
            view = dict(camera_id=cid, PSNR=20. + index, SSIM=.9, LPIPS=.1, M_IoU=.8,
                        foreground_PSNR=21., fixed_GT_RGB_SHA256=rgb_sha, fixed_GT_alpha_SHA256=h(alpha))
            views.append(view)
            scores = dict(DreamSim_fullframe=.1, CLIP_cosine_fullframe=.9,
                DreamSim_GT_foreground_crop=.1, CLIP_cosine_GT_foreground_crop=.9)
            reports[index >= 5]['per_view'].append(dict(arm=name, sample_id='asset', camera_id=cid,
                source_cluster_id='cluster', shared_npz_SHA256=hashlib.sha256(pixels.read_bytes()).hexdigest(),
                scores=scores, image_audit=dict(original_float_GT_SHA256=rgb_sha,
                original_GT_alpha_SHA256=h(alpha), GT_only_crop_box=[0, 0, 2, 2],
                converted_image_SHA256=dict(reference_full=rgb_sha, reference_GTcrop=rgb_sha))))
            values[cid] = {k: view[k] for k in ['PSNR', 'SSIM', 'LPIPS', 'M_IoU', 'foreground_PSNR']}
            values[cid].update(DreamSim=.1, CLIP=.9, DreamSim_GTcrop=.1, CLIP_GTcrop=.9)
        rp = save(folder / 'render.json', dict(status='PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE',
            sample_id='asset', same_reference_pixels_for_every_model=True, per_view=views))
        models[name] = [dict(sample_id='asset', source_cluster_id='cluster', generation=gp, render=rp,
                            closed_inputs={gp: str(receipt), rp: str(receipt)})]
        # Match the real camera archive receipt convention.
        (folder / 'file4-closure.json').symlink_to(receipt)
        records[name] = [dict(sample_id='asset', source_cluster_id='cluster', views=values, CD=.01, FScore=.8)]
    if fault == 'scores':
        reports[1]['per_view'][0]['shared_npz_SHA256'] = 'wrong-pixels'
    paths = [save(tmp_path / f'perceptual{i}.json', r) for i, r in enumerate(reports)]
    contrasts = [['full-R', 'full-v1'], ['iso-R', 'full-v1']]
    old = summarize_view_roles({n: records[n] for n in REPLAY_MODELS}, contrasts, draws=10, seed=1)
    old_path = save(tmp_path / 'old.json', old)
    receipt.write_text(json.dumps(dict(status='PASS', local_equals_actual_file4_SHA256=True,
        files=[dict(relative_path=str(p.relative_to(tmp_path)), bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])))
    return dict(canonical_local_root=str(tmp_path), models=models, expected_assets=1,
        perceptuals={p: str(receipt) for p in paths}, PBR_checkpoint_SHA256={n: 'checkpoint' for n in REPLAY_MODELS},
        contrasts=contrasts + [['full-R', 'zero-edited-source3']], draws=10, seed=1,
        existing_summary=old_path, existing_summary_receipt=str(receipt))


def test_end_to_end_seven_model_join_preserves_existing_statistics(tmp_path):
    summary = evaluate_plan(fixture_plan(tmp_path))
    assert summary['actual_perceptual_views'] == 21
    assert summary['maximum_previous_five_model_allview_statistic_difference'] == 0
    assert summary['zero_vs_replay_texture_noise_streams_identical'] is False
    assert summary['model_adopted'] is False


@pytest.mark.parametrize('fault, message', [('GT', 'Cross-model'), ('scores', 'different rendered pixels'),
                                          ('isolation', 'PBR isolation')])
def test_join_rejects_cross_model_confound(tmp_path, fault, message):
    with pytest.raises(ValueError, match=message):
        evaluate_plan(fixture_plan(tmp_path, fault))
