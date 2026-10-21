import copy

import numpy as np
import pytest

from experiments.deformnext_retargeting.frozen_cascade_metrics import (
    PinnedFiles, read_terminal, verify_native_composition, verify_RGB_inputs, verify_texture_replay, verify_shared_reference, prediction_pixel_identity,
)
from experiments.deformnext_retargeting.evaluate_bound_population import sha
from experiments.deformnext_retargeting.pbr_replay_artifacts import GEOMETRY_FIELDS


def fixture():
    pin = lambda path: dict(path=path, SHA256='fixture-only')
    model = dict(ss64=dict(checkpoint=pin('/ss.pt'), conditioner_checkpoint=pin('/ss-source.pt'),
                          arm_config=pin('/ss.json'), step=480, conditioning_mode='edited-rgb'),
        stages={name: dict(config=pin('/'+name+'.json'), checkpoint=pin('/'+name+'.pt'))
                for name in ['shape512', 'shape1024', 'pbr1024']},
        pipeline_config=pin('/pipeline.json'), pbr_conditioner_checkpoint=None,
        original_zero_shot_stack=False, texture_protocol='native-inline')
    ss = dict(status='PASS', stage='ss64', weights='raw', step=480, optimizer_updates_during_evaluation=0,
        checkpoint='/ss.pt', conditioner_checkpoint='/ss-source.pt', arm_config='/ss.json',
        conditioning_mode='edited-rgb', target_modality='rgb', input_view_count=4, input_camera_ids=[8, 9, 10, 11],
        native_SS_artifacts=dict(artifacts=[dict(sample_id='asset', prediction_voxels_SHA256='predicted-SS')]))
    native = dict(row=dict(sample_id='asset'), seed=540, SS_evaluation='/SS/evaluation.json',
        input_states=['M1', 'M0', 'M0', 'M0'], RGB_paths=['/cache/target.png', '/cache/9.png', '/cache/10.png', '/cache/11.png'],
        stages={name: {key: value['path'] for key, value in spec.items()} for name, spec in model['stages'].items()},
        pipeline_config='/pipeline.json', source_geometry='/cache/source.npz', target_vertices='/cache/target.npy')
    generated = dict(sample_id='asset', SS_source_evaluation='/SS/evaluation.json', SS_prediction_SHA256='predicted-SS',
        actual_resolution=1024, source_defined_normalization=True, per_output_alignment=False,
        GT_geometry_or_coords_used_in_generation=False, GT_loaded_only_after_generation=True,
        additional_PBR_conditioner=False, PBR_checkpoint='/pbr1024.pt',
        metrics=dict(threshold=.02, prediction_samples=20000, target_samples=20000,
                     per_output_alignment=False, chamfer_L1=.01, surface_FScore=.9))
    return native, generated, ss, model


def verify(f):
    native, generated, ss, model = f
    verify_native_composition(native, generated, ss, model, sample_id='asset', seed=540)


def test_actual_native_full_composition_and_four_RGB_views_agree():
    verify(fixture())


@pytest.mark.parametrize('mode', ['source-only', 'edited-rgb'])
def test_original_zero_shot_uses_all_four_original_stages_and_declared_four_RGB_inputs(mode):
    f = fixture(); native, generated, ss, model = f
    model.update(original_zero_shot_stack=True)
    model['ss64'].update(step=4210, conditioner_checkpoint=None, conditioning_mode=mode)
    ss.update(step=4210, conditioner_checkpoint=None, conditioning_mode=mode)
    native['input_states'] = ['M0']*4 if mode == 'source-only' else ['M1','M0','M0','M0']
    generated.update(original_zero_shot_full_stack=True, deformation_finetuning_updates=0,
        original_multiview_steps=dict(ss64=4210, shape512=4210, shape1024=16850, pbr1024=16850))
    verify(f)
    generated['original_multiview_steps']['pbr1024'] = 762
    with pytest.raises(ValueError): verify(f)


@pytest.mark.parametrize('problem', ['SS-checkpoint', 'SS-conditioner', 'SS-config', 'step', 'EMA', 'GT-support',
    'depth-input', 'three-views', 'noise', 'shape512', 'shape1024', 'PBR', 'pipeline', 'predicted-SS',
    'per-output-alignment', 'surface-count', 'F-threshold', 'unexpected-PBR-module'])
def test_wrong_weights_GT_leakage_and_mismatched_native_protocol_are_rejected(problem):
    f = fixture(); native, generated, ss, model = f
    if problem == 'SS-checkpoint': ss['checkpoint'] = '/other.pt'
    elif problem == 'SS-conditioner': ss['conditioner_checkpoint'] = None
    elif problem == 'SS-config': ss['arm_config'] = '/other.json'
    elif problem == 'step': ss['step'] = 96
    elif problem == 'EMA': ss['weights'] = 'ema'
    elif problem == 'GT-support': generated['GT_geometry_or_coords_used_in_generation'] = True
    elif problem == 'depth-input': ss['target_modality'] = 'depth'
    elif problem == 'three-views': ss['input_view_count'] = 3
    elif problem == 'noise': native['seed'] = 541
    elif problem in ['shape512', 'shape1024']: native['stages'][problem]['checkpoint'] = '/other.pt'
    elif problem == 'PBR': native['stages']['pbr1024']['checkpoint'] = '/original-zero-PBR.pt'
    elif problem == 'pipeline': native['pipeline_config'] = '/different-sampler.json'
    elif problem == 'predicted-SS': generated['SS_prediction_SHA256'] = 'GT-SS'
    elif problem == 'per-output-alignment': generated['per_output_alignment'] = True
    elif problem == 'surface-count': generated['metrics']['target_samples'] = 1000
    elif problem == 'F-threshold': generated['metrics']['threshold'] = .1
    else: generated['additional_PBR_conditioner'] = True
    with pytest.raises(ValueError): verify(f)


def test_source_only_keeps_original_camera8_anchor_and_three_other_sources():
    native, generated, ss, model = fixture()
    row = dict(source_anchor_rgb='/file4/source8.png', target_rgb_input='/file4/target.png',
               source_images=['/file4/9.png', '/file4/10.png', '/file4/11.png'],
               conditioning_camera_ids=[8, 9, 10, 11], source_geometry='/file4/source.npz', target_vertices='/file4/target.npy')
    native['RGB_paths'][0] = '/cache/source8.png'
    verify_RGB_inputs(native, row, mode='source-only', file4_cache_root='/cache')
    native['RGB_paths'][0] = '/cache/target.png'
    with pytest.raises(ValueError): verify_RGB_inputs(native, row, mode='source-only', file4_cache_root='/cache')


def test_pinned_weight_changes_are_detected_even_when_other_cases_share_the_path(tmp_path):
    path = tmp_path/'weight.pt'; path.write_bytes(b'original')
    spec = dict(path=str(path), SHA256=sha(path)); pins = PinnedFiles()
    assert pins.check(spec) == path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError): pins.check(spec)
    with pytest.raises(ValueError): pins.finish()


def test_actual_terminal_must_execute_the_receipted_native_or_exact_remote_plan():
    module = 'experiments.deformnext_retargeting.cascade_from_ss_artifact'
    terminal = dict(returncode=0, command=['python', '-m', module, '--plan', '/native.json'])
    read_terminal(terminal, '/native.json', None, read=lambda _: None, module=module)
    with pytest.raises(ValueError): read_terminal(terminal, '/other.json', None, read=lambda _: None, module=module)
    terminal['returncode'] = 1
    with pytest.raises(ValueError): read_terminal(terminal, '/native.json', None, read=lambda _: None, module=module)


def test_original_reference_creator_identity_comes_from_bound_native_pixels_not_an_invented_ID():
    creator = dict(status='PASS_ACTUAL_SHARED_RENDERER_MATERIAL_REFERENCE',
        shared_renderer_lighting_camera_AO_seed=True, GT_VAE_reconstruction_not_used=True,
        prediction_renders_reused_not_regenerated=True, GT_not_used_for_model_generation=True,
        maximum_GT_world_vertex_nearest_difference=1e-7)
    verify_shared_reference(creator, sample_id='asset')
    creator['maximum_GT_world_vertex_nearest_difference'] = .1
    with pytest.raises(ValueError): verify_shared_reference(creator, sample_id='asset')
    copied = dict(status='PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE', sample_id='wrong', same_reference_pixels_for_every_model=True)
    with pytest.raises(ValueError): verify_shared_reference(copied, sample_id='asset')


def test_actual_native_shaded_alpha_and_shared_prediction_arrays_have_identical_pixels():
    rgb = np.ones((3, 512, 512), dtype=np.float32); alpha = np.zeros((512, 512), dtype=np.float32)
    native = dict(shaded=rgb, alpha=alpha)
    shared = dict(prediction=rgb.copy(), prediction_alpha=alpha.copy())
    assert prediction_pixel_identity(native, native=True) == prediction_pixel_identity(shared, native=False)
    shared['prediction'][0, 0, 0] = 0
    assert prediction_pixel_identity(native, native=True) != prediction_pixel_identity(shared, native=False)


@pytest.mark.parametrize('problem', [None, 'geometry', 'partial-geometry-audit', 'GT-support', 'checkpoint', 'conditioner', 'noise', 'source-SS'])
def test_generated_shape_PBR_replay_binding_preserves_geometry_and_final_weights(problem):
    _, original, _, model = fixture(); model['texture_protocol'] = 'native-replay'
    model['pbr_conditioner_checkpoint'] = dict(path='/PBR-source.pt', SHA256='fixture-only')
    replay = dict(PBR_replay_seed=906, PBR_config='/pbr1024.json', PBR_checkpoint='/pbr1024.pt', conditioner_checkpoint='/PBR-source.pt')
    actual = copy.deepcopy(original)
    actual.update(status='PASS_ACTUAL_NATIVE_PBR_REPLAY_ON_GENERATED_SHAPE', PBR_replay_seed=906,
        PBR_checkpoint_SHA256='fixture-only', conditioner_checkpoint='/PBR-source.pt', additional_PBR_conditioner=True,
        raw_surface_metrics_unchanged=True, geometry_array_byte_identity=dict.fromkeys(GEOMETRY_FIELDS, True),
        GT_sparse_latents_or_projection_cache_used=False, GT_evaluation_artifact_copied_only_after_prediction_saved=True,
        conditioning_preserves_CPU_and_CUDA_RNG=True, training_updates=0)
    if problem == 'geometry': actual['geometry_array_byte_identity']['vertices'] = False
    elif problem == 'partial-geometry-audit': actual['geometry_array_byte_identity'].pop('faces')
    elif problem == 'GT-support': actual['GT_sparse_latents_or_projection_cache_used'] = True
    elif problem == 'checkpoint': replay['PBR_checkpoint'] = '/wrong.pt'
    elif problem == 'conditioner': actual['conditioner_checkpoint'] = None
    elif problem == 'noise': actual['PBR_replay_seed'] = 907
    elif problem == 'source-SS': actual['SS_prediction_SHA256'] = 'other-model-grid'
    if problem is None: verify_texture_replay(replay, actual, original, model, seed=906)
    else:
        with pytest.raises(ValueError): verify_texture_replay(replay, actual, original, model, seed=906)
