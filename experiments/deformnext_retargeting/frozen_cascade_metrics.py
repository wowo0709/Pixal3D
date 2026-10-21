"""Collect eight metrics from exact native model compositions and shared pixels.

No sampling, rendering, feature extraction or training happens here. Final runs
require the same development-only model freeze as the FID/KID evaluator. An
explicit validation32 engineering mode can replay existing scores and intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .cascade_view_roles import (
    ClosedInputs, IMAGE_METRICS, require, summarize_view_roles,
)
from .evaluate_bound_population import sha
from .pbr_replay_artifacts import GEOMETRY_FIELDS
from .metric_image_identity import scored_image_observation
from .summarize_frozen_inception_pools import validate_frozen_models


class PinnedFiles:
    """Hash each immutable stage weight once; recheck bytes at completion."""
    def __init__(self):
        self.files = {}

    def check(self, spec):
        path = Path(spec['path']).resolve()
        before = path.stat()
        signature = (before.st_size, before.st_mtime_ns)
        if path not in self.files:
            digest = sha(path)
            require(signature == (path.stat().st_size, path.stat().st_mtime_ns), 'Pinned input changed while hashing')
            self.files[path] = (digest, signature)
        digest, saved = self.files[path]
        require(saved == signature and digest == spec['SHA256'], 'Frozen input bytes differ: '+str(path))
        return path

    def finish(self):
        for path, (digest, signature) in self.files.items():
            require(sha(path) == digest and signature == (path.stat().st_size, path.stat().st_mtime_ns),
                    'Frozen input changed during metric collection: '+str(path))


def verify_native_composition(native, generated, ss, model, *, sample_id, seed):
    """Verify resolved original stage paths, RGB protocol and actual SS identity."""
    ss_spec = model['ss64']
    require(native['row']['sample_id'] == sample_id == generated['sample_id'], 'Generation sample differs')
    require(native['seed'] == seed and generated['SS_source_evaluation'] == native['SS_evaluation'],
            'Prescribed native shape noise or SS source differs')
    require(ss['status'] == 'PASS' and ss['stage'] == 'ss64' and ss['weights'] == 'raw'
            and ss['step'] == ss_spec['step'] and ss['optimizer_updates_during_evaluation'] == 0,
            'Actual native SS checkpoint evaluation differs')
    for key in ['checkpoint', 'conditioner_checkpoint', 'arm_config']:
        expected = ss_spec[key]
        require(ss[key] == (expected['path'] if expected is not None else None), 'Actual SS '+key+' differs')
    mode = ss_spec['conditioning_mode']
    require(mode in {'source-only', 'edited-rgb'} and ss['conditioning_mode'] == mode
            and ss['target_modality'] == 'rgb' and ss['input_view_count'] == 4
            and ss['input_camera_ids'] == [8, 9, 10, 11], 'Actual four-view RGB SS input mode differs')
    expected_states = ['M0']*4 if mode == 'source-only' else ['M1', 'M0', 'M0', 'M0']
    require(native['input_states'] == expected_states and len(native['RGB_paths']) == 4,
            'Native downstream input must retain the declared four RGB views')
    require(set(native['stages']) == {'shape512', 'shape1024', 'pbr1024'}, 'Complete native downstream stages required')
    for stage in ['shape512', 'shape1024', 'pbr1024']:
        # An explicit later PBR replay replaces only the final texture weights.
        if stage == 'pbr1024' and model['texture_protocol'] == 'native-replay':
            continue
        for key in ['config', 'checkpoint']:
            require(native['stages'][stage][key] == model['stages'][stage][key]['path'],
                    'Actual native '+stage+' '+key+' differs')
    require(native['pipeline_config'] == model['pipeline_config']['path'], 'Native sampler configuration differs')
    entries = [entry for entry in ss['native_SS_artifacts']['artifacts'] if entry['sample_id'] == sample_id]
    require(len(entries) == 1 and generated['SS_prediction_SHA256'] == entries[0]['prediction_voxels_SHA256'],
            'Actual cascade did not consume this archived predicted SS grid')
    geometry = generated['metrics']
    require(generated['actual_resolution'] == 1024 and generated['source_defined_normalization']
            and not generated['per_output_alignment'] and not generated['GT_geometry_or_coords_used_in_generation']
            and generated['GT_loaded_only_after_generation']
            and geometry['threshold'] == .02 and geometry['prediction_samples'] == geometry['target_samples'] == 20000
            and geometry['per_output_alignment'] is False,
            'Raw generated surface normalization/GT access/20k F@.02 protocol differs')
    require(not generated['additional_PBR_conditioner']
            and generated['PBR_checkpoint'] == native['stages']['pbr1024']['checkpoint'],
            'Inline native PBR metadata differs')
    require(bool(generated.get('original_zero_shot_full_stack', False)) == model['original_zero_shot_stack'],
            'Original zero-shot and deformation-finetuned model roles differ')
    if model['original_zero_shot_stack']:
        require(generated['original_multiview_steps'] == dict(ss64=4210, shape512=4210, shape1024=16850, pbr1024=16850)
                and generated['deformation_finetuning_updates'] == 0 and ss_spec['step'] == 4210
                and ss_spec['conditioner_checkpoint'] is None, 'Original four-stage LoRA zero-shot lineage required')


def verify_texture_replay(replay, generated, original, model, *, seed):
    require(model['texture_protocol'] == 'native-replay', 'Unexpected extra PBR replay')
    require(replay['PBR_replay_seed'] == seed == generated['PBR_replay_seed']
            and replay['PBR_config'] == model['stages']['pbr1024']['config']['path']
            and replay['PBR_checkpoint'] == model['stages']['pbr1024']['checkpoint']['path']
            and generated['PBR_checkpoint'] == replay['PBR_checkpoint']
            and generated['PBR_checkpoint_SHA256'] == model['stages']['pbr1024']['checkpoint']['SHA256'],
            'Actual final PBR checkpoint/config/paired noise differs')
    condition = model['pbr_conditioner_checkpoint']
    expected = condition['path'] if condition is not None else None
    require(replay.get('conditioner_checkpoint') == expected == generated['conditioner_checkpoint']
            and generated['additional_PBR_conditioner'] == (condition is not None), 'Physical PBR conditioner differs')
    require(generated['status'] == 'PASS_ACTUAL_NATIVE_PBR_REPLAY_ON_GENERATED_SHAPE'
            and generated['sample_id'] == original['sample_id']
            and generated['SS_source_evaluation'] == original['SS_source_evaluation']
            and generated['SS_prediction_SHA256'] == original['SS_prediction_SHA256']
            and generated['metrics'] == original['metrics'] and generated['raw_surface_metrics_unchanged']
            and set(generated['geometry_array_byte_identity']) == set(GEOMETRY_FIELDS)
            and all(generated['geometry_array_byte_identity'].values())
            and not generated['GT_geometry_or_coords_used_in_generation']
            and not generated['GT_sparse_latents_or_projection_cache_used']
            and generated['GT_evaluation_artifact_copied_only_after_prediction_saved']
            and generated['conditioning_preserves_CPU_and_CUDA_RNG'] and generated['training_updates'] == 0,
            'Final texture replay must preserve actual generated geometry and SS, without GT support')


def read_terminal(terminal, native_path, binding, *, read, module):
    require(terminal['returncode'] == 0, 'Native case process did not complete successfully')
    command = terminal['command']
    require(command.count('-m') == command.count('--plan') == 1, 'Explicit native module/plan execution required')
    actual_module = command[command.index('-m')+1]
    actual_plan = command[command.index('--plan')+1]
    if binding is None:
        require(actual_module == module and Path(actual_plan) == Path(native_path), 'Executed native module/plan differs')
    else:
        expected = ('experiments.deformnext_retargeting.pbr_remote_results' if module.endswith('replay_pbr_on_generated_shape')
                    else 'experiments.deformnext_retargeting.cascade_remote_results')
        require(actual_module == expected and actual_plan == binding, 'Executed remote IO binding differs')
        io = read(binding)
        require(io['native_plan'] == native_path and io['native_plan_SHA256'] == sha(native_path),
                'Remote IO binding changed the actual native generation plan')


def verify_RGB_inputs(native, row, *, mode, file4_cache_root):
    def local(path):
        return str(Path(file4_cache_root)/Path(path).relative_to('/file4'))
    anchor = row['source_anchor_rgb'] if mode == 'source-only' else row['target_rgb_input']
    require(row['conditioning_camera_ids'] == [8, 9, 10, 11] and len(row['source_images']) == 3
            and native['RGB_paths'] == [local(path) for path in [anchor, *row['source_images']]],
            'Actual four RGB inputs must retain the original camera8 source anchor for source-only')
    require(native['source_geometry'] == local(row['source_geometry'])
            and native['target_vertices'] == local(row['target_vertices']), 'Actual scoring geometry differs from the prescribed sample')


def verify_shared_reference(render, *, sample_id):
    if render['status'] == 'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE':
        require(render['sample_id'] == sample_id and render['same_reference_pixels_for_every_model'],
                'Copied shared reference sample or pixels differ')
    else:
        # The original-material creator has no sample_id field. Its identity is
        # established by the checked native render plan and exact reused pixels.
        require(render['status'] == 'PASS_ACTUAL_SHARED_RENDERER_MATERIAL_REFERENCE'
                and render['shared_renderer_lighting_camera_AO_seed'] and render['GT_VAE_reconstruction_not_used']
                and render['prediction_renders_reused_not_regenerated'] and render['GT_not_used_for_model_generation']
                and render['maximum_GT_world_vertex_nearest_difference'] < 1e-5,
                'Native original-material M1 reference geometry/renderer differs')
        if 'sample_id' in render:
            require(render['sample_id'] == sample_id, 'Reference sample differs')


def prediction_pixel_identity(arrays, *, native):
    # These are two distinct, explicit existing archive contracts.
    rgb_key, alpha_key = ('shaded', 'alpha') if native else ('prediction', 'prediction_alpha')
    rgb, alpha = arrays[rgb_key], arrays[alpha_key]
    require(rgb.shape == (3, 512, 512) and alpha.shape == (512, 512)
            and rgb.dtype == alpha.dtype == np.float32, 'Native512 FP32 prediction RGB/alpha required')
    return tuple(hashlib.sha256(value.tobytes()).hexdigest() for value in (rgb, alpha))


def verify_test_SS_binding(ss, model, *, freeze_spec, manifest_spec, seed_map_SHA256):
    require('bound_population' in ss, 'Final SS must come from an explicitly frozen full-test evaluation')
    bound = ss['bound_population']
    require(bound['scope'] == 'final-test508' and bound['assets'] == 508 and bound['source_clusters'] == 134
            and bound['manifest'] == manifest_spec and bound['seeds']['SHA256'] == seed_map_SHA256
            and bound['model_freeze'] == freeze_spec and bound['model_binding'] == model['ss64']
            and ss['native_SS_artifacts']['archived_asset_count'] == 508,
            'Final SS predictions must bind the same pre-test model freeze, manifest and noise population')


def collect(plan):
    pins = PinnedFiles()
    if plan['scope'] == 'final-test508':
        binding = json.loads(pins.check(plan['model_freeze']).read_text())
        ids, clusters, contrasts = validate_frozen_models(binding, plan['models'])
        for evidence in binding['development_evidence']:
            pins.check(evidence)
    else:
        require(plan['scope'] == 'engineering-validation32' and 'model_freeze' not in plan,
                'Explicit validation-only engineering or frozen full-test scope required')
        binding = plan['engineering_binding']
        require(binding['status'] == 'ENGINEERING_VALIDATION32_REPLAY', 'Do not relabel development evidence as a final model freeze')
        ids, clusters, contrasts = (binding['population']['sample_ids'], binding['population']['source_clusters_by_sample'], binding['contrasts'])
        require(len(ids) == len(set(ids)) == 32 and set(clusters) == set(ids)
                and tuple(plan['models']) == tuple(binding['models']), 'Exact validation32 model population required')
    population = binding['population']
    manifest = [json.loads(x) for x in pins.check(plan['manifest']).read_text().splitlines() if x.strip()]
    expected_split = 'test' if plan['scope'] == 'final-test508' else 'validation'
    require(all(row['split'] == expected_split for row in manifest), 'Metric manifest split differs')
    manifest_rows = {row['sample_id']: row for row in manifest}
    require(len(manifest_rows) == len(manifest) and all(sid in manifest_rows and manifest_rows[sid]['source_cluster_id'] == clusters[sid] for sid in ids),
            'Actual metric manifest IDs or source clusters differ')
    if expected_split == 'test':
        require([row['sample_id'] for row in manifest] == ids and plan['manifest']['SHA256'] == population['manifest_SHA256'],
                'Every prescribed test508 asset must be retained in order')
        qa = json.loads(pins.check(plan['cache_QA']).read_text())
        require(qa['status'] == 'PASS_ACTUAL_TEST508_NATIVE_CACHE_AND_LOADER_QA' and qa['assets'] == 508
                and qa['SS_PBR_cache_ready'] and qa['no_test_asset_dropped']
                and qa['train_validation_identity_disjoint'] and qa['all_input_RGB_fingerprints_verified']
                and qa['published_manifest_SHA256'] == plan['manifest']['SHA256'], 'Actual full test508 native cache QA required')
    for model in binding['models'].values():
        require(model['texture_protocol'] in {'native-inline', 'native-replay'}, 'Explicit final texture protocol required')
        for spec in [model['pipeline_config'], model['ss64']['checkpoint'], model['ss64']['conditioner_checkpoint'], model['ss64']['arm_config'], model['pbr_conditioner_checkpoint']]:
            if spec is not None: pins.check(spec)
        for stage in model['stages'].values():
            for key in ['config', 'checkpoint']: pins.check(stage[key])
    pins.check(binding['render_protocol']['environment_map'])
    closed = ClosedInputs(plan['canonical_local_root'])
    perceptual, protocol = {}, None
    for path, receipt in plan['perceptuals'].items():
        closed.check(path, receipt); report = json.loads(Path(path).read_text())
        require(report['status'] == 'PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION'
                and report['full_panel'] and report['fullframe_primary']
                and report['GPU_sanity']['status'] == 'PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY', 'Incomplete official perceptual scores')
        actual = (report['metric_registry_SHA256'], report['model_weights'], report['model_precision'])
        if protocol is None: protocol = actual
        require(protocol == actual, 'Perceptual packages/weights/precision differ across models')
        for row in report['per_view']:
            key = (row['arm'], row['sample_id'], row['camera_id'])
            require(key not in perceptual, 'Duplicate perceptual observation')
            perceptual[key] = row
    references, records, expected_keys, compositions, scored_pixels = {}, {}, set(), {}, {}
    for alias, cases in plan['models'].items():
        require([r['sample_id'] for r in cases] == ids, 'Complete ordered frozen case population required: '+alias)
        model = binding['models'][alias]; records[alias] = []; compositions[alias] = []; scored_pixels[alias] = []
        for row in cases:
            sid = row['sample_id']; cluster = row['source_cluster_id']
            require(cluster == clusters[sid], 'Case source cluster differs')
            def read(path):
                require(path in row['closed_inputs'], 'Unreceipted native case input: '+path)
                closed.check(path, row['closed_inputs'][path]); return json.loads(Path(path).read_text())
            generated = read(row['generation']); native = read(row['native_generation_plan'])
            original = generated if model['texture_protocol'] == 'native-inline' else read(row['shape_generation'])
            terminal = read(row['generation_terminal'])
            module = ('experiments.deformnext_retargeting.cascade_zero_shot_from_ss_artifact' if model['original_zero_shot_stack']
                      else 'experiments.deformnext_retargeting.cascade_from_ss_artifact')
            read_terminal(terminal, row['native_generation_plan'], row.get('generation_IO_binding'), read=read, module=module)
            ss = read(native['SS_evaluation'])
            if plan['scope'] == 'final-test508':
                verify_test_SS_binding(ss, model, freeze_spec=plan['model_freeze'], manifest_spec=plan['manifest'],
                                       seed_map_SHA256=population['seed_map_SHA256'])
            verify_native_composition(native, original, ss, model, sample_id=sid, seed=population['cascade_seeds_by_sample'][sid])
            require(native['row']['conditioning_input_sha256'] == manifest_rows[sid]['conditioning_input_sha256'], 'Native generation RGB input identity differs')
            verify_RGB_inputs(native, manifest_rows[sid], mode=model['ss64']['conditioning_mode'], file4_cache_root=plan['file4_cache_root'])
            for path, digest in native['inputs_SHA256'].items():
                pins.check(dict(path=path, SHA256=digest))
            if model['texture_protocol'] == 'native-replay':
                replay = read(row['PBR_replay_plan'])
                require(replay['generated_shape_root'] == str(Path(row['shape_generation']).parent)
                        and generated['generated_shape_source'] == replay['generated_shape_root'],
                        'Final PBR must replay exactly this model\'s archived generated shape')
                read_terminal(read(row['PBR_replay_terminal']), row['PBR_replay_plan'], row.get('PBR_IO_binding'), read=read,
                              module='experiments.deformnext_retargeting.replay_pbr_on_generated_shape')
                verify_texture_replay(replay, generated, original, model, seed=population['texture_seeds_by_sample'][sid])
            camera_audit = read(row['render_camera_audit'])
            render_plan = read(row['native_render_plan'])
            read_terminal(read(row['render_terminal']), row['native_render_plan'], row.get('render_IO_binding'), read=read,
                          module='experiments.deformnext_retargeting.render_cascade_artifact_v2')
            require(render_plan['generation_root'] == str(Path(row['generation']).parent)
                    and render_plan['output'] == str(Path(row['render_camera_audit']).parent)
                    and render_plan['row']['sample_id'] == sid
                    and render_plan['environment_map'] == binding['render_protocol']['environment_map']['path']
                    and render_plan['AO_seed'] == binding['render_protocol']['AO_seed']
                    and render_plan['prediction_render_face_budget'] == 1000000,
                    'Scored native render must use this exact generated model and common lighting/face budget')
            geom = camera_audit['prediction_geometry_postprocess']
            require(camera_audit['sample_id'] == sid and camera_audit['prediction_is_generated_full_cascade']
                    and camera_audit['GT_shape_not_used_for_prediction'] and geom['same_budget_for_all_predictions']
                    and geom['raw_surface_metrics_unchanged'] and not geom['GT_geometry_used_for_simplification']
                    and not geom['per_output_alignment'] and geom['render_faces'] <= 1000000,
                    'Native shared1M-face render or raw geometric evaluation differs')
            render = read(row['render'])
            verify_shared_reference(render, sample_id=sid)
            views = {view['camera_id']: view for view in render['per_view']}
            require(len(render['per_view']) == len(views) == 3 and set(views) == {8, 9, 10}, 'Exactly three scored cameras required')
            values = {}
            for cid, view in views.items():
                key = (alias, sid, cid); expected_keys.add(key); p = perceptual[key]
                require(p['source_cluster_id'] == cluster, 'Perceptual source differs')
                pixels = Path(row['render']).parent/f'camera{cid:02d}.npz'
                digest = closed.check(pixels, pixels.parent/'file4-closure.json')
                require(p['shared_npz_SHA256'] == digest, 'Perceptual scores belong to different native pixels')
                audit = p['image_audit']
                with np.load(pixels, allow_pickle=False) as arrays:
                    require(arrays['prediction'].shape == arrays['GT_shared_renderer'].shape == (3, 512, 512)
                            and arrays['GT_alpha'].shape == (512, 512)
                            and all(arrays[name].dtype == np.float32 for name in ['prediction', 'GT_shared_renderer', 'GT_alpha']),
                            'Explicit native512 FP32 RGB/alpha schema required')
                    identity = tuple(hashlib.sha256(arrays[name].tobytes()).hexdigest() for name in ['prediction', 'GT_shared_renderer', 'GT_alpha'])
                    predicted_pixels = prediction_pixel_identity(arrays, native=False)
                    scored_pixels[alias].append(scored_image_observation(arrays, sample_id=sid, camera_id=cid,
                        source_cluster_id=cluster, archive=pixels, archive_SHA256=digest))
                native_pixels = Path(row['render_camera_audit']).parent/f'camera{cid:02d}.npz'
                closed.check(native_pixels, native_pixels.parent/'file4-closure.json')
                with np.load(native_pixels, allow_pickle=False) as arrays:
                    require(prediction_pixel_identity(arrays, native=True) == predicted_pixels,
                            'Shared-reference scores must reuse this native model\'s RGB and alpha exactly')
                require(identity == tuple(audit[name] for name in ['original_float_prediction_SHA256', 'original_float_GT_SHA256', 'original_GT_alpha_SHA256']),
                        'Actual prediction/GT/alpha array identity differs')
                reference = identity[1:] + (tuple(audit['GT_only_crop_box']), audit['converted_image_SHA256']['reference_full'], audit['converted_image_SHA256']['reference_GTcrop'])
                require(references.setdefault((sid, cid), reference) == reference, 'Cross-model GT reference/crop differs')
                values[cid] = {metric: view[metric] for metric in IMAGE_METRICS[:5]}
                scores = p['scores']
                values[cid].update(DreamSim=scores['DreamSim_fullframe'], CLIP=scores['CLIP_cosine_fullframe'],
                    DreamSim_GTcrop=scores['DreamSim_GT_foreground_crop'], CLIP_GTcrop=scores['CLIP_cosine_GT_foreground_crop'])
            records[alias].append(dict(sample_id=sid, source_cluster_id=cluster, views=values,
                CD=generated['metrics']['chamfer_L1'], FScore=generated['metrics']['surface_FScore']))
            compositions[alias].append(dict(sample_id=sid, native_plan=row['native_generation_plan'], SS_evaluation=native['SS_evaluation'],
                generation=row['generation'], texture_protocol=model['texture_protocol']))
    require(set(perceptual) == expected_keys, 'Unexpected or missing perceptual views')
    summary = summarize_view_roles(records, contrasts)
    pins.finish()
    summary.update(status='PASS_ACTUAL_BOUND_CASCADE_EIGHT_METRICS', scope=plan['scope'],
        actual_completed_cases=sum(len(rows) for rows in records.values()), actual_perceptual_views=len(perceptual),
        model_bindings=binding['models'], native_case_compositions=compositions, input_SHA256=closed.inputs,
        scored_image_observations=scored_pixels,
        model_freeze=plan.get('model_freeze'), manifest=plan['manifest'], native_stage_weight_hashes_verified=True,
        source_preservation_reference='Original M0 material identity on edited M1 geometry; source-view scores measure edit adaptation, not M0 geometry preservation',
        raw20k_surface_CD_and_F_at_point02_source_bbox_noICP=True, native512_common1M_face_render=True,
        source_cluster_CIs_condition_on_one_training_seed=True, intervals_uncorrected_for_multiple_contrasts=True,
        no_neural_model_forward_or_training=True, FID_KID_still_required=True, model_adopted=False, full_goal_complete=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve previous metric outputs')
    result = collect(json.loads(args.plan.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], scope=result['scope'], assets=result['assets'],
                          actual_completed_cases=result['actual_completed_cases'])), flush=True)


if __name__ == '__main__':
    main()
