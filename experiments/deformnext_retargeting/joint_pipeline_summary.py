"""Compare actual SS/PBR compositions, without inferring them from stage tests.

Every texture, including vanilla, uses the same seed-reset replay protocol.
Texture noise bytes are required to match only for identical geometric support.
"""
from copy import deepcopy
import json
from pathlib import Path

from .archived_native_guides import POLICY
from .cascade_view_roles import IMAGE_METRICS, summarize_view_roles
from .common_geometry_pbr_contract import CASE_STATUS, MODES, validate_replay
from .common_pipeline_contract import CASE_STATUS as BASE_STATUS
from .generated_cascade_comparison import validate_perceptual_reports
from .module_pipeline_protocol_v2 import protocol
from .native_final_freeze import validate_spec
from .unified_module_summary import ALIASES, SOURCE_COUNTS
from .unified_population_contract import SCHEMA, family


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compositions(selected=None):
    pairs = dict(V1=('V1', 'vanilla'), RETARGET=('C_DECODER', 'top1-only'))
    if selected is None:
        return pairs, None
    require(isinstance(selected, tuple) and len(selected) == 2
        and selected[0] in protocol()['arms'] and selected[1] in MODES,
        'one measured physical SS arm and native PBR mode required')
    for name, pair in pairs.items():
        if selected == pair:
            return pairs, name
    pairs['SELECTED'] = selected
    return pairs, 'SELECTED'


def validate_binding(ss, pbr, pair):
    arm, mode = pair
    require(ss.get('schema') == SCHEMA and ss.get('module_arm') == arm
        and ss.get('variant') == ALIASES.get(arm, arm) and ss.get('runtime_family') == family(arm)
        and type(ss.get('step')) is int and ss['step'] == 480
        and type(ss.get('training_seed')) is int and ss['training_seed'] == 101
        and type(ss.get('source_tensors')) is int and ss['source_tensors'] == SOURCE_COUNTS[arm]
        and ss.get('conditioning_mode') == 'edited-rgb'
        and ss.get('source_teacher_updates') == 0 and ss.get('native_task_objective') == 'SS_flow_mse',
        'actual own native480 physical SS binding required')
    require(pbr.get('schema') == 'native-cold-PBR1900-frozen-binding-v1'
        and pbr.get('mode') == mode and type(pbr.get('step')) is int and pbr['step'] == 1900
        and type(pbr.get('training_seed')) is int and pbr['training_seed'] == 101
        and type(pbr.get('source_tensors')) is int
        and pbr['source_tensors'] == (0 if mode == 'vanilla' else 10 if mode == 'top1-only' else 19),
        'actual own native1900 physical PBR binding required')
    for binding, absent, step in ((ss, arm == 'V1', 480), (pbr, mode == 'vanilla', 1900)):
        for key in ('native_config', 'checkpoint'):
            validate_spec(binding[key])
        root = Path(binding['native_config']['path']).parent
        require(Path(binding['checkpoint']['path']) == root/'ckpts'/f'denoiser_step{step:07d}.pt'
            and (binding['conditioner_checkpoint'] is None) == absent,
            'same actual model root and physical source presence required')
        if not absent:
            validate_spec(binding['conditioner_checkpoint'])
            require(Path(binding['conditioner_checkpoint']['path']) == root/'ckpts'/f'deform_conditioner_step{step:07d}.pt',
                    'own NN/source pair required')


def summarize(bindings, cases, generations, bases, shared, official, *, selected,
              sample_ids, clusters, archives, original_shapes, pixel_SHA256, pixel_GT,
              metric_registry_SHA256, replay_seed=20261020, draws=4000):
    """`bases` records the actual saved SS/Shape case, generation and bound plan."""
    pairs, selected_alias = compositions(selected)
    require(len(sample_ids) == len(set(sample_ids)) == 32 and set(clusters) == set(sample_ids)
        and all(set(values) == set(pairs) for values in (bindings, cases, generations, shared))
        and type(replay_seed) is int and 0 <= replay_seed < 2**32,
        'deduplicated actual compositions on the same unique32 development assets required')
    require(set(original_shapes) == {'shape512', 'shape1024'}, 'original two Shape stages required')
    for stage, step in [('shape512', 4210), ('shape1024', 16850)]:
        spec = original_shapes[stage]
        require(type(spec['step']) is int and spec['step'] == step, 'original Shape step mismatch')
        for key in ('config', 'checkpoint'):
            validate_spec(spec[key])
        require(spec['checkpoint']['path'].endswith(f'denoiser_step{step:07d}.pt'), 'original Shape checkpoint name mismatch')
    require(official['metric_registry_SHA256'] == metric_registry_SHA256
        and official['model_precision'] == 'float32,TF32off,noAMP'
        and official['GPU_sanity']['CPU_model_forward_or_training_updates'] == 0,
        'actual same pinned official FP32 GPU metric protocol required')
    scores = validate_perceptual_reports([official])
    expected = {(a, s, c) for a in pairs for s in sample_ids for c in (8, 9, 10)}
    require(set(scores) == set(pixel_SHA256) == set(pixel_GT) == expected,
            'complete actual predicted pixels and official scores required')
    references, support_noise, sampler, records = {}, {}, None, {}
    for alias, pair in pairs.items():
        ss, pbr = bindings[alias]['ss64'], bindings[alias]['pbr1024']
        validate_binding(ss, pbr, pair)
        arm, mode = pair
        require(all(set(values[alias]) == set(sample_ids) for values in (cases, generations, shared)),
                'every actual composition case must be complete')
        archive = archives[arm]
        validate_spec(archive['spec'])
        ev = archive['record']
        require(ev.get('status') == 'PASS' and ev.get('step') == 480 and ev.get('weights') == 'raw'
            and ev.get('native_variant') == ALIASES.get(arm, arm)
            and ev.get('optimizer_updates_during_evaluation') == 0,
            'actual raw480 own SS archive required')
        for key, field in [('native_config', 'arm_config'), ('checkpoint', 'checkpoint'),
                           ('conditioner_checkpoint', 'conditioner_checkpoint')]:
            spec = ss[key]
            require(ev[field] == (None if spec is None else spec['path']), 'SS archive belongs to another model')
            if spec is not None:
                require(ev['verified_inputs'][spec['path']] == spec['SHA256'], 'SS archive model bytes differ')
        artifacts = {r['sample_id']: r for r in ev['native_SS_artifacts']['artifacts']}
        require(set(sample_ids) <= set(artifacts), 'actual own predicted SS occupancy archives required')
        records[alias] = []
        for sid in sample_ids:
            case, g, r = cases[alias][sid], generations[alias][sid], shared[alias][sid]
            base = bases[arm][sid]
            bc, bg, bp = base['case'], base['generation'], base['plan']
            for key in ('case_spec', 'plan_spec', 'prediction_spec'):
                validate_spec(base[key])
            require(bc['status'] == BASE_STATUS and bc['SS_arm'] == arm and bc['sample_id'] == sid
                and bc['source_cluster_id'] == clusters[sid]
                and bc['new_SS_neural_forwards'] == 0 and bc['new_Shape_cascade_calls'] == 1
                and bc['new_PBR_texture_calls'] == 1
                and len(bc['actual_owned_terminals']) == 3
                and all(t['returncode'] == 0 and t['actual_session_drained'] for t in bc['actual_owned_terminals'])
                and bc['SS_prediction_SHA256'] == artifacts[sid]['prediction_voxels_SHA256']
                and bc['verified_inputs'][archive['spec']['path']] == archive['spec']['SHA256']
                and bc['verified_inputs'][base['plan_spec']['path']] == base['plan_spec']['SHA256']
                and bg['SS_source_evaluation'] == archive['spec']['path']
                and bg['SS_prediction_SHA256'] == bc['SS_prediction_SHA256']
                and bg['GT_loaded_only_after_generation'] is True
                and bg['source_defined_normalization'] is True
                and bg['original_stage_configs_and_weights_unchanged'] is True
                and bg['metrics']['threshold'] == .02 and bg['metrics']['per_output_alignment'] is False,
                'actual selected SS prediction through the original unaligned Shape cascade required')
            for stage in original_shapes:
                for key in ('config', 'checkpoint'):
                    spec = original_shapes[stage][key]
                    require(bp['stages'][stage][key] == spec['path']
                        and bp['inputs_SHA256'][spec['path']] == spec['SHA256'],
                        'geometry was generated by a different original Shape stage')
            require(case['status'] == CASE_STATUS and case['mode'] == mode and case['sample_id'] == sid
                and case['source_cluster_id'] == clusters[sid]
                and case['common_case'] == str(Path(base['case_spec']['path']).parent)
                and case['common_prediction_SHA256'] == base['prediction_spec']['SHA256']
                and case['native_guide_SHA256'] == bg['actual_native_guide_archive_SHA256']
                and case['PBR_checkpoint_SHA256'] == pbr['checkpoint']['SHA256']
                and case['actual_four_rank_stage_completion_checked'] is True
                and case['new_SS_forwards'] == case['new_Shape_forwards'] == 0
                and case['new_PBR_sampling_calls'] == 1
                and len(case['actual_owned_terminals']) == 3
                and all(t['returncode'] == 0 and t['actual_session_drained'] for t in case['actual_owned_terminals']),
                'actual own1900 replay/render/shared case on this saved geometry required')
            for key in ('native_config', 'checkpoint', 'conditioner_checkpoint'):
                spec = pbr[key]
                if spec is not None:
                    require(case['verified_inputs'][spec['path']] == spec['SHA256'], 'PBR replay used other model bytes')
            validate_replay(g, bg, mode=mode, model_SHA256=pbr['checkpoint']['SHA256'],
                            prediction_SHA256=base['prediction_spec']['SHA256'])
            require(g['PBR_replay_seed'] == replay_seed and g['PBR_checkpoint'] == pbr['checkpoint']['path'],
                    'all textures, including V1, require the same seed-reset protocol')
            own_sampler = json.dumps([g['sampler_class'], g['sampler_params']], sort_keys=True)
            require(sampler is None or sampler == own_sampler, 'PBR sampler differs between compositions')
            sampler = own_sampler
            identity = (sid, g['original_prediction_SHA256'], g['geometry_evaluation_SHA256'],
                        g['actual_native_guide_archive_SHA256'])
            require(support_noise.setdefault(identity, g['actual_initial_texture_noise_SHA256'])
                == g['actual_initial_texture_noise_SHA256'], 'texture noise differs for identical geometric support')
            require(r['status'] == 'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE' and r['sample_id'] == sid
                and r['same_reference_pixels_for_every_model'] is True and len(r['per_view']) == 3,
                'same original material M1 references required')
            views = {}
            for view in r['per_view']:
                cid = view['camera_id']
                require(cid in (8, 9, 10) and cid not in views, 'unique target8/source9/10 cameras required')
                key = alias, sid, cid
                score = scores[key]
                audit = score['image_audit']
                gt = view['fixed_GT_RGB_SHA256'], view['fixed_GT_alpha_SHA256']
                require(score['source_cluster_id'] == clusters[sid] and score['shared_npz_SHA256'] == pixel_SHA256[key]
                    and gt == pixel_GT[key] == (audit['original_float_GT_SHA256'], audit['original_GT_alpha_SHA256'])
                    and view['identical_camera_matrices'] is True and view['source_material_identity_preserved'] is True,
                    'actual pixels/cameras/GT/official score identities differ')
                ref = gt + (tuple(audit['GT_only_crop_box']), audit['converted_image_SHA256']['reference_full'],
                            audit['converted_image_SHA256']['reference_GTcrop'])
                require(references.setdefault((sid, cid), ref) == ref, 'reference RGB/mask/crop differs across models')
                views[cid] = {k: view[k] for k in IMAGE_METRICS[:5]}
                v = score['scores']
                views[cid].update(DreamSim=v['DreamSim_fullframe'], CLIP=v['CLIP_cosine_fullframe'],
                    DreamSim_GTcrop=v['DreamSim_GT_foreground_crop'], CLIP_GTcrop=v['CLIP_cosine_GT_foreground_crop'])
            records[alias].append(dict(sample_id=sid, source_cluster_id=clusters[sid], views=views,
                CD=g['metrics']['chamfer_L1'], FScore=g['metrics']['surface_FScore']))
    contrasts = [['RETARGET', 'V1']]
    if 'SELECTED' in pairs:
        contrasts += [['SELECTED', 'V1'], ['SELECTED', 'RETARGET']]
    result = summarize_view_roles(records, contrasts, draws=draws, seed=20261023)
    result.update(status='PASS_ACTUAL_DEV_PIPELINE_COMPOSITIONS32_EIGHT_METRICS',
        measured_compositions=deepcopy(bindings), original_shapes=deepcopy(original_shapes),
        prospective_selected_alias=selected_alias, actual_official_views=32*3*len(pairs),
        actual_joint_SS_and_PBR_compositions_measured=True,
        stage_isolation_not_substituted_for_joint_measurement=True,
        texture_support_policy=POLICY, PBR_replay_seed=replay_seed,
        identical_geometry_has_identical_actual_texture_noise=True,
        different_geometry_texture_noise_byte_identity_not_claimed=True,
        original_cascade_texture_not_mixed_with_seed_reset_replays=True,
        reference='Original material on edited M1 geometry, not undeformed M0 RGB',
        exploratory_DEV32_intervals_not_final_TEST_confirmation=True,
        model_adopted=False, FID_KID_and_TEST508_still_required=True, full_goal_complete=False)
    return result
