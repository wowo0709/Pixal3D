"""Per-sample frozen selected-pipeline plans with original captured guides.

No model, array or quality metric is executed here. Numerical templates and
the actual selected model bindings come from the closed DEV freeze.
"""
from copy import deepcopy
from pathlib import Path

from .archived_native_guides import POLICY
from .assigned_gpu_entry import validate_assignment
from .generated_pbr_conditioning import resolve_path
from .native_final_case_integrity import paired_seed
from .native_final_case_plans import IMAGE_FIELDS
from .native_final_freeze import validate_spec
from .original_zero_population import MODES
from .prepare_frozen_cascade_cases import sample_binding
from .selected_final_population import compose_models
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


def required_sources():
    root = Path(__file__).resolve().parent
    return [str(root/(name+'.py')) for name in ['selected_final_case_plans','selected_final_population',
        'selected_original_zero_cascade_population','selected_pipeline_model_contract','selected_pipeline_freeze_v2',
        'native_guide_entry','archived_native_guides','native_guide_capture_context','cascade_from_ss_artifact','cascade_zero_shot_from_ss_artifact',
        'replay_r_pbr_on_generated_shape','generated_r_mediated_pbr','assigned_gpu_entry','common_pipeline_render_entry',
        'render_cascade_artifact_v2','score_cascade_shared_material','score_cascade_fixed_reference','zero_shot_cascade_inputs']]


def _sources(template, frozen):
    require(all(frozen.get(path) == value for path,value in template.get('source_SHA256',{}).items()),
            'every original numerical template source must retain its frozen hash')
    return deepcopy(frozen)


def _retained(mapping, old_files, old_SS_root):
    return {p:v for p,v in mapping.items() if p not in old_files and not Path(p).is_relative_to(old_SS_root)}


def case_plans(*, parent, controls, alias, row, asset, file4_cache_root, assignment, SS_evaluation,
               SS_artifact, geometry_template, geometry_template_spec, render_template,
               shared_template, output, reference_root, reference_render_root,
               PBR_template=None, PBR_config=None, PBR_inputs=None, PBR_closed_inputs=None,
               geometry_reference=None):
    models = compose_models(parent, controls)
    frozen_sources = parent['evaluation_protocol']['source_SHA256']
    require(set(required_sources()) <= set(frozen_sources), 'freeze every selected and original numerical entry source')
    require(alias in models, 'one actual frozen composition or original zero control required')
    model = models[alias]; zero = alias in MODES; sid = row['sample_id']; pop = parent['population']
    devices = assignment.get('assigned_GPUs')
    require(isinstance(devices,list) and bool(devices), 'explicit assigned physical devices required')
    validate_assignment(assignment, devices[0], ','.join(devices))
    require(Path(assignment['canonical_local_root']).is_absolute(), 'explicit canonical results root required')
    require(row['split'] == 'test' and sid in pop['sample_ids']
        and row['source_cluster_id'] == pop['source_clusters_by_sample'][sid], 'own frozen test sample/source required')
    sample_binding(row, asset, file4_cache_root=file4_cache_root)
    require(exact(geometry_template_spec, parent['evaluation_protocol']['generation_template']),
            'the actual frozen original numerical geometry template is required')
    for spec in [SS_evaluation, geometry_template_spec]: validate_spec(spec)
    require(SS_artifact['sample_id'] == sid and SS_artifact['seed'] == pop['generation_seeds'][sid],
            'same actual own frozen SS sample/noise required')
    validate_spec(dict(path=SS_artifact['path'], SHA256=SS_artifact['SHA256']))
    root = Path(output)
    require(root.is_absolute() and Path(reference_root).is_absolute() and Path(reference_render_root).is_absolute(),
            'explicit absolute result and shared reference paths required')
    require(all(Path(p).resolve().is_relative_to(Path(assignment['canonical_local_root']).resolve())
        for p in [output,reference_root,reference_render_root]), 'results and references must stay in the assigned canonical root')
    geometry = deepcopy(geometry_template); old = geometry_template
    old_files = set(old['RGB_paths']) | {old['SS_evaluation'],old['source_geometry'],old['target_vertices']}
    old_files.update(old.get('RGB_path_map', {}).values())
    for stage in old['stages'].values(): old_files.update([stage['config'],stage['checkpoint']])
    old_SS_root = Path(old['SS_evaluation']).parent
    for field in ['inputs_SHA256','input_SHA256','closed_inputs']:
        if field in geometry: geometry[field] = _retained(geometry[field], old_files, old_SS_root)
    mode = MODES[alias] if zero else 'edited-rgb'
    RGB = deepcopy(asset['source_only_RGB_paths'] if mode == 'source-only' else asset['edited_RGB_paths'])
    stages = {name:dict(config=s['config']['path'],checkpoint=s['checkpoint']['path'],original_multiview_step=s['step'])
              for name,s in model['geometry_stages'].items()}
    files = {s['path']:s['SHA256'] for s in asset['inputs'].values()}
    required = set(RGB) | {asset['inputs']['source_geometry']['path'],asset['inputs']['target_vertices']['path']}
    inputs = {p:files[p] for p in required}
    for stage in model['geometry_stages'].values():
        for key in ['config','checkpoint']: inputs[stage[key]['path']] = stage[key]['SHA256']
    inputs.update({SS_evaluation['path']:SS_evaluation['SHA256'],SS_artifact['path']:SS_artifact['SHA256']})
    geometry.update(output=str(root/'geometry'),row=deepcopy(row),SS_evaluation=SS_evaluation['path'],
        RGB_paths=RGB,input_states=['M0']*4 if mode == 'source-only' else ['M1','M0','M0','M0'],
        stages=stages,seed=paired_seed(sid,'cascade'),surface_seed=101,
        source_geometry=asset['inputs']['source_geometry']['path'],target_vertices=asset['inputs']['target_vertices']['path'],
        inputs_SHA256={**geometry['inputs_SHA256'],**inputs},source_SHA256=_sources(geometry_template,frozen_sources))
    if zero:
        geometry.update(original_control_alias=alias,conditioning_mode=mode,
            original_SS_checkpoint=model['ss64']['checkpoint']['path'],
            original_stage_checkpoints={k:v['checkpoint'] for k,v in stages.items()},
            RGB_path_map={row['source_anchor_rgb']:asset['inputs']['source_anchor_rgb']['path'],
                row['target_anchor']:asset['inputs']['target_rgb']['path'],**dict(zip(row['source_images'],RGB[1:]))})
        first = dict(name='geometry',module='selected_original_zero_cascade_population',kind='selected-zero-bound-entry',
            arguments=[],plan=geometry,result='generation.json')
    else:
        require(geometry.get('texture_support_policy') == POLICY, 'capture the original guide logits/order/scale required')
        for key in ['original_control_alias','conditioning_mode','original_SS_checkpoint','original_stage_checkpoints','RGB_path_map']:
            geometry.pop(key,None)
        first = dict(name='geometry',module='native_guide_entry',kind='native-guide-capture',arguments=['--kind','capture'],
            plan=geometry,result='generation.json')
    canonical_alias = next(name for name,m in models.items() if m['geometry_group'] == model['geometry_group'])
    require((geometry_reference is None) == (alias == canonical_alias),
            'generate one actual geometry per physical SS model and reuse it for further PBR compositions')
    final = root/'geometry'
    if geometry_reference is not None:
        require(not zero and geometry_reference['alias'] == canonical_alias
            and geometry_reference['sample_id'] == sid
            and exact(geometry_reference['SS_evaluation'], SS_evaluation)
            and geometry_reference['SS_prediction_voxels_SHA256'] == SS_artifact['prediction_voxels_SHA256']
            and Path(geometry_reference['generation_root']).is_absolute()
            and Path(geometry_reference['generation_root']).resolve().is_relative_to(
                Path(assignment['canonical_local_root']).resolve()),
            'reuse only the same actual archived SS prediction and frozen original Shape geometry')
        final = Path(geometry_reference['generation_root'])
        first = dict(name='geometry',module=None,kind='reuse-closed-native-guides',arguments=[],
            reference=deepcopy(geometry_reference),plan=geometry,result='generation.json')
    geometry_source = final; phases = [first]
    if not zero:
        pbr = model['pbr1024']; template = PBR_template; config = PBR_config
        require(isinstance(PBR_inputs,dict) and isinstance(PBR_closed_inputs,dict), 'explicit actual replay hashes/receipts required')
        require(isinstance(template,dict) and isinstance(config,dict)
            and template['mode'] == model['PBR_mode'] and template['checkpoint_kind'] == 'trained-native1900'
            and template['engineering_only'] is False and template['input_states'] == ['M1','M0','M0','M0']
            and template['texture_support_policy'] == POLICY and config['training_seed'] == 101
            and config['experiment']['PBR_source_mode'] == model['PBR_mode']
            and str(Path(config['output_dir'])/'config.json') == pbr['native_config']['path'],
            'own actual primary native1900 source/config/template required')
        for key, field in [('native_config','PBR_config'),('checkpoint','PBR_checkpoint'),('conditioner_checkpoint','conditioner_checkpoint')]:
            spec = pbr[key]
            require(template[field] == (None if spec is None else spec['path']), 'actual selected PBR template belongs to another model')
            if spec is not None:
                require(PBR_inputs.get(spec['path']) == spec['SHA256'], 'actual selected PBR bytes must be pinned')
        require(all(PBR_inputs.get(p) == files[p] for p in RGB), 'all four actual RGB files must be pinned for replay')
        if model['PBR_mode'] != 'vanilla':
            for field in ['transport_cache','transport_cache_pbr']:
                path = str(resolve_path(row[field], config['dataset']['args']['path_map']))
                require(path in PBR_inputs and path in PBR_closed_inputs, 'actual own input-image descriptor bytes/receipt required')
        replay = deepcopy(template)
        replay.pop('cold_reference',None)
        replay.update(input_SHA256=deepcopy(PBR_inputs),closed_inputs=deepcopy(PBR_closed_inputs),
            output=str(root/'PBR'),generated_shape_root=str(final),native_guides=str(final/'native-shape-guides.pt'),
            row={k:deepcopy(row[k]) for k in IMAGE_FIELDS},RGB_paths=RGB,
            input_states=['M1','M0','M0','M0'],PBR_replay_seed=paired_seed(sid,'texture'),
            source_SHA256=_sources(template,frozen_sources))
        for filename in ['generation.json','prediction.npz','geometry-evaluation.npz','native-shape-guides.pt']:
            replay['closed_inputs'][str(final/filename)] = str(final/'file4-closure.json')
        phases.append(dict(name='PBR',module='native_guide_entry',kind='native-guide-replay',arguments=['--kind','replay'],
            plan=replay,result='generation.json')); final = root/'PBR'
    render = deepcopy(render_template)
    render.update(generation_root=str(final),output=str(root/'render'),row=deepcopy(row),views=deepcopy(asset['render_views']),
        prediction_render_face_budget=1000000,input_SHA256={render['environment_map']:render_template['input_SHA256'][render['environment_map']],
            **{v['GT_RGBA']:files[v['GT_RGBA']] for v in asset['render_views']}},
        source_SHA256=_sources(render_template,frozen_sources))
    phases.append(dict(name='render',module='render_cascade_artifact_v2',kind='assigned-native-render',arguments=[],plan=render,result='render-evaluation.json'))
    if alias == 'zero-source4':
        require(str(reference_root) == str(root/'shared') and str(reference_render_root) == str(root/'render'),
                'original source4 must produce the same shared material reference for every model')
        shared = deepcopy(shared_template)
        shared.update(output=str(root/'shared'),generation_root=str(final),previous_render_root=str(root/'render'),
            row=deepcopy(row),views=deepcopy(asset['render_views']),pbr_dump=asset['inputs']['pbr_dump']['path'],
            input_SHA256={render['environment_map']:render_template['input_SHA256'][render['environment_map']],
                asset['inputs']['pbr_dump']['path']:asset['inputs']['pbr_dump']['SHA256']},
            source_SHA256=_sources(shared_template,frozen_sources))
        module = 'score_cascade_shared_material'
    else:
        shared = dict(output=str(root/'shared'),sample_id=sid,views=deepcopy(asset['render_views']),
            prediction_render_root=str(root/'render'),fixed_GT_shared_root=str(reference_root),
            fixed_GT_render_root=str(reference_render_root),source_SHA256=_sources(shared_template,frozen_sources))
        module = 'score_cascade_fixed_reference'
    phases.append(dict(name='shared',module=module,kind='assigned-native-score',arguments=[],plan=shared,result='render-evaluation.json'))
    for phase in phases:
        phase['plan'].update(assigned_GPUs=deepcopy(devices),canonical_local_root=assignment['canonical_local_root'])
    return dict(schema='selected-physical-pipeline-TEST508-sample-plan-v1',alias=alias,model=deepcopy(model),
        sample_id=sid,source_cluster_id=row['source_cluster_id'],geometry_group=model['geometry_group'],
        SS_evaluation=SS_evaluation,geometry_template=geometry_template_spec,phases=phases,
        final_generation_root=str(final),reference_dependency=None if alias == 'zero-source4'
            else str(Path(reference_root)/'file4-closure.json'),
        geometry_source_root=str(geometry_source),geometry_reference=deepcopy(geometry_reference),
        raw_archive_roots=[str(geometry_source)]+([] if zero else [str(root/'PBR')]),
        actual_generation_or_metric_execution_performed=False)
