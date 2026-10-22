"""Native six-model held-out cascade plans, without model or metric execution."""
import copy
from pathlib import Path

from .native_fullcascade_contract import PAIRS
from .original_zero_population import MODES


ALIASES = tuple(MODES) + tuple(PAIRS)
IMAGE_FIELDS = ('sample_id','camera_angle_x','transform_matrix','mesh_scale',
    'conditioning_input_schema','conditioning_input_sha256','target_anchor_kind',
    'conditioning_camera_ids','target_anchor','source_images','transport_cache','transport_cache_pbr')


def compose_models(parent, controls):
    if (parent.get('status')!='FROZEN_FROM_DEVELOPMENT_ONLY'
            or parent.get('heldout_quality_read_before_freeze')is not False
            or controls.get('status')!='FROZEN_ORIGINAL_ZERO_RGB_CONTROLS_BEFORE_TEST'
            or controls.get('heldout_model_quality_read')is not False
            or set(parent['models'])!=set(PAIRS) or set(controls['models'])!=set(MODES)):
        raise ValueError('actual four-model DEV-only freeze and two pre-test original RGB controls required')
    original=controls['downstream_original_stages']
    models={}
    for alias in MODES:
        models[alias]=dict(kind='original-zero',ss64=controls['models'][alias],
            geometry_stages=copy.deepcopy(original),pbr1024=original['pbr1024'])
    for alias,mode in PAIRS.items():
        model=parent['models'][alias]
        if (model['ss64']['variant']!=alias or model['ss64']['step']!=480
                or model['pbr1024']['mode']!=mode or model['pbr1024']['step']!=1900):
            raise ValueError('own native480/native1900 matched compositions required')
        models[alias]=dict(kind='native-trained',ss64=model['ss64'],pbr1024=model['pbr1024'],
            geometry_stages={**{k:copy.deepcopy(model[k])for k in ['shape512','shape1024']},
                             'pbr1024':copy.deepcopy(parent['original_PBR_for_SS_isolation'])})
    for model in models.values():
        for name,step in [('shape512',4210),('shape1024',16850),('pbr1024',16850)]:
            stage=model['geometry_stages'][name]
            if (stage['step']!=step or stage['checkpoint']['SHA256']!=original[name]['checkpoint']['SHA256']):
                raise ValueError('same user-specified original Shape and inline PBR prior required')
    return copy.deepcopy(models)


def case_plans(*, alias, model, row, asset, SS_evaluation, SS_evaluation_SHA256,
               SS_artifact, geometry_template, render_template, shared_template,
               output, reference_root, reference_render_root, cascade_seed,
               texture_seed, PBR_template=None, PBR_inputs=None, PBR_closed_inputs=None):
    if (alias not in ALIASES or row['split']!='test' or asset['sample_id']!=row['sample_id']
            or asset['source_cluster_id']!=row['source_cluster_id']
            or type(cascade_seed)is not int or not 0<=cascade_seed<2**31
            or type(texture_seed)is not int or not 0<=texture_seed<2**31
            or SS_artifact['sample_id']!=row['sample_id']):
        raise ValueError('one exact frozen test sample, archived SS and explicit paired noise required')
    root=Path(output);geometry=copy.deepcopy(geometry_template)
    old=geometry_template
    old_files=set(old['RGB_paths'])|{old['SS_evaluation'],old['source_geometry'],old['target_vertices']}
    old_files.update(old.get('RGB_path_map',{}).values())
    for stage in old['stages'].values():old_files.update([stage['config'],stage['checkpoint']])
    old_SS_root=Path(old['SS_evaluation']).parent
    fixed={p:d for p,d in old['inputs_SHA256'].items()
           if p not in old_files and not Path(p).is_relative_to(old_SS_root)}
    zero=model['kind']=='original-zero'
    if zero!=(alias in MODES):raise ValueError('original and trained case types cannot be interchanged')
    mode=MODES[alias]if zero else'edited-rgb'
    RGB=asset['source_only_RGB_paths']if mode=='source-only'else asset['edited_RGB_paths']
    if len(RGB)!=4 or len(set(RGB))!=4 or row['conditioning_camera_ids']!=[8,9,10,11]:
        raise ValueError('four distinct calibrated original/edited RGB inputs required')
    stages={name:dict(config=stage['config']['path'],checkpoint=stage['checkpoint']['path'],
                       original_multiview_step=stage['step'])for name,stage in model['geometry_stages'].items()}
    inputs={s['path']:s['SHA256']for s in asset['inputs'].values()
            if s['path']in set(RGB)|{asset['inputs']['source_geometry']['path'],asset['inputs']['target_vertices']['path']}}
    for stage in model['geometry_stages'].values():
        for k in ['config','checkpoint']:inputs[stage[k]['path']]=stage[k]['SHA256']
    inputs.update({SS_evaluation:SS_evaluation_SHA256,SS_artifact['path']:SS_artifact['SHA256']})
    geometry.update(output=str(root/'geometry'),row=copy.deepcopy(row),SS_evaluation=SS_evaluation,
        RGB_paths=RGB,input_states=['M0']*4 if mode=='source-only'else['M1','M0','M0','M0'],
        stages=stages,seed=cascade_seed,surface_seed=101,
        source_geometry=asset['inputs']['source_geometry']['path'],target_vertices=asset['inputs']['target_vertices']['path'],
        inputs_SHA256={**fixed,**inputs})
    if zero:
        geometry.update(original_control_alias=alias,conditioning_mode=mode,
            original_SS_checkpoint=model['ss64']['checkpoint']['path'],
            original_stage_checkpoints={k:v['checkpoint']for k,v in stages.items()},
            RGB_path_map={row['source_anchor_rgb']:asset['inputs']['source_anchor_rgb']['path'],
                row['target_anchor']:asset['inputs']['target_rgb']['path'],**dict(zip(row['source_images'],asset['edited_RGB_paths'][1:]))})
    else:
        for k in ['original_control_alias','conditioning_mode','original_SS_checkpoint','original_stage_checkpoints','RGB_path_map']:
            geometry.pop(k,None)
    phases=[dict(name='geometry',module='original_zero_cascade_population'if zero else'cascade_from_ss_artifact',
                 plan=geometry,kind='zero-bound-entry'if zero else'native',result='generation.json')]
    final=root/'geometry'
    if not zero:
        pbr=model['pbr1024']
        if (PBR_template is None or PBR_template['mode']!=PAIRS[alias]
                or PBR_template['PBR_checkpoint']!=pbr['checkpoint']['path']
                or PBR_template['PBR_config']!=pbr['native_config']['path']
                or PBR_template['conditioner_checkpoint']!=(None if pbr['conditioner_checkpoint']is None else pbr['conditioner_checkpoint']['path'])
                or PBR_template['checkpoint_kind']!='trained-native1900' or PBR_template['engineering_only']is not False):
            raise ValueError('actual own native1900 image-only replay template required')
        if not isinstance(PBR_inputs,dict)or not isinstance(PBR_closed_inputs,dict):
            raise ValueError('explicit resolved final-sample/model PBR input hashes and receipts required')
        for key in ['native_config','checkpoint','conditioner_checkpoint']:
            spec=pbr[key]
            if spec is not None and PBR_inputs.get(spec['path'])!=spec['SHA256']:
                raise ValueError('actual replay input hashes differ from frozen PBR weights/config')
        replay=copy.deepcopy(PBR_template)
        replay['input_SHA256']=copy.deepcopy(PBR_inputs)
        replay['closed_inputs']=copy.deepcopy(PBR_closed_inputs)
        replay.pop('cold_reference',None)
        replay.update(output=str(root/'PBR'),generated_shape_root=str(final),
            row={k:copy.deepcopy(row[k])for k in IMAGE_FIELDS},RGB_paths=RGB,
            input_states=['M1','M0','M0','M0'],PBR_replay_seed=texture_seed)
        # The factory pins actual resolved RGB/descriptor files. No GT geometry,
        # latent, projection or matching-label path is introduced into this row.
        phases.append(dict(name='PBR',module='replay_r_pbr_on_generated_shape',plan=replay,kind='native',result='generation.json'))
        final=root/'PBR'
    render=copy.deepcopy(render_template)
    render.update(generation_root=str(final),output=str(root/'render'),row=copy.deepcopy(row),
        views=copy.deepcopy(asset['render_views']),prediction_render_face_budget=1000000,
        input_SHA256={render['environment_map']:render_template['input_SHA256'][render['environment_map']],
            **{v['GT_RGBA']:next(s['SHA256']for s in asset['inputs'].values()if s['path']==v['GT_RGBA'])for v in asset['render_views']}})
    phases.append(dict(name='render',module='render_cascade_artifact_v2',plan=render,kind='native',result='render-evaluation.json'))
    if alias=='zero-source4':
        if str(reference_root)!=str(root/'shared')or str(reference_render_root)!=str(root/'render'):
            raise ValueError('predeclared source4 produces the same shared GT reference for every model')
        shared=copy.deepcopy(shared_template)
        shared.update(output=str(root/'shared'),generation_root=str(final),previous_render_root=str(root/'render'),
            row=copy.deepcopy(row),views=copy.deepcopy(asset['render_views']),pbr_dump=asset['inputs']['pbr_dump']['path'],
            input_SHA256={render['environment_map']:render_template['input_SHA256'][render['environment_map']],
                asset['inputs']['pbr_dump']['path']:asset['inputs']['pbr_dump']['SHA256']})
        module='score_cascade_shared_material'
    else:
        shared=dict(output=str(root/'shared'),sample_id=row['sample_id'],views=copy.deepcopy(asset['render_views']),
            prediction_render_root=str(root/'render'),fixed_GT_shared_root=str(reference_root),
            fixed_GT_render_root=str(reference_render_root),source_SHA256=copy.deepcopy(shared_template['source_SHA256']))
        module='score_cascade_fixed_reference'
    phases.append(dict(name='shared',module=module,plan=shared,kind='native',result='render-evaluation.json'))
    return dict(alias=alias,sample_id=row['sample_id'],source_cluster_id=row['source_cluster_id'],
        final_generation_root=str(final),phases=phases,
        reference_dependency=None if alias=='zero-source4'else str(Path(reference_root)/'file4-closure.json'),
        raw_archive_roots=[str(root/'geometry')]+([]if zero else[str(root/'PBR')]),
        actual_generation_or_metric_execution_performed=False)
