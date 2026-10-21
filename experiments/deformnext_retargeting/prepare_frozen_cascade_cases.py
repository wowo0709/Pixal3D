"""Native generation/render plans for every frozen model and all508 assets.

No model or metric runs here. Completed SS archives, actual cache QA and a
development-only model freeze precede this step. Numerical native templates
are pinned in that freeze; only explicit sample, stage and output bindings move.
Large archive IO is resolved by the executor after each producer completes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from .bind_generated_pbr_dataset import load_QA, validate_binding
from .evaluate_bound_population import pinned, sha
from .frozen_cascade_metrics import PinnedFiles, verify_RGB_inputs, verify_test_SS_binding
from .summarize_frozen_inception_pools import validate_frozen_models
from .zero_shot_cascade_inputs import zero_shot_inputs


PACKAGE = 'experiments.deformnext_retargeting.'
CORE_FIELDS = ('sample_id','source_uid','source_cluster_id','conditioning_input_schema','conditioning_input_sha256',
    'target_anchor_kind','target_rgb_input','source_anchor_rgb','source_images','evaluation_target_images',
    'conditioning_camera_ids','camera_angle_x','transform_matrix','mesh_scale','source_geometry','target_vertices')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def core_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sample_binding(row, packet, *, file4_cache_root):
    if set(packet['immutable_asset_fields'])!=set(CORE_FIELDS) or row['target_anchor']!=row['target_rgb_input']:
        raise ValueError('complete immutable RGB/camera/world-frame asset fields required')
    core = {key:row[key] for key in packet['immutable_asset_fields']}
    if (core != packet['immutable_asset_fields']
            or core_digest(core) != packet['immutable_asset_fields_SHA256']
            or row['sample_id'] != packet['sample_id'] or row['source_cluster_id'] != packet['source_cluster_id']):
        raise ValueError('actual published sample differs from its sealed native asset inputs')
    expected = [dict(camera_id=8,input_camera_index=0,GT_RGBA=packet['inputs']['target_rgb']['path'])]
    expected += [dict(camera_id=cid,input_camera_index=cid-8,GT_RGBA=packet['inputs'][f'GT_rgb_{cid}']['path']) for cid in (9,10)]
    if packet['render_views'] != expected:
        raise ValueError('exact target8 and source9/10 scoring cameras required')
    for mode, key in [('edited-rgb','edited_RGB_paths'),('source-only','source_only_RGB_paths')]:
        verify_RGB_inputs(dict(RGB_paths=packet[key],source_geometry=packet['inputs']['source_geometry']['path'],
            target_vertices=packet['inputs']['target_vertices']['path']),row,mode=mode,file4_cache_root=file4_cache_root)
    paths={row['target_anchor']:packet['inputs']['target_rgb']['path'],
           row['source_anchor_rgb']:packet['inputs']['source_anchor_rgb']['path'],
           **dict(zip(row['source_images'],packet['edited_RGB_paths'][1:]))}
    zero_shot_inputs(row,'source-only',paths)
    if packet['inputs']['pbr_dump'].get('canonical_path') is None:
        raise ValueError('explicit original material dump from actual inventory required')


def native_case_plans(*, alias, model, row, asset, SS_evaluation, seed, texture_seed,
                      output, reference_root, reference_render, is_reference,
                      generation_template, render_template, shared_template,
                      PBR_template=None, PBR_input_SHA256=None):
    """Pure, explicit native plan construction; no filesystem/model execution."""
    root=Path(output)
    generation=copy.deepcopy(generation_template)
    original_inputs=set(generation['RGB_paths'])|{generation['SS_evaluation'],generation['source_geometry'],generation['target_vertices']}
    original_inputs.update(generation.get('RGB_path_map',{}).values())
    for stage in generation['stages'].values():
        original_inputs.update(stage[key] for key in ['config','checkpoint'])
    fixed={p:d for p,d in generation['inputs_SHA256'].items() if p not in original_inputs}
    stages={name:{**copy.deepcopy(generation_template['stages'][name]),
                  **{key:spec[key]['path'] for key in ['config','checkpoint']}} for name,spec in model['stages'].items()}
    # Replay preserves the inline template's PBR pass and later replaces texture
    # on that exact generated shape. Its initial PBR choice is itself frozen by
    # the generation-template digest, not guessed from a checkpoint filename.
    if model['texture_protocol']=='native-replay':
        stages['pbr1024']=copy.deepcopy(generation_template['stages']['pbr1024'])
    mode=model['ss64']['conditioning_mode']
    paths=asset['source_only_RGB_paths'] if mode=='source-only' else asset['edited_RGB_paths']
    inputs={spec['path']:spec['SHA256'] for spec in asset['inputs'].values()
            if spec['path'] in set(paths)|{asset['inputs']['source_geometry']['path'],asset['inputs']['target_vertices']['path']}}
    for name,stage in stages.items():
        for key in ['config','checkpoint']:
            path=stage[key]
            if name=='pbr1024' and model['texture_protocol']=='native-replay':
                inputs[path]=generation_template['inputs_SHA256'][path]
            else:
                inputs[path]=model['stages'][name][key]['SHA256']
    inputs[model['pipeline_config']['path']]=model['pipeline_config']['SHA256']
    generation.update(output=str(root/'generation'),row=row,SS_evaluation=SS_evaluation,
        input_states=['M0']*4 if mode=='source-only' else ['M1','M0','M0','M0'],RGB_paths=paths,
        stages=stages,pipeline_config=model['pipeline_config']['path'],seed=seed,
        source_geometry=asset['inputs']['source_geometry']['path'],target_vertices=asset['inputs']['target_vertices']['path'],
        inputs_SHA256={**fixed,**inputs})
    if model['original_zero_shot_stack']:
        generation.update(conditioning_mode=mode,
            RGB_path_map={row['source_anchor_rgb']:asset['inputs']['source_anchor_rgb']['path'],
                          row['target_anchor']:asset['inputs']['target_rgb']['path'],
                          **dict(zip(row['source_images'],asset['edited_RGB_paths'][1:]))},
            original_SS_checkpoint=model['ss64']['checkpoint']['path'],
            original_stage_checkpoints={name:stage['checkpoint'] for name,stage in stages.items()})
    elif mode!='edited-rgb':
        raise ValueError('source-only is an original four-stage zero-shot control')
    phases=[dict(name='generation',module=PACKAGE+('cascade_zero_shot_from_ss_artifact' if model['original_zero_shot_stack']
        else 'cascade_from_ss_artifact'),remote_stage='zero-generation' if model['original_zero_shot_stack'] else 'generation',
        input_archive_roots=[],native=generation,result='generation.json')]
    final_root=root/'generation'
    if model['texture_protocol']=='native-replay':
        if PBR_template is None:
            raise ValueError('frozen native PBR replay template required')
        replay=copy.deepcopy(PBR_template)
        replay.update(arm=model['pbr_runtime_arm'],output=str(root/'PBR'),generated_shape_root=str(final_root),row=row,
            input_states=generation['input_states'],RGB_paths=paths,PBR_replay_seed=texture_seed,
            PBR_config=model['stages']['pbr1024']['config']['path'],PBR_checkpoint=model['stages']['pbr1024']['checkpoint']['path'],
            pipeline_config=model['pipeline_config']['path'])
        replay['source_SHA256']=model['PBR_replay_source_SHA256']
        replay['input_SHA256']=dict(PBR_input_SHA256 or {})
        if model['pbr_conditioner_checkpoint'] is not None:
            replay.update(conditioner_checkpoint=model['pbr_conditioner_checkpoint']['path'],
                          conditioner_config=model['pbr_conditioner_config']['path'])
        else:
            replay.pop('conditioner_checkpoint',None)
            replay.pop('conditioner_config',None)
        phases.append(dict(name='PBR',module=PACKAGE+'replay_pbr_on_generated_shape',remote_stage='PBR-replay',
            input_archive_roots=[str(final_root)],native=replay,result='generation.json'))
        final_root=root/'PBR'
    render=copy.deepcopy(render_template)
    render.update(generation_root=str(final_root),output=str(root/'render'),row=row,views=asset['render_views'],
        prediction_render_face_budget=1000000,
        input_SHA256={render['environment_map']:render_template['input_SHA256'][render['environment_map']],
            **{view['GT_RGBA']:next(spec['SHA256'] for spec in asset['inputs'].values() if spec['path']==view['GT_RGBA'])
               for view in asset['render_views']}})
    phases.append(dict(name='render',module=PACKAGE+'render_cascade_artifact_v2',remote_stage='render',
        input_archive_roots=[str(final_root)],native=render,result='render-evaluation.json'))
    if is_reference:
        shared=copy.deepcopy(shared_template)
        shared.update(output=str(reference_root),generation_root=str(final_root),previous_render_root=render['output'],
            row=row,views=asset['render_views'],pbr_dump=asset['inputs']['pbr_dump']['path'],
            input_SHA256={render['environment_map']:render_template['input_SHA256'][render['environment_map']],
                          asset['inputs']['pbr_dump']['path']:asset['inputs']['pbr_dump']['SHA256']})
        phases.append(dict(name='shared',module=PACKAGE+'score_cascade_shared_material',remote_stage='shared-material',
            input_archive_roots=[str(final_root)],native=shared,result='render-evaluation.json'))
    else:
        shared=dict(output=str(root/'shared'),sample_id=row['sample_id'],views=asset['render_views'],
            prediction_render_root=render['output'],fixed_GT_shared_root=str(reference_root),
            fixed_GT_render_root=str(reference_render),source_SHA256=shared_template['source_SHA256'])
        phases.append(dict(name='shared',module=PACKAGE+'score_cascade_fixed_reference',remote_stage=None,
            input_archive_roots=[],native=shared,result='render-evaluation.json'))
    return dict(alias=alias,sample_id=row['sample_id'],source_cluster_id=row['source_cluster_id'],phases=phases,
        reference_dependency=None if is_reference else str(Path(reference_root)/'file4-closure.json'))


def prepare(plan, output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('preserve existing native case plans')
    required={'model_freeze','cache_QA','cache_QA_plan','manifest','asset_packet','SS_evaluations'}
    if plan.get('scope')!='final-test508' or not required.issubset(plan):
        raise ValueError('frozen full-test508 models, cache QA and completed SS evaluations required')
    pins=PinnedFiles()
    freeze=json.loads(pins.check(plan['model_freeze']).read_text())
    ids,clusters,_=validate_frozen_models(freeze,freeze['models'])
    for name in ['cascade_seeds_by_sample','texture_seeds_by_sample']:
        seeds=freeze['population'][name]
        if (set(seeds)!=set(ids) or any(not isinstance(v,int) or isinstance(v,bool) or not 0<=v<2**31 for v in seeds.values())):
            raise ValueError('complete frozen integer native generation noise maps required')
    if tuple(plan['SS_evaluations'])!=tuple(freeze['models']):
        raise ValueError('completed SS evaluations for every frozen composition in order required')
    for evidence in freeze['development_evidence']:pins.check(evidence)
    load_QA(plan)
    rows=[json.loads(line) for line in pins.check(plan['manifest']).read_text().splitlines() if line.strip()]
    if [r['sample_id'] for r in rows]!=ids or plan['manifest']['SHA256']!=freeze['population']['manifest_SHA256']:
        raise ValueError('exact published full-test manifest and order required')
    packet=json.loads(pins.check(plan['asset_packet']).read_text())
    if (packet['schema']!='native-final-cascade-asset-inputs-v1' or packet['assets']!=508 or packet['source_clusters']!=134
            or [r['sample_id'] for r in packet['records']]!=ids
            or packet['file4_cache_root']!=plan['file4_cache_root']):
        raise ValueError('actual ordered all508 native asset packet required')
    for row,asset in zip(rows,packet['records']):
        sample_binding(row,asset,file4_cache_root=plan['file4_cache_root'])
        if row['source_cluster_id']!=clusters[row['sample_id']]:raise ValueError('frozen source cluster differs')
        for spec in asset['inputs'].values():pins.check(spec)
    render_protocol=freeze['render_protocol']
    render=json.loads(pins.check(render_protocol['native_template']).read_text())
    shared=json.loads(pins.check(render_protocol['shared_material_template']).read_text())
    env=str(pins.check(render_protocol['environment_map']))
    if any(t['environment_map']!=env or t['AO_seed']!=render_protocol['AO_seed'] for t in [render,shared]):
        raise ValueError('one frozen native lighting/AO protocol required')
    for template in [render,shared]:
        for path,digest in template['source_SHA256'].items():pins.check(dict(path=path,SHA256=digest))
    case_map={};reference_model=list(freeze['models']).index(freeze['comparison_roles']['vanilla'])
    for mi,(alias,model) in enumerate(freeze['models'].items()):
        generation=json.loads(pins.check(model['generation_template']).read_text())
        PBR=None
        if (set(model['stages'])!={'shape512','shape1024','pbr1024'}
                or generation['pipeline_config']!=model['pipeline_config']['path']
                or generation['surface_seed']!=101):
            raise ValueError('complete frozen original pipeline and source-world surface protocol required')
        for path,digest in generation['source_SHA256'].items():pins.check(dict(path=path,SHA256=digest))
        for stage in model['stages'].values():
            for key in ['config','checkpoint']:pins.check(stage[key])
        pins.check(model['pipeline_config'])
        for name,stage in generation['stages'].items():
            for key in ['config','checkpoint']:
                path=stage[key]
                if name!='pbr1024' or model['texture_protocol']=='native-inline':
                    if path!=model['stages'][name][key]['path']:raise ValueError('template stage differs from frozen composition')
        if model['texture_protocol']=='native-replay':
            if model['original_zero_shot_stack']:raise ValueError('original zero-shot retains the original inline PBR stage')
            PBR=json.loads(pins.check(model['PBR_replay_template']).read_text())
            if not model.get('PBR_replay_source_SHA256'):
                raise ValueError('actual development PBR replay source bindings required')
            for path,digest in model['PBR_replay_source_SHA256'].items():pins.check(dict(path=path,SHA256=digest))
            for path,digest in PBR.get('source_SHA256',{}).items():pins.check(dict(path=path,SHA256=digest))
            if PBR['sampler_params']!=json.loads(pinned(model['pipeline_config']).read_text())['args']['tex_slat_sampler']['params']:
                raise ValueError('unchanged native PBR sampling parameters required')
            if (PBR['DINO_local_path']!=generation['DINO_local_path']
                    or any(generation['decoders'].get(name)!=prefix for name,prefix in PBR['decoders'].items())):
                raise ValueError('PBR replay must retain the same pinned native feature backbone and decoders')
            if model['pbr_conditioner_checkpoint'] is not None:
                pins.check(model['pbr_conditioner_checkpoint'])
                validate_binding(model['pbr_dataset_binding'],conditioner_config=model['pbr_conditioner_config'],
                    cache_QA=plan['cache_QA'],cache_QA_plan=plan['cache_QA_plan'],manifest=plan['manifest'])
        elif model['texture_protocol']!='native-inline' or model['pbr_conditioner_checkpoint'] is not None:
            raise ValueError('explicit complete native inline or generated-shape PBR replay required')
        ss_path=pins.check(plan['SS_evaluations'][alias])
        ss=json.loads(ss_path.read_text())
        verify_test_SS_binding(ss,model,freeze_spec=plan['model_freeze'],manifest_spec=plan['manifest'],
            seed_map_SHA256=freeze['population']['seed_map_SHA256'])
        archived=ss['native_SS_artifacts']['artifacts']
        expected=model['ss64']
        if (ss['status']!='PASS' or ss['stage']!='ss64' or ss['weights']!='raw' or ss['step']!=expected['step']
                or ss['optimizer_updates_during_evaluation']!=0 or ss['sampling_steps']!=12
                or ss['guidance']!=3. or ss['generation_batch_size']!=8 or ss['target_modality']!='rgb'
                or ss['input_view_count']!=4 or ss['input_camera_ids']!=[8,9,10,11]
                or ss['conditioning_mode']!=expected['conditioning_mode']
                or any(ss[key]!=(expected[key]['path'] if expected[key] is not None else None)
                       for key in ['checkpoint','conditioner_checkpoint','arm_config'])
                or ss['native_SS_artifacts']['status']!='PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_BOUND_POPULATION'
                or [r['sample_id'] for r in archived]!=ids):raise ValueError('all actual508 archived predicted SS outputs required')
        if len({r['artifact'] for r in archived})!=508:raise ValueError('one distinct SS archive per prescribed sample required')
        for key in ['checkpoint','conditioner_checkpoint','arm_config']:
            if expected[key] is not None:pins.check(expected[key])
        for artifact in archived:
            path=Path(artifact['artifact'])
            if path.is_absolute() or '..' in path.parts:raise ValueError('explicit relative native SS artifact path required')
            pins.check(dict(path=str(ss_path.parent/path),SHA256=artifact['artifact_SHA256']))
        cases=[]
        for row,asset in zip(rows,packet['records']):
            sid=row['sample_id'];key=hashlib.sha256(sid.encode()).hexdigest()[:16]
            ref=output/'models'/f'model{reference_model:03d}'/key
            PBR_inputs={}
            if PBR is not None and model['pbr_conditioner_checkpoint'] is not None:
                cfg=json.loads(pinned(model['pbr_conditioner_config']).read_text())['dataset']['args']
                def local(value):
                    return cfg['path_map']['/file4']+value[len('/file4'):] if value.startswith('/file4/') else value
                paths=[local(row[key]) for key in ['transport_cache','transport_cache_pbr']]
                paths.append(local(str(Path(cfg['match_root'])/(sid+'.npz'))))
                PBR_inputs={path:sha(path) for path in paths}
            case=native_case_plans(alias=alias,model=model,row=row,asset=asset,SS_evaluation=str(ss_path),
                seed=freeze['population']['cascade_seeds_by_sample'][sid],
                texture_seed=freeze['population']['texture_seeds_by_sample'][sid],
                output=output/'models'/f'model{mi:03d}'/key,reference_root=ref/'shared',reference_render=ref/'render',
                is_reference=alias==freeze['comparison_roles']['vanilla'],generation_template=generation,
                render_template=render,shared_template=shared,PBR_template=PBR,PBR_input_SHA256=PBR_inputs)
            if model['original_zero_shot_stack']:
                case['phases'][0]['native']['bound_zero_shot']=dict(model_freeze=plan['model_freeze'],
                    manifest=plan['manifest'],model_alias=alias)
            for phase in case['phases']:
                for path,digest in {**phase['native'].get('inputs_SHA256',{}),**phase['native'].get('input_SHA256',{}),
                                    **phase['native'].get('source_SHA256',{})}.items():
                    pins.check(dict(path=path,SHA256=digest))
            cases.append(case)
        case_map[alias]=cases
    pins.finish()
    output.mkdir(parents=True,exist_ok=False)
    index={}
    for mi,(alias,cases) in enumerate(case_map.items()):
        index[alias]=[]
        for ci,case in enumerate(cases):
            folder=output/'plans'/f'model{mi:03d}'/f'case{ci:03d}';folder.mkdir(parents=True)
            phases=[]
            for phase in case['phases']:
                path=folder/(phase['name']+'.json');dump(path,phase['native'])
                native_source=Path(__file__).with_name(phase['module'].rsplit('.',1)[1]+'.py')
                if phase['remote_stage']=='PBR-replay':execution_module=PACKAGE+'pbr_remote_results'
                elif phase['remote_stage']=='zero-generation':execution_module=PACKAGE+'bound_zero_shot_cascade'
                elif phase['remote_stage'] is not None:execution_module=PACKAGE+'cascade_remote_results'
                else:execution_module=phase['module']
                execution_source=Path(__file__).with_name(execution_module.rsplit('.',1)[1]+'.py')
                phases.append({**{k:v for k,v in phase.items() if k!='native'},
                    'native_plan':dict(path=str(path),SHA256=sha(path)),
                    'native_source':dict(path=str(native_source),SHA256=sha(native_source)),
                    'execution_module':execution_module,
                    'execution_source':dict(path=str(execution_source),SHA256=sha(execution_source))})
            descriptor={**{k:v for k,v in case.items() if k!='phases'},'phases':phases,
                        'model_freeze':plan['model_freeze'],'source_inputs':plan}
            path=folder/'case.json';dump(path,descriptor)
            index[alias].append(dict(sample_id=case['sample_id'],plan=dict(path=str(path),SHA256=sha(path))))
    result=dict(status='PREPARED_ALL_FROZEN_TEST508_NATIVE_CASCADE_CASES',scope='final-test508',source_inputs=plan,
        model_freeze=plan['model_freeze'],models=list(freeze['models']),cases=index,assets=508,source_clusters=134,
        total_cases=508*len(freeze['models']),native512_scored_cameras=[8,9,10],
        native_templates_and_stage_weights_preserved=True,one_original_material_reference_per_sample=True,
        actual_gpu_generation_rendering_and_queue_installation_not_performed=True,full_goal_complete=False)
    dump(output/'manifest.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=prepare(json.loads(args.plan.read_text()),args.output)
    print(json.dumps(dict(status=result['status'],models=result['models'],total_cases=result['total_cases'])),flush=True)


if __name__=='__main__':main()
