"""Synthetic metadata-only full508 planning; no model or quality evidence."""
import copy
import json
from pathlib import Path

import pytest

from test_frozen_ss_population_plans import fixture as SS_fixture, put
from experiments.deformnext_retargeting import prepare_frozen_cascade_cases as factory
from experiments.deformnext_retargeting.bound_zero_shot_cascade import validate_bound_zero_shot


def fixture(tmp_path):
    ss_plan=SS_fixture(tmp_path)
    freeze=json.loads(Path(ss_plan['model_freeze']['path']).read_text())
    rows=[json.loads(z) for z in Path(ss_plan['manifest']['path']).read_text().splitlines()]
    cache=tmp_path/'cache';cache.mkdir()
    def file(canonical):
        path=cache/Path(canonical).relative_to('/file4');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'synthetic asset bytes, no native inference')
        return dict(path=str(path),SHA256=factory.sha(path),canonical_path=canonical)
    records=[]
    for i,row in enumerate(rows):
        identity=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,1.],[0.,0.,0.,1.]]
        row.update(conditioning_input_schema='deformnext-edited-rgb-input-v1',conditioning_input_sha256=f'{i:064x}',
            conditioning_camera_ids=[8,9,10,11],target_anchor_kind='rgb',
            target_anchor=f'/file4/{i}/target.png',target_rgb_input=f'/file4/{i}/target.png',
            source_anchor_rgb=f'/file4/{i}/source8.png',source_images=[f'/file4/{i}/source{c}.png' for c in [9,10,11]],
            evaluation_target_images=[f'/file4/{i}/GT{c}.png' for c in [9,10]],
            transform_matrix=[identity]*4,camera_angle_x=[.77]*4,source_anchor_transform_matrix=identity,
            source_anchor_camera_angle_x=.77,mesh_scale=.96,source_geometry='/file4/source.npz',target_vertices=f'/file4/{i}/target.npy')
        roles=dict(target_rgb=file(row['target_rgb_input']),source_anchor_rgb=file(row['source_anchor_rgb']),
                   source_geometry=file(row['source_geometry']),target_vertices=file(row['target_vertices']),pbr_dump=file(f'/file4/{i}/material.pickle'))
        roles.update({f'source_rgb_{c}':file(p) for c,p in zip([9,10,11],row['source_images'])})
        roles.update({f'GT_rgb_{c}':file(p) for c,p in zip([9,10],row['evaluation_target_images'])})
        core={k:row[k] for k in factory.CORE_FIELDS}
        records.append(dict(sample_id=row['sample_id'],source_cluster_id=row['source_cluster_id'],immutable_asset_fields=core,
            immutable_asset_fields_SHA256=factory.core_digest(core),inputs=roles,
            edited_RGB_paths=[roles['target_rgb']['path'],*[roles[f'source_rgb_{c}']['path'] for c in [9,10,11]]],
            source_only_RGB_paths=[roles['source_anchor_rgb']['path'],*[roles[f'source_rgb_{c}']['path'] for c in [9,10,11]]],
            render_views=[dict(camera_id=8,input_camera_index=0,GT_RGBA=roles['target_rgb']['path']),
                *[dict(camera_id=c,input_camera_index=c-8,GT_RGBA=roles[f'GT_rgb_{c}']['path']) for c in [9,10]]]))
    manifest=put(Path(ss_plan['manifest']['path']),rows,jsonl=True)
    freeze['population']['manifest_SHA256']=manifest['SHA256']
    ids=freeze['population']['sample_ids']
    freeze['population'].update(cascade_seeds_by_sample={sid:101+i for i,sid in enumerate(ids)},
                                texture_seeds_by_sample={sid:201+i for i,sid in enumerate(ids)})
    qa=json.loads(Path(ss_plan['cache_QA']['path']).read_text());qa['published_manifest_SHA256']=manifest['SHA256']
    qa_spec=put(Path(ss_plan['cache_QA']['path']),qa)
    qp=json.loads(Path(ss_plan['cache_QA_plan']['path']).read_text());qp['file4_cache']=str(cache)
    qp_spec=put(Path(ss_plan['cache_QA_plan']['path']),qp)
    asset_spec=put(tmp_path/'assets.json',dict(schema='native-final-cascade-asset-inputs-v1',assets=508,source_clusters=134,
                                            file4_cache_root=str(cache),records=records))
    env=put(tmp_path/'environment.exr','synthetic bytes, not lighting',raw=True)
    pipeline=put(tmp_path/'pipeline.json',{'args':{'tex_slat_sampler':{'params':{'steps':12}}}})
    src=Path(factory.__file__).with_name('cascade_from_ss_artifact.py')
    source={str(src):factory.sha(src)}
    render=dict(output='/old/render',generation_root='/old/generation',row={},views=[],environment_map=env['path'],AO_seed=0,
                source_SHA256=source,input_SHA256={env['path']:env['SHA256']},prediction_render_face_budget=1000000)
    shared=dict(output='/old/shared',generation_root='/old/generation',previous_render_root='/old/render',row={},views=[],
                pbr_dump='/old/material',environment_map=env['path'],AO_seed=0,source_SHA256=source)
    freeze['render_protocol']=dict(native_template=put(tmp_path/'render-template.json',render),
        shared_material_template=put(tmp_path/'shared-template.json',shared),environment_map=env,AO_seed=0)
    stage_specs={}
    for name,step in [('shape512',4210),('shape1024',16850),('pbr1024',16850)]:
        folder=tmp_path/name;folder.mkdir()
        stage_specs[name]=dict(config=put(folder/'config.json',{'synthetic_config':True}),
            checkpoint=put(folder/f'denoiser_step{step:07d}.pt','synthetic native weight bytes',raw=True))
    for alias,model in freeze['models'].items():
        zero=alias.startswith('zero-')
        recipe=Path(model['ss64']['arm_config']['path']);cfg=json.loads(recipe.read_text());cfg['validation']['dataset_args']['path_map']={'/file4':str(cache)}
        model['ss64']['arm_config']=put(recipe,cfg)
        model.update(stages=copy.deepcopy(stage_specs),pipeline_config=pipeline,pbr_conditioner_checkpoint=None,
                     original_zero_shot_stack=zero,texture_protocol='native-inline')
        generation=dict(status='SYNTHETIC_NATIVE_PLAN_METADATA_ONLY',output='/old/generation',row={},SS_evaluation='/old/SS.json',
            RGB_paths=['/old/target.png','/old/9.png','/old/10.png','/old/11.png'],input_states=['M1','M0','M0','M0'],
            source_geometry='/old/source.npz',target_vertices='/old/target.npy',pipeline_config=pipeline['path'],
            stages={name:{key:value['path'] for key,value in stage.items()} for name,stage in stage_specs.items()},
            DINO_local_path='/synthetic/DINO',decoders={},seed=42,surface_seed=101,allocator_fraction=.4,
            inputs_SHA256={pipeline['path']:pipeline['SHA256']},source_SHA256=source)
        for name,step in [('shape512',4210),('shape1024',16850),('pbr1024',16850)]:
            if zero:generation['stages'][name]['original_multiview_step']=step
        model['generation_template']=put(tmp_path/(alias+'-generation.json'),generation)
    freeze_spec=put(Path(ss_plan['model_freeze']['path']),freeze)
    evaluations={}
    for alias,model in freeze['models'].items():
        folder=tmp_path/('SS-'+alias);folder.mkdir();archives=[]
        for i,sid in enumerate(ids):
            path=folder/f'{i:03d}.npz';path.write_bytes(b'synthetic archive identity fixture, not an actual prediction')
            archives.append(dict(sample_id=sid,artifact=path.name,artifact_SHA256=factory.sha(path)))
        mode=model['ss64']['conditioning_mode']
        report=dict(status='PASS',stage='ss64',weights='raw',step=model['ss64']['step'],optimizer_updates_during_evaluation=0,
            sampling_steps=12,guidance=3.,generation_batch_size=8,target_modality='rgb',input_view_count=4,input_camera_ids=[8,9,10,11],
            conditioning_mode=mode,input_states=['M0']*4 if mode=='source-only' else ['M1','M0','M0','M0'],diagnostic_only=False,
            **{key:model['ss64'][key]['path'] if model['ss64'][key] else None for key in ['checkpoint','conditioner_checkpoint','arm_config']},
            bound_population=dict(scope='final-test508',assets=508,source_clusters=134,manifest=manifest,seeds=ss_plan['seeds'],
                model_freeze=freeze_spec,model_binding=model['ss64']),
            native_SS_artifacts=dict(status='PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_BOUND_POPULATION',archived_asset_count=508,artifacts=archives))
        evaluations[alias]=put(folder/'evaluation.json',report)
    return dict(scope='final-test508',model_freeze=freeze_spec,cache_QA=qa_spec,cache_QA_plan=qp_spec,
        manifest=manifest,asset_packet=asset_spec,SS_evaluations=evaluations,file4_cache_root=str(cache))


def test_all_models_all508_and_shared_reference_dependencies_are_preserved(tmp_path):
    plan=fixture(tmp_path);out=tmp_path/'prepared';result=factory.prepare(plan,out)
    assert result['total_cases']==2540 and len(result['models'])==5
    assert all(len(v)==508 for v in result['cases'].values())
    for alias,cases in result['cases'].items():
        descriptor=json.loads(Path(cases[0]['plan']['path']).read_text())
        assert [p['name'] for p in descriptor['phases']]==['generation','render','shared']
        native=json.loads(Path(descriptor['phases'][0]['native_plan']['path']).read_text())
        assert native['seed']==101 and native['surface_seed']==101 and len(native['RGB_paths'])==4
        if alias=='v1':assert descriptor['reference_dependency'] is None
        else:assert descriptor['reference_dependency'].endswith('/shared/file4-closure.json')
        if alias.startswith('zero-'):
            ss=json.loads(Path(plan['SS_evaluations'][alias]['path']).read_text())
            validated=validate_bound_zero_shot(native,ss)
            assert validated['conditioning_mode']==('source-only' if alias=='zero-source4' else 'edited-rgb')
            assert native['RGB_path_map'][native['row']['source_anchor_rgb']].endswith('source8.png')
    assert not (out/'models').exists()  # Planning never executes a candidate.


@pytest.mark.parametrize('problem',['omitted-model','missing-SS-asset','wrong-SS-step','RGB-order','row-frame','noise-bool'])
def test_subset_or_method_changes_fail_before_any_output(tmp_path,problem):
    plan=fixture(tmp_path)
    if problem=='omitted-model':plan['SS_evaluations'].pop('zero-edited')
    if problem in ['missing-SS-asset','wrong-SS-step']:
        path=Path(plan['SS_evaluations']['candidate']['path']);r=json.loads(path.read_text())
        if problem=='missing-SS-asset':r['native_SS_artifacts']['artifacts'].pop()
        else:r['step']=96
        plan['SS_evaluations']['candidate']=put(path,r)
    if problem in ['RGB-order','row-frame']:
        path=Path(plan['asset_packet']['path']);r=json.loads(path.read_text())
        if problem=='RGB-order':r['records'][0]['source_only_RGB_paths'].pop(0)
        else:r['records'][0]['immutable_asset_fields']['mesh_scale']=.5
        plan['asset_packet']=put(path,r)
    if problem=='noise-bool':
        path=Path(plan['model_freeze']['path']);r=json.loads(path.read_text());r['population']['cascade_seeds_by_sample']['test-sample-0']=True
        plan['model_freeze']=put(path,r)
    with pytest.raises(ValueError):factory.prepare(plan,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_scope_and_original_zero_shot_step_guards_do_not_relabel_development(tmp_path):
    with pytest.raises(ValueError,match='full-test508'):factory.prepare({'scope':'engineering-validation32'},tmp_path/'out')
    with pytest.raises(ValueError,match='full508'):validate_bound_zero_shot({}, {})


@pytest.mark.parametrize('arm',['vanilla','retargeting','direct'])
def test_generated_shape_PBR_replay_preserves_native_sampler_and_final_stage_bindings(tmp_path,arm):
    plan=fixture(tmp_path)
    freeze=json.loads(Path(plan['model_freeze']['path']).read_text());model=copy.deepcopy(freeze['models']['v1'])
    generation=json.loads(Path(model['generation_template']['path']).read_text())
    for stage in model['stages'].values():
        for value in stage.values():generation['inputs_SHA256'][value['path']]=value['SHA256']
    packet=json.loads(Path(plan['asset_packet']['path']).read_text())['records'][0]
    row=json.loads(Path(plan['manifest']['path']).read_text().splitlines()[0])
    render=json.loads(Path(freeze['render_protocol']['native_template']['path']).read_text())
    shared=json.loads(Path(freeze['render_protocol']['shared_material_template']['path']).read_text())
    final=put(tmp_path/'final-PBR.pt','synthetic final1905 weights',raw=True)
    model['stages']['pbr1024']['checkpoint']=final
    model.update(texture_protocol='native-replay',pbr_runtime_arm=arm,
        pbr_conditioner_checkpoint=None if arm=='vanilla' else {'path':'/frozen/conditioner.pt'},
        pbr_conditioner_config={'path':'/frozen/derived-test-data.json'},
        PBR_replay_source_SHA256=generation['source_SHA256'])
    replay=dict(arm='direct',output='/old/PBR',generated_shape_root='/old/shape',row={},input_states=[],RGB_paths=[],
        PBR_config='/old/config.json',PBR_checkpoint='/old/weights.pt',PBR_replay_seed=0,
        conditioner_checkpoint='/old/conditioner.pt',conditioner_config='/old/constructor.json',
        sampler_params={'steps':12},allocator_fraction=.4,DINO_local_path='/synthetic/DINO',decoders={})
    case=factory.native_case_plans(alias='candidate',model=model,row=row,asset=packet,SS_evaluation='/SS/evaluation.json',
        seed=101,texture_seed=201,output=tmp_path/'output',reference_root='/shared/GT',reference_render='/shared/render',
        is_reference=False,generation_template=generation,render_template=render,shared_template=shared,PBR_template=replay)
    assert [p['name'] for p in case['phases']]==['generation','PBR','render','shared']
    original,actual=case['phases'][0]['native'],case['phases'][1]['native']
    assert original['stages']['pbr1024']==generation['stages']['pbr1024']
    assert actual['PBR_checkpoint']==final['path'] and actual['PBR_replay_seed']==201
    assert actual['sampler_params']==replay['sampler_params'] and actual['allocator_fraction']==replay['allocator_fraction']
    assert case['phases'][2]['input_archive_roots']==[actual['output']]
    if arm=='vanilla':assert 'conditioner_checkpoint' not in actual and 'conditioner_config' not in actual
    else:assert actual['conditioner_config']=='/frozen/derived-test-data.json'
