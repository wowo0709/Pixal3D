"""CPU metadata/control fixtures only; never real model freeze or test results."""
import json
from pathlib import Path
import shutil

import pytest

from experiments.deformnext_retargeting import source_input_frozen_campaign as adapter
from experiments.deformnext_retargeting import evaluate_bound_population as bound
from experiments.deformnext_retargeting import frozen_campaign as campaign
from experiments.deformnext_retargeting import frozen_campaign_stage as stage
from experiments.deformnext_retargeting.source_input_joint_runtime import TEACHER_INITIAL_FUNCTION
from test_frozen_ss_population_plans import fixture as old_fixture,put,mutate
from test_frozen_campaign import master_input


def fixture(tmp_path,plan=None):
    if plan is None:plan=old_fixture(tmp_path)
    freeze=json.loads(Path(plan['model_freeze']['path']).read_text());model=freeze['models']['candidate']['ss64']
    root=tmp_path/'unit-completed-joint';(root/'ckpts').mkdir(parents=True)
    cfg=json.loads(Path(model['arm_config']['path']).read_text());cfg['validation']['dataset_name']='DeformNeXtSS64'
    cfg.update(output_dir=str(root),experiment=dict(source_input_design='source-contrast-NoDQ',
        additional_source_teacher_supervision=True,initial_function=TEACHER_INITIAL_FUNCTION))
    recipe=put(root/'config.json',cfg)
    for key,name in [('checkpoint','denoiser_step0000480.pt'),('conditioner_checkpoint','deform_conditioner_step0000480.pt')]:
        destination=root/'ckpts'/name;shutil.copyfile(model[key]['path'],destination)
        model[key]=dict(path=str(destination),SHA256=bound.sha(destination))
    completion=dict(progress=put(root/'progress.json',dict(stage='completed',current=480,total=480)),
        startup=put(root/'startup.json',dict(status='PASS_ACTUAL_FRESH_MEMBER_FEATURE_SOURCE_RAY_SS_STARTUP',
            initial_function=TEACHER_INITIAL_FUNCTION,global_batch=96,microbatch=96,accumulation=1,world_size=1,
            trainable_NN_LoRA_tensors=360,trainable_source_tensors=19,optimizer_states=0)),
        runtime=put(root/'source-input-joint-runtime.json',dict(status='PASS_COMPLETED_NATIVE_SOURCE_INPUT_PHASE',
            phase='science',source_input_design='source-contrast-NoDQ',initialization=[dict(variant='source-contrast-NoDQ',
                source_tensors=19,additional_TRAIN_M1_source9_10_supervision=True)])))
    args=model['native_arguments']
    for flag in ['--phase','--conditioning-mode']:
        position=args.index(flag);del args[position:position+2]
    for flag,key in [('--checkpoint','checkpoint'),('--conditioner-checkpoint','conditioner_checkpoint')]:args[args.index(flag)+1]=model[key]['path']
    args[args.index('--arm-config')+1]=recipe['path']
    args+=['--completed-training-root',str(root),'--source-input-design','source-contrast-NoDQ']
    source=adapter.SELF.with_name(adapter.JOINT+'.py');package=adapter.SELF.parent;repo=package.parents[1]
    required=[adapter.SELF,source,package/'source_teacher_initialization.py',package/'source_input_joint_runtime.py',
        repo/'pixal3d/modules/source_contrast_member_feature.py',repo/'pixal3d/modules/source_referenced_member_feature.py',
        repo/'pixal3d/modules/member_feature_source_ray.py']
    model.update(evaluation_module=adapter.JOINT,evaluator_source=dict(path=str(source),SHA256=bound.sha(source)),
        arm_config=recipe,joint_training_completion=completion)
    model['source_SHA256'].update({str(p):bound.sha(p)for p in required})
    plan['model_freeze']=put(Path(plan['model_freeze']['path']),freeze)
    return plan


def test_prepares_all508_models_with_new_candidate_and_restores_original_allowlist(tmp_path):
    plan=fixture(tmp_path);old_modules=bound.MODULES
    result=adapter.prepare(plan,tmp_path/'unit-prepared')
    assert bound.MODULES is old_modules and adapter.JOINT not in bound.MODULES
    assert result['models']==['v1','retarget','candidate','zero-source4','zero-edited']
    candidate=result['evaluations']['model002'];p=json.loads(bound.pinned(candidate['native_bound_plan']).read_text())
    assert p['evaluation_module']==adapter.JOINT and p['assets']==508
    assert bound.argument(p['native_arguments'],'--expected-assets')=='508'
    assert bound.argument(p['native_arguments'],'--source-input-design')=='source-contrast-NoDQ'
    assert bound.argument(p['native_arguments'],'--completed-training-root').endswith('unit-completed-joint')
    assert not Path(p['output']).exists()


@pytest.mark.parametrize('damage',['missing-completion','completion-changed','missing-adapter-source','peeked','incomplete-QA','pilot-step'])
def test_cannot_bypass_original_or_new_freeze_guards(tmp_path,damage):
    plan=fixture(tmp_path)
    if damage=='peeked':mutate(plan,'model_freeze',lambda v:v.update(heldout_quality_read_before_freeze=True))
    elif damage=='incomplete-QA':mutate(plan,'cache_QA',lambda v:v.update(SS_PBR_cache_ready=False))
    else:
        def change(freeze):
            model=freeze['models']['candidate']['ss64']
            if damage=='missing-completion':model.pop('joint_training_completion')
            elif damage=='completion-changed':Path(model['joint_training_completion']['runtime']['path']).write_text('{}')
            elif damage=='missing-adapter-source':model['source_SHA256'].pop(str(adapter.SELF))
            else:model['step']=112
        mutate(plan,'model_freeze',change)
    old_modules=bound.MODULES;out=tmp_path/'unit-invalid'
    with pytest.raises(ValueError):adapter.prepare(plan,out)
    assert bound.MODULES is old_modules and not out.exists()


def test_population_allowlist_and_campaign_functions_restore_after_failure():
    old=bound.MODULES;functions=(campaign.load_master,campaign.module_command,campaign.gpu_job,stage.command_for,adapter.factory.prepare)
    with pytest.raises(RuntimeError,match='unit failure'):
        with adapter.campaign_binding(),adapter.population_binding():
            assert bound.MODULES==old|{adapter.JOINT}
            assert campaign.module_command is adapter.module_command
            raise RuntimeError('unit failure')
    assert bound.MODULES is old
    assert functions==(campaign.load_master,campaign.module_command,campaign.gpu_job,stage.command_for,adapter.factory.prepare)


def test_joint_stage_dispatches_pinned_adapter_but_legacy_and_metric_stages_keep_native_modules(tmp_path):
    native=put(tmp_path/'unit-native.json',dict(evaluation_module=adapter.JOINT))
    execution=dict(python='/unit-python',source_input_campaign_adapter=dict(path=str(adapter.SELF),SHA256=bound.sha(adapter.SELF)))
    plan=dict(kind='SS',native_plan=native,entry_source=execution['source_input_campaign_adapter'])
    cmd=adapter.stage_command(plan,execution)
    assert cmd==['/unit-python','-u','-m',campaign.PACKAGE+adapter.NAME,'SS-evaluate','--plan',native['path']]
    for kind,module in [('SS','evaluate_bound_population'),('perceptual','bound_official_perceptual'),('Inception','extract_official_inception')]:
        legacy=put(tmp_path/(kind+'-native.json'),dict(evaluation_module='member_feature_runtime'))
        source=adapter.SELF.with_name(module+'.py')
        p=dict(kind=kind,native_plan=legacy,entry_source=dict(path=str(source),SHA256=bound.sha(source)),output='/unit-output')
        assert adapter.stage_command(p,execution)==adapter._STAGE_COMMAND(p,execution)


def test_new_SS_job_and_deferred_controls_keep_source_root_and_adapter_entry(tmp_path):
    native=put(tmp_path/'unit-native.json',dict(evaluation_module=adapter.JOINT))
    initial=put(tmp_path/'unit-initial.json',dict(model_freeze={'path':'unit-freeze','SHA256':'unit'}))
    execution=dict(python='/unit-python',source_root=str(adapter.SELF.parents[2]),
        source_input_campaign_adapter=dict(path=str(adapter.SELF),SHA256=bound.sha(adapter.SELF)))
    master=dict(SS_plan=initial,GPU_priority=50,execution={'path':'unit-execution','SHA256':'unit'})
    job=adapter.gpu_job(master,execution,name='unit-source-SS',kind='SS',native=native,output='/unit-output',
        result_file='evaluation.json',result_status='PASS',folder=tmp_path,depends=['unit-SS-control'])
    spec=json.loads(Path(job['command'][job['command'].index('--plan')+1]).read_text())
    assert spec['entry_source']==execution['source_input_campaign_adapter']
    assert campaign.PACKAGE+adapter.NAME in job['command']and 'stage'in job['command']
    with adapter.campaign_binding():
        follow=campaign.module_command(execution,'frozen_campaign','advance','--plan','unit')
        plain=campaign.module_command(execution,'execute_frozen_cascade_case','--plan','unit')
    assert campaign.PACKAGE+adapter.NAME in follow
    assert campaign.PACKAGE+'execute_frozen_cascade_case'in plain


def test_all_six_deferred_phases_retain_joint_adapter_and_both_zero_controls(master_input,tmp_path,monkeypatch):
    from test_frozen_campaign import (
        test_all_deferred_campaign_phases_keep_the_same_freeze_and_finish_unit_DAG as exercise,
        close_unit,
    )
    spec,master,initial=master_input
    initial=fixture(tmp_path,initial)
    master['SS_plan']=put(Path(master['SS_plan']['path']),initial)
    execution=json.loads(bound.pinned(master['execution']).read_text())
    execution['source_input_campaign_adapter']=dict(path=str(adapter.SELF),SHA256=bound.sha(adapter.SELF))
    master['execution']=put(Path(master['execution']['path']),execution)
    master['source_SHA256'][str(adapter.SELF)]=bound.sha(adapter.SELF)
    spec=put(Path(spec['path']),master);close_unit(tmp_path,tmp_path)
    # Reuse the full six-phase controller harness. Only numerical producers and
    # canonical transport are mocked; real plan/freeze validation and dispatch
    # routing execute. This never installs anything into the actual queue.
    with adapter.campaign_binding():exercise((spec,master,initial),tmp_path,monkeypatch)
    queue=json.loads((tmp_path/'queue.json').read_text())
    controls=[j for j in queue if j['name'] in ['unit-'+s for s in ('SS','cases','perceptual','eight','distribution','table')]]
    assert len(controls)==6
    assert all(campaign.PACKAGE+adapter.NAME in j['command'] and 'advance' in j['command'] for j in controls)
    ss=[j for j in queue if j['resource']=='exclusive' and j['name'].startswith('unit-SS-')]
    assert len(ss)==5 and all(campaign.PACKAGE+adapter.NAME in j['command'] and 'stage' in j['command'] for j in ss)
    manifest=json.loads((Path(master['output'])/'SS/manifest.json').read_text())
    candidate=next(row for row in manifest['evaluations'].values() if row['alias']=='candidate')
    native=json.loads(bound.pinned(candidate['native_bound_plan']).read_text())
    assert native['evaluation_module']==adapter.JOINT and native['assets']==508
    assert len([j for j in queue if j['name'].startswith('unit-case-')])==2540
