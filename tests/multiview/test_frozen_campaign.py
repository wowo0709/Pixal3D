"""CPU control tests only: no campaign is installed into the actual queue."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting import frozen_campaign as campaign
from experiments.deformnext_retargeting import frozen_campaign_stage as gpu
from experiments.deformnext_retargeting.evaluate_bound_population import sha
from test_frozen_cascade_metric_plans import put, close_unit


def test_bootstrap_imports_assigned_source_even_from_other_dispatcher_checkout(tmp_path):
    source=tmp_path/'assigned';foreign=tmp_path/'dispatcher';foreign.mkdir()
    for root,marker in [(source,'assigned'),(foreign,'wrong-checkout')]:
        package=root/'experiments'/'deformnext_retargeting';package.mkdir(parents=True)
        (root/'experiments'/'__init__.py').write_text('')
        (package/'__init__.py').write_text('')
        (package/'probe.py').write_text('import json,os,sys\nprint(json.dumps(dict(source='+repr(marker)+',args=sys.argv[1:],path=os.environ["PYTHONPATH"])))\n')
    command=campaign.module_command({'python':sys.executable,'source_root':str(source)},'probe','--plan','unit')
    actual=json.loads(subprocess.check_output(command,cwd=foreign,text=True))
    assert actual==dict(source='assigned',args=['--plan','unit'],path=str(source))


@pytest.mark.parametrize('problem',[None,'concurrent','duplicate','dependency'])
def test_queue_append_preserves_existing_prefix_and_never_edits_runtime_state(tmp_path,monkeypatch,problem):
    queue=tmp_path/'queue.json';old=[dict(name='old-running',command=['unchanged-native'],resource='exclusive')]
    put(queue,old);folder=tmp_path/'control';folder.mkdir();events=[]
    jobs=[dict(name='new-frozen',resource='exclusive',depends_on=['old-running'])]
    if problem=='duplicate':jobs[0]['name']='old-running'
    elif problem=='dependency':jobs[0]['depends_on']=['undefined']
    def close(path,**kwargs):
        events.append(path)
        if problem=='concurrent':put(queue,old+[dict(name='other-writer')])
    monkeypatch.setattr(campaign,'close',close)
    if problem is None:
        campaign.append_jobs(queue,jobs,folder=folder,execution={})
        assert json.loads(queue.read_text())==old+jobs and events==[folder]
    else:
        with pytest.raises(ValueError):campaign.append_jobs(queue,jobs,folder=folder,execution={})
        assert json.loads(queue.read_text())==old+([dict(name='other-writer')] if problem=='concurrent' else [])
        if problem=='concurrent':assert queue.with_suffix('.frozen-campaign.tmp').exists()
    assert not (tmp_path/'queue-state.json').exists()


def test_all2540_cases_keep_both_zeros_and_depend_on_the_same_sample_reference(tmp_path):
    from test_frozen_cascade_case_plans import fixture
    from experiments.deformnext_retargeting.prepare_frozen_cascade_cases import prepare
    plan=fixture(tmp_path);compiled=prepare(plan,tmp_path/'native-plans')
    master=dict(job_prefix='unit',GPU_priority=50,execution={'path':'/unit-execution','SHA256':'unit'})
    execution=dict(python=sys.executable,source_root=str(Path(campaign.__file__).parents[2]))
    jobs=campaign.case_jobs(compiled,master=master,execution=execution,folder=tmp_path/'control',dependency='unit-cases')
    assert len(jobs)==2540 and len({j['name'] for j in jobs})==2540
    assert all(j['resource']=='exclusive' and j['depends_on'][0]=='unit-cases' for j in jobs)
    # The metadata fixture orders vanilla first and retains original-zero models3/4.
    for i in range(508):
        rows=jobs[5*i:5*i+5]
        assert [j['name'] for j in rows]==[f'unit-case-m{m:03d}-s{i:03d}' for m in range(5)]
        assert rows[0]['depends_on']==['unit-cases']
        assert all(j['depends_on'][-1]==rows[0]['name'] for j in rows[1:])
        assert all('--execution-sha256' in j['command'] for j in rows)


@pytest.mark.parametrize('kind',['SS','perceptual','Inception'])
def test_GPU_stage_uses_original_module_and_only_its_explicit_arguments(tmp_path,kind):
    module={'SS':'evaluate_bound_population','perceptual':'bound_official_perceptual','Inception':'extract_official_inception'}[kind]
    path=Path(gpu.__file__).with_name(module+'.py')
    plan=dict(kind=kind,native_plan={'path':'/frozen-native','SHA256':'native'},output='/frozen-output',
              entry_source=dict(path=str(path),SHA256=sha(path)))
    command=gpu.command_for(plan,{'python':'/assigned-python'})
    assert command[:5]==['/assigned-python','-u','-m','experiments.deformnext_retargeting.'+module,'--plan']
    if kind=='perceptual':assert command[-2:]==['--plan-sha256','native']
    elif kind=='Inception':assert command[-2:]==['--output','/frozen-output']
    else:assert command[-1]=='/frozen-native'


def test_unassigned_GPU_rejected_before_opening_any_native_plan(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7');monkeypatch.setenv('MATCHING_V2_ALLOWED_GPUS','0,1,2,3')
    with pytest.raises(ValueError,match='assigned'):gpu.execute({'path':'/must-not-read'},dict(allowed_physical_GPUs=['0','1','2','3']))


@pytest.mark.parametrize('fail',[False,True])
def test_GPU_wrapper_records_native_terminal_closes_success_and_preserves_failure(tmp_path,monkeypatch,fail):
    source=Path(gpu.__file__);root=source.parents[2]
    native=put(tmp_path/'native-plan.json',{'unit_only':True});freeze=put(tmp_path/'freeze.json',{'unit_only':True})
    output=tmp_path/'prediction'
    entry=source.with_name('evaluate_bound_population.py')
    spec=put(tmp_path/'job.json',dict(kind='SS',scope='final-test508',native_plan=native,model_freeze=freeze,
        output=str(output),entry_source=dict(path=str(entry),SHA256=sha(entry)),result_file='evaluation.json',result_status='PASS'))
    execution=dict(allowed_physical_GPUs=['0'],stage_executor_source=dict(path=str(source),SHA256=sha(source)),
        source_root=str(root),canonical_local_root=str(tmp_path),python='unit-mock-python')
    closures=[]
    class Child:
        pid=123456
        def __init__(self,command,**kwargs):
            assert kwargs['cwd']==str(root) and kwargs['env']['PYTHONPATH']==str(root)
            assert kwargs['env']['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
            kwargs['stdout'].write(b'unit-only simulated native output')
            output.mkdir()
            put(output/'evaluation.json',dict(status='PASS',bound_population=dict(scope='final-test508',model_freeze=freeze),
                native_SS_artifacts={'archived_asset_count':508},optimizer_updates_during_evaluation=0))
        def wait(self):return int(fail)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setenv('MATCHING_V2_ALLOWED_GPUS','0')
    monkeypatch.setattr(gpu.subprocess,'Popen',Child)
    monkeypatch.setattr(gpu.psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:123.))
    monkeypatch.setattr(gpu,'close',lambda folder,**kwargs:closures.append(folder))
    if fail:
        with pytest.raises(RuntimeError,match='preserved'):gpu.execute(spec,execution)
        assert not closures and json.loads((output/'terminal.json').read_text())['returncode']==1
    else:
        actual=gpu.execute(spec,execution)
        assert actual['returncode']==0 and closures==[output]
    assert (output/'native.log').read_bytes()==b'unit-only simulated native output'


@pytest.fixture
def master_input(tmp_path,monkeypatch):
    from test_frozen_ss_population_plans import fixture
    initial=fixture(tmp_path);freeze=json.loads(Path(initial['model_freeze']['path']).read_text())
    for alias,model in freeze['models'].items():model['original_zero_shot_stack']=alias.startswith('zero-')
    initial['model_freeze']=put(Path(initial['model_freeze']['path']),freeze)
    initial_spec=put(tmp_path/'initial-SS.json',initial)
    source=Path(campaign.__file__).parents[2]
    execution=dict(source_root=str(source),canonical_local_root=str(tmp_path),allowed_physical_GPUs=['0','1'],
        executor_source={'path':str(source/'experiments/deformnext_retargeting/execute_frozen_cascade_case.py'),
                         'SHA256':sha(source/'experiments/deformnext_retargeting/execute_frozen_cascade_case.py')},
        stage_executor_source={'path':str(Path(gpu.__file__)),'SHA256':sha(gpu.__file__)},python=sys.executable)
    execution_spec=put(tmp_path/'execution.json',execution)
    ids=freeze['population']['sample_ids']
    packet=put(tmp_path/'assets.json',dict(schema='native-final-cascade-asset-inputs-v1',assets=508,source_clusters=134,
        file4_cache_root='/cache/file4',
        records=[dict(sample_id=sid) for sid in ids]))
    metric=put(tmp_path/'metric.json',dict(status='PASS_OFFICIAL_PERCEPTUAL_WEIGHT_SOURCE_CACHE_AND_CPU_LOAD'))
    weight=tmp_path/'unit-weight';weight.write_bytes(b'unit-only not official weights')
    registry=put(tmp_path/'Inception.json',dict(package='torch-fidelity0.3.0',weights_SHA256=sha(weight)))
    GT=tmp_path/'GT';GT.mkdir()
    reference=put(GT/'audit.json',dict(status='PASS_ALL508_GT_SURFACE_REFERENCES',samples=508,source_clusters=134,
        surface_samples=20000,surface_seed=101,FScore_threshold=.02,source_defined_normalization=True,
        per_output_alignment=False,target_recentered_or_rescaled=False,geometry_frame='source-defined normalized WORLD',
        all508_source_target_topologies_and_final_vertices_exact=True,reference_points_readback_exact=True,source_SHA256={},
        records=[dict(sample_id=sid,source_cluster_id=freeze['population']['source_clusters_by_sample'][sid]) for sid in ids]))
    close_unit(GT,tmp_path)
    master=dict(scope='final-test508',job_prefix='unit',execution=execution_spec,SS_plan=initial_spec,
        output=str(tmp_path/'campaign'),master_receipt=str(tmp_path/'file4-closure.json'),file4_cache_root='/cache/file4',
        source_SHA256={str(source/'experiments/deformnext_retargeting'/name):sha(source/'experiments/deformnext_retargeting'/name) for name in campaign.SOURCES},
        asset_packet=packet,metric_registry=metric,GT_surface_reference=reference,GT_surface_reference_receipt=str(GT/'file4-closure.json'),
        Inception=dict(metric_registry=registry,weights=dict(path=str(weight),SHA256=sha(weight)),package_root=str(tmp_path),batch_size=32),
        minimum_free_bytes=1,queue_file=str(tmp_path/'queue.json'),CPU_priority=-1,GPU_priority=50,initial_dependencies=[])
    spec=put(tmp_path/'master.json',master);close_unit(tmp_path,tmp_path)
    monkeypatch.setenv('MATCHING_V2_ALLOWED_GPUS','0,1');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    return spec,master,initial


def test_master_preflight_and_real_metadata_SS_phase_append_only_unit_queue(master_input,tmp_path,monkeypatch):
    spec,master,initial=master_input
    campaign.load_master(spec)
    monkeypatch.setattr(campaign,'close',lambda folder,**kwargs:close_unit(folder,tmp_path))
    put(tmp_path/'queue.json',[])
    campaign.start(spec)
    queue=json.loads((tmp_path/'queue.json').read_text())
    assert [j['name'] for j in queue]==['unit-SS']
    command=queue[0]['command'];action={'path':command[command.index('--plan')+1],'SHA256':command[command.index('--plan-sha256')+1]}
    result=campaign.advance(action)
    after=json.loads((tmp_path/'queue.json').read_text())
    assert after[:1]==queue and result['jobs']==6
    assert len([j for j in after if j['resource']=='exclusive'])==5
    assert after[-1]['name']=='unit-cases' and len(after[-1]['depends_on'])==5
    assert not (tmp_path/'queue-state.json').exists()


@pytest.mark.parametrize('problem',['source','zero-control','assigned-GPU','Inception','storage'])
def test_incomplete_master_cannot_install_a_campaign(master_input,tmp_path,problem):
    spec,master,initial=master_input
    if problem=='source':master['source_SHA256'][str(Path(campaign.__file__))]='wrong-source'
    elif problem=='zero-control':
        path=Path(initial['model_freeze']['path']);freeze=json.loads(path.read_text());freeze['models']['zero-source4']['original_zero_shot_stack']=False
        initial['model_freeze']=put(path,freeze);master['SS_plan']=put(Path(master['SS_plan']['path']),initial)
    elif problem=='assigned-GPU':
        path=Path(master['execution']['path']);cfg=json.loads(path.read_text());cfg['allowed_physical_GPUs']=['7'];master['execution']=put(path,cfg)
    elif problem=='Inception':master['Inception']['batch_size']=True
    else:master['minimum_free_bytes']=True
    spec=put(Path(spec['path']),master);close_unit(tmp_path,tmp_path)
    with pytest.raises(ValueError):campaign.start(spec)
    assert not Path(master['output']).exists()


def test_all_deferred_campaign_phases_keep_the_same_freeze_and_finish_unit_DAG(master_input,tmp_path,monkeypatch):
    # Control-plane integration with mocked numerical producers only. No GPU,
    # feature bank, official score, real result receipt or live queue is made.
    spec,master,initial=master_input
    root=Path(master['output'])
    freeze=json.loads(Path(initial['model_freeze']['path']).read_text())
    aliases=list(freeze['models']);ids=freeze['population']['sample_ids']
    monkeypatch.setattr(campaign,'close',lambda folder,**kwargs:close_unit(folder,tmp_path))
    put(tmp_path/'queue.json',[])
    campaign.start(spec)
    def action(stage):
        queue=json.loads((tmp_path/'queue.json').read_text())
        job=next(j for j in queue if j['name']=='unit-'+stage);c=job['command']
        return {'path':c[c.index('--plan')+1],'SHA256':c[c.index('--plan-sha256')+1]}
    campaign.advance(action('SS'))
    ss=json.loads((root/'SS/manifest.json').read_text())
    for row in ss['evaluations'].values():
        folder=Path(row['output']);folder.mkdir()
        put(folder/'evaluation.json',{'unit_only_not_native_result':True});close_unit(folder,tmp_path)
    def cases(plan,output):
        assert plan['model_freeze']==initial['model_freeze'] and tuple(plan['SS_evaluations'])==tuple(aliases)
        output.mkdir();rows={}
        for mi,alias in enumerate(aliases):
            rows[alias]=[]
            for ci,sid in enumerate(ids):
                folder=output/'plans'/f'm{mi:03d}'/f's{ci:03d}'
                native=put(folder/'generation.json',dict(output=str(output/'models'/f'm{mi:03d}'/f's{ci:03d}'/'generation')))
                case=put(folder/'case.json',dict(phases=[dict(native_plan=native)],unit_control_only=True))
                rows[alias].append(dict(sample_id=sid,plan=case))
        result=dict(model_freeze=initial['model_freeze'],cases=rows,source_inputs=plan,unit_control_only=True)
        put(output/'manifest.json',result)
        return result
    monkeypatch.setattr(campaign.case_factory,'prepare',cases)
    campaign.advance(action('cases'))
    def perceptual(plan,output):
        assert plan['GT_surface_reference']==master['GT_surface_reference']
        assert tuple(plan['SS_evaluation_receipts'])==tuple(aliases)
        output.mkdir();perceptuals={}
        for i,alias in enumerate(aliases):
            native=put(output/f'model{i:03d}'/'plan.json',dict(unit_control_only=True))
            perceptuals[alias]=dict(plan=native,output=str(output/f'model{i:03d}'/'actual'))
        result=dict(source_inputs=plan,perceptuals=perceptuals,unit_control_only=True)
        put(output/'manifest.json',result)
        return result
    monkeypatch.setattr(campaign.metric_factory,'prepare',perceptual)
    campaign.advance(action('perceptual'))
    def bind_metrics(manifest):
        assert Path(manifest['path'])==root/'metric-inputs/manifest.json'
        return dict(scope='final-test508',model_freeze=initial['model_freeze'],unit_control_only=True)
    monkeypatch.setattr(campaign.metric_factory,'bind_completed',bind_metrics)
    calls=[]
    def numerical_boundary(execution,module,plan,output,status,control):
        calls.append(module);assert module=='assemble_final_metric_table' or json.loads(Path(plan['path']).read_text())['scope']=='final-test508'
        output.mkdir();result=put(output/'summary.json',dict(status=status,unit_control_only=True))
        close_unit(output,tmp_path);return result
    monkeypatch.setattr(campaign,'run_CPU',numerical_boundary)
    def features(plan,output):
        assert plan['model_freeze']==initial['model_freeze'] and plan['eight_metrics']['path']==str(root/'eight-metrics/summary.json')
        output.mkdir();pools={}
        for i,alias in enumerate([*aliases,None]):
            name=f'pool{i:03d}';native=put(output/(name+'.json'),dict(unit_control_only=True))
            pools[name]=dict(alias=alias,extract_plan=native,output=str(output/(name+'-features')))
        result=dict(pools=pools,source_inputs=plan,unit_control_only=True);put(output/'manifest.json',result);return result
    monkeypatch.setattr(campaign.pool_factory,'prepare',features)
    campaign.advance(action('eight'))
    pools=json.loads((root/'Inception-inputs/manifest.json').read_text())
    for row in pools['pools'].values():
        folder=Path(row['output']);folder.mkdir()
        put(folder/'audit.json',{'unit_control_only':True});(folder/'features.npz').write_bytes(b'unit control bytes, not features')
        close_unit(folder,tmp_path)
    monkeypatch.setattr(campaign.pool_factory,'bind_completed_pools',lambda manifest:dict(scope='final-test508',unit_control_only=True))
    campaign.advance(action('distribution'))
    campaign.advance(action('table'))
    assert calls==['frozen_cascade_metrics','summarize_frozen_inception_pools','assemble_final_metric_table']
    queue=json.loads((tmp_path/'queue.json').read_text())
    # 6 controller CPU jobs, 5 SS, 2540 cascades, 5 perceptual and 6 features.
    assert len(queue)==2562 and all(j['resource']=='exclusive' for j in queue if j['resource']!='cpu')
    complete=json.loads((root/'control/table/campaign-completed.json').read_text())
    assert complete['models']==aliases and complete['assets']==508 and not complete['full_goal_complete']
    assert not (tmp_path/'queue-state.json').exists()
