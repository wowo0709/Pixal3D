"""Append a frozen full508 campaign to an existing resource dispatcher.

Each CPU phase publishes immutable plans and real canonical receipts before
appending its dependent jobs. The existing queue prefix/state and live jobs
are never edited. Model selection is an input, never an action here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from .bind_generated_pbr_dataset import load_QA
from .cascade_view_roles import ClosedInputs, require
from .evaluate_bound_population import pinned, sha
from .execute_frozen_cascade_case import close
from .bind_native_surface_reference import SurfaceReferences
from .summarize_frozen_inception_pools import validate_frozen_models
from . import prepare_frozen_ss_population as SS_factory
from . import prepare_frozen_cascade_cases as case_factory
from . import prepare_frozen_cascade_metrics as metric_factory
from . import prepare_frozen_inception_pools as pool_factory

PACKAGE='experiments.deformnext_retargeting.'
BOOTSTRAP=('import os,runpy,sys; root=sys.argv.pop(1); module=sys.argv.pop(1); '
           'sys.path.insert(0,root); os.environ["PYTHONPATH"]=root; runpy.run_module(module,run_name="__main__")')
SOURCES=('frozen_campaign.py','frozen_campaign_stage.py','prepare_frozen_ss_population.py','evaluate_bound_population.py',
         'prepare_frozen_cascade_cases.py','execute_frozen_cascade_case.py','prepare_frozen_cascade_metrics.py',
         'bound_official_perceptual.py','frozen_cascade_metrics.py','bind_native_surface_reference.py',
         'prepare_frozen_inception_pools.py','extract_official_inception.py',
         'summarize_frozen_inception_pools.py','assemble_final_metric_table.py')


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    return dict(path=str(path),SHA256=sha(path))


def module_command(execution,module,*args):
    return [execution['python'],'-u','-c',BOOTSTRAP,execution['source_root'],PACKAGE+module,*args]


def guard_sources(master):
    for path,digest in master['source_SHA256'].items():
        require(sha(path)==digest,'frozen campaign source changed during execution: '+path)


def load_master(spec):
    master=json.loads(pinned(spec).read_text())
    require(master['scope']=='final-test508' and re.fullmatch(r'[A-Za-z0-9_-]+',master['job_prefix']),
            'explicit frozen final508 campaign/prefix required')
    execution=json.loads(pinned(master['execution']).read_text())
    root=Path(master['output']).resolve();source=Path(execution['source_root']).resolve()
    require(source==Path(__file__).resolve().parents[2]
            and root!=Path(execution['canonical_local_root']).resolve()
            and root.is_relative_to(Path(execution['canonical_local_root']).resolve()),
            'compiled source checkout and owned canonical campaign output required')
    for name in SOURCES:
        path=source/'experiments/deformnext_retargeting'/name
        require(master['source_SHA256'].get(str(path))==sha(path),'frozen campaign source differs: '+name)
    allowed=execution['allowed_physical_GPUs']
    require(allowed and len(set(allowed))==len(allowed) and all(isinstance(g,str) and g.isdigit() for g in allowed)
            and set(allowed)==set(os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(',')),
            'explicit assigned campaign GPU allow-list required')
    require(pinned(execution['executor_source']).resolve()==source/'experiments/deformnext_retargeting/execute_frozen_cascade_case.py'
            and pinned(execution['stage_executor_source']).resolve()==source/'experiments/deformnext_retargeting/frozen_campaign_stage.py',
            'pinned native case and GPU stage executors required')
    initial=json.loads(pinned(master['SS_plan']).read_text())
    load_QA(initial)
    freeze=json.loads(pinned(initial['model_freeze']).read_text())
    ids,clusters,_=validate_frozen_models(freeze,freeze['models'])
    modes=[m['ss64']['conditioning_mode'] for m in freeze['models'].values() if m['original_zero_shot_stack']]
    require(sorted(modes)==['edited-rgb','source-only'],'retain both original full-stack zero-shot RGB controls')
    closed=ClosedInputs(execution['canonical_local_root'])
    closed.check(spec['path'],master['master_receipt'])
    packet=json.loads(pinned(master['asset_packet']).read_text())
    require(packet['schema']=='native-final-cascade-asset-inputs-v1' and packet['assets']==508
            and packet['source_clusters']==134 and [r['sample_id'] for r in packet['records']]==ids
            and packet['file4_cache_root']==master['file4_cache_root']
            and master['file4_cache_root']==json.loads(pinned(initial['cache_QA_plan']).read_text())['file4_cache'],
            'complete ordered native508 cascade asset packet required')
    metric=json.loads(pinned(master['metric_registry']).read_text())
    require(metric['status']=='PASS_OFFICIAL_PERCEPTUAL_WEIGHT_SOURCE_CACHE_AND_CPU_LOAD',
            'prepared official perceptual models required before campaign start')
    SurfaceReferences(master['GT_surface_reference'],receipt=master['GT_surface_reference_receipt'],
        closed=closed,ids=ids,clusters=clusters)
    inception=master['Inception'];registry=json.loads(pinned(inception['metric_registry']).read_text())
    pinned(inception['weights'])
    require(registry['package']=='torch-fidelity0.3.0' and registry['weights_SHA256']==inception['weights']['SHA256']
            and Path(inception['package_root']).is_dir() and isinstance(inception['batch_size'],int)
            and not isinstance(inception['batch_size'],bool) and inception['batch_size']>0,
            'explicit prepared official Inception package/weights and positive batch required')
    require(master['minimum_free_bytes']>0 and isinstance(master['minimum_free_bytes'],int)
            and not isinstance(master['minimum_free_bytes'],bool),
            'explicit positive local storage reserve required')
    return master,execution,root,initial,freeze,ids,clusters,closed


def append_jobs(queue,jobs,*,folder,execution):
    """Preserve existing descriptors; fail instead of overwriting a changed queue."""
    before=queue.read_bytes();previous=json.loads(before)
    names=[j['name'] for j in previous];added=[j['name'] for j in jobs]
    require(len(set(names))==len(names) and len(set(added))==len(added) and not set(names)&set(added),
            'unique append-only campaign jobs required')
    declared=set(names+added)
    require(all(dep in declared for job in jobs for dep in job.get('depends_on',[])),
            'all campaign dependencies must name declared jobs')
    dump(folder/'queue-before.json',previous);dump(folder/'queue-after.json',previous+jobs)
    dump(folder/'append.json',dict(status='PREPARED_APPEND_ONLY_FROZEN_CAMPAIGN_JOBS',added_jobs=len(jobs),
        previous_jobs=len(previous),old_prefix_unchanged=True,no_queue_state_mutation=True))
    close(folder,execution=execution,tag='campaign-control')
    candidate=queue.with_suffix('.frozen-campaign.tmp')
    if candidate.exists():raise FileExistsError('preserve another queue append candidate')
    candidate.write_text(json.dumps(previous+jobs,indent=2)+'\n')
    require(queue.read_bytes()==before,'concurrent queue writer: preserve candidate and original queue')
    os.replace(candidate,queue)


def action_job(master_spec,master,execution,stage,folder,depends):
    plan=dump(folder/('next-'+stage+'.json'),dict(master=master_spec,stage=stage))
    return dict(name=master['job_prefix']+'-'+stage,resource='cpu',priority=master['CPU_priority'],
        depends_on=depends,requires=[str(folder/'file4-closure.json')],
        outputs=[str(Path(master['output'])/'control'/stage/'file4-closure.json')],
        command=module_command(execution,'frozen_campaign','advance','--plan',plan['path'],'--plan-sha256',plan['SHA256']))


def gpu_job(master,execution,*,name,kind,native,output,result_file,result_status,folder,depends):
    module={'SS':'evaluate_bound_population','perceptual':'bound_official_perceptual','Inception':'extract_official_inception'}[kind]
    source=Path(execution['source_root'])/'experiments/deformnext_retargeting'/(module+'.py')
    initial=json.loads(pinned(master['SS_plan']).read_text())
    plan=dump(folder/(name+'-job.json'),dict(scope='final-test508',kind=kind,native_plan=native,output=output,
        result_file=result_file,result_status=result_status,entry_source=dict(path=str(source),SHA256=sha(source)),
        model_freeze=initial['model_freeze']))
    return dict(name=name,resource='exclusive',priority=master['GPU_priority'],depends_on=depends,
        requires=[str(folder/'file4-closure.json')],outputs=[str(Path(output)/'file4-closure.json')],
        command=module_command(execution,'frozen_campaign_stage','--plan',plan['path'],'--plan-sha256',plan['SHA256'],
            '--execution',master['execution']['path'],'--execution-sha256',master['execution']['SHA256']))


def case_jobs(manifest,*,master,execution,folder,dependency):
    """Interleave samples/models, with one shared-reference producer per sample."""
    freeze=json.loads(pinned(manifest['model_freeze']).read_text())
    aliases=list(freeze['models']);reference=freeze['comparison_roles']['vanilla']
    reference_index=aliases.index(reference);jobs=[]
    for ci in range(508):
        for mi,alias in enumerate(aliases):
            item=manifest['cases'][alias][ci];case=json.loads(pinned(item['plan']).read_text())
            native=json.loads(pinned(case['phases'][0]['native_plan']).read_text())
            root=Path(native['output']).parent
            name=f"{master['job_prefix']}-case-m{mi:03d}-s{ci:03d}"
            depends=[dependency]
            if alias!=reference:depends.append(f"{master['job_prefix']}-case-m{reference_index:03d}-s{ci:03d}")
            jobs.append(dict(name=name,resource='exclusive',priority=master['GPU_priority'],depends_on=depends,
                requires=[str(folder/'file4-closure.json')],outputs=[str(root/'file4-closure.json')],
                command=module_command(execution,'execute_frozen_cascade_case','--plan',item['plan']['path'],
                    '--plan-sha256',item['plan']['SHA256'],'--execution',master['execution']['path'],
                    '--execution-sha256',master['execution']['SHA256'])))
    return jobs


def checked_manifest(root,closed):
    path=root/'manifest.json';closed.check(path,root/'file4-closure.json')
    return json.loads(path.read_text()),dict(path=str(path),SHA256=sha(path))


def run_CPU(execution,module,plan,output,status,control):
    pinned(plan)
    command=[execution['python'],'-u','-m',PACKAGE+module,'--plan',plan['path'],'--output',str(output)]
    with (control/(module+'.log')).open('xb') as stream:
        result=subprocess.run(command,cwd=execution['source_root'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',
            PYTHONPATH=execution['source_root'],OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2'),
            stdout=stream,stderr=subprocess.STDOUT)
    require(result.returncode==0,'CPU metric reduction failed; outputs preserved')
    pinned(plan)
    path=output/'summary.json';report=json.loads(path.read_text())
    require(report['status']==status,'CPU metric completion status differs')
    close(output,execution=execution,tag=module)
    return dict(path=str(path),SHA256=sha(path))


def advance(action_spec):
    action=json.loads(pinned(action_spec).read_text());master_spec=action['master']
    master,execution,root,initial,freeze,ids,clusters,closed=load_master(master_spec)
    require(os.environ.get('CUDA_VISIBLE_DEVICES','')=='','campaign coordinator is CPU-only')
    stage=action['stage'];current=master['job_prefix']+'-'+stage
    require(stage in {'SS','cases','perceptual','eight','distribution','table'},'explicit campaign phase required')
    import shutil
    require(shutil.disk_usage(root).free>=master['minimum_free_bytes'],'campaign storage reserve unavailable; outputs preserved')
    control=root/'control'/stage
    if control.exists():raise FileExistsError('preserve previous campaign phase and failed outputs')
    control.mkdir(parents=True)
    queue=Path(master['queue_file']);jobs=[]
    if stage=='SS':
        result=SS_factory.prepare(initial,root/'SS')
        close(root/'SS',execution=execution,tag='SS-plans')
        for key,row in result['evaluations'].items():
            jobs.append(gpu_job(master,execution,name=master['job_prefix']+'-SS-'+key,kind='SS',
                native=row['native_bound_plan'],output=row['output'],result_file='evaluation.json',result_status='PASS',
                folder=control,depends=[current]))
        next_stage='cases'
    elif stage=='cases':
        ss,_=checked_manifest(root/'SS',closed);evaluations={}
        for row in ss['evaluations'].values():
            path=Path(row['output'])/'evaluation.json';closed.check(path,path.parent/'file4-closure.json')
            evaluations[row['alias']]=dict(path=str(path),SHA256=sha(path))
        plan=dict(scope='final-test508',**{k:initial[k] for k in ['model_freeze','cache_QA','cache_QA_plan','manifest']},
            SS_evaluations=evaluations,asset_packet=master['asset_packet'],file4_cache_root=master['file4_cache_root'])
        result=case_factory.prepare(plan,root/'cascade')
        close(root/'cascade',execution=execution,tag='native-case-plans')
        jobs=case_jobs(result,master=master,execution=execution,folder=control,dependency=current)
        next_stage='perceptual'
    elif stage=='perceptual':
        _,cases=checked_manifest(root/'cascade',closed);ss,_=checked_manifest(root/'SS',closed)
        receipts={row['alias']:str(Path(row['output'])/'file4-closure.json') for row in ss['evaluations'].values()}
        plan=dict(scope='final-test508',case_manifest=cases,prepared_cases_receipt=str(root/'cascade/file4-closure.json'),
            SS_evaluation_receipts=receipts,canonical_local_root=execution['canonical_local_root'],
            metric_registry=master['metric_registry'],GT_surface_reference=master['GT_surface_reference'],
            GT_surface_reference_receipt=master['GT_surface_reference_receipt'])
        result=metric_factory.prepare(plan,root/'metric-inputs')
        close(root/'metric-inputs',execution=execution,tag='perceptual-plans')
        for i,(alias,row) in enumerate(result['perceptuals'].items()):
            jobs.append(gpu_job(master,execution,name=f"{master['job_prefix']}-perceptual-m{i:03d}",kind='perceptual',
                native=row['plan'],output=row['output'],result_file='evaluation.json',
                result_status='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION',folder=control,depends=[current]))
        next_stage='eight'
    elif stage=='eight':
        _,metric_manifest=checked_manifest(root/'metric-inputs',closed)
        plan=metric_factory.bind_completed(metric_manifest)
        bound=dump(control/'eight-plan.json',plan)
        eight=run_CPU(execution,'frozen_cascade_metrics',bound,root/'eight-metrics',
            'PASS_ACTUAL_BOUND_CASCADE_EIGHT_METRICS',control)
        pool_plan=dict(scope='final-test508',model_freeze=initial['model_freeze'],eight_metrics=eight,**master['Inception'])
        result=pool_factory.prepare(pool_plan,root/'Inception-inputs')
        close(root/'Inception-inputs',execution=execution,tag='Inception-plans')
        for name,row in result['pools'].items():
            jobs.append(gpu_job(master,execution,name=master['job_prefix']+'-Inception-'+name,kind='Inception',
                native=row['extract_plan'],output=row['output'],result_file='audit.json',
                result_status='PASS_ACTUAL_OFFICIAL_INCEPTION2048_FEATURE_POOL',folder=control,depends=[current]))
        next_stage='distribution'
    elif stage=='distribution':
        result,manifest=checked_manifest(root/'Inception-inputs',closed)
        for row in result['pools'].values():
            folder=Path(row['output'])
            for name in ['audit.json','features.npz']:closed.check(folder/name,folder/'file4-closure.json')
        plan=pool_factory.bind_completed_pools(manifest)
        bound=dump(control/'distribution-plan.json',plan)
        run_CPU(execution,'summarize_frozen_inception_pools',bound,root/'distribution-metrics',
            'PASS_ACTUAL_FROZEN_MODEL_TEST508_OFFICIAL_FID_KID',control)
        next_stage='table'
    else:
        values={}
        for key,folder in [('eight_metrics','eight-metrics'),('distribution_metrics','distribution-metrics')]:
            path=root/folder/'summary.json';closed.check(path,path.parent/'file4-closure.json')
            values[key]=dict(path=str(path),SHA256=sha(path))
        bound=dump(control/'table-plan.json',dict(model_freeze=initial['model_freeze'],**values))
        table=run_CPU(execution,'assemble_final_metric_table',bound,root/'ten-metric-table',
            'PASS_ACTUAL_FROZEN_TEST508_TEN_METRIC_TABLE',control)
        dump(control/'campaign-completed.json',dict(status='PASS_ACTUAL_COMPLETE_FROZEN_NATIVE_TEST508_CAMPAIGN',
            model_freeze=initial['model_freeze'],table=table,assets=508,models=list(freeze['models']),
            every_frozen_model_and_original_zero_control_retained=True,model_selected_by_this_campaign=False,full_goal_complete=False))
        pinned(master_spec);pinned(action_spec);guard_sources(master)
        close(control,execution=execution,tag='campaign-final')
        return {'status':'PASS_ACTUAL_COMPLETE_FROZEN_NATIVE_TEST508_CAMPAIGN','table':table}
    jobs.append(action_job(master_spec,master,execution,next_stage,control,[j['name'] for j in jobs] or [current]))
    pinned(master_spec);pinned(action_spec);guard_sources(master)
    append_jobs(queue,jobs,folder=control,execution=execution)
    return dict(status='APPENDED_FROZEN_CAMPAIGN_PHASE',stage=stage,jobs=len(jobs),full_goal_complete=False)


def start(master_spec):
    master,execution,root,*_=load_master(master_spec)
    if root.exists():raise FileExistsError('fresh frozen campaign namespace required')
    folder=root/'bootstrap';folder.mkdir(parents=True)
    job=action_job(master_spec,master,execution,'SS',folder,master['initial_dependencies'])
    append_jobs(Path(master['queue_file']),[job],folder=folder,execution=execution)
    return dict(status='APPENDED_FROZEN_CAMPAIGN_BOOTSTRAP',job=job['name'],full_goal_complete=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['start','advance']);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();spec=dict(path=str(args.plan),SHA256=args.plan_sha256)
    print(json.dumps(start(spec) if args.command=='start' else advance(spec)),flush=True)


if __name__=='__main__':main()
