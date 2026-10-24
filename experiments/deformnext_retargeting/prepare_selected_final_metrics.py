"""Prepare exact-image selected508 official metric jobs after actual cascade."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from .cascade_view_roles import ClosedInputs,file_sha256 as sha
from .frozen_cascade_metrics import PinnedFiles
from .run_selected_final_campaign import atomic,close,closed_spec,write
from .run_selected_final_metric import SCHEMA as WORKER_SCHEMA,registry_guards
from .selected_campaign_resources import live,source_guard
from .selected_final_metric_inputs import MANIFEST_STATUS,feature_observations,load_campaign,required_sources
from .selected_final_metric_summary import reference_binding
from .selected_pipeline_model_contract import require


SCHEMA='selected-physical-final-metric-preparation-v1'
COLLECT_SCHEMA='selected-physical-final-ten-metric-collection-v1'


def append_jobs(context,jobs):
    prepared={}
    for queue in {job['queue'] for job in jobs}:
        raw=Path(queue).read_bytes();old=json.loads(raw);added=[{k:v for k,v in job.items() if k!='queue'} for job in jobs if job['queue']==queue]
        names=[j['name'] for j in old];new=[j['name'] for j in added]
        require(len(names)==len(set(names)) and len(new)==len(set(new)) and not set(names)&set(new), 'never overwrite or duplicate actual metric queue jobs')
        prepared[queue]=(raw,old,added)
    for queue,(raw,old,new) in prepared.items():
        require(Path(queue).read_bytes()==raw,'actual queue changed during metric preparation; no activation')
        atomic(queue,old+new)
    for queue,(_,old,new) in prepared.items():
        actual=json.loads(Path(queue).read_text());indexed={j['name']:j for j in actual}
        require(len(indexed)==len(actual) and all(indexed.get(j['name'])==j for j in [*old,*new]), 'retain all actual old and new metric descriptors')
    return {p:dict(SHA256=sha(p),added=[j['name'] for j in added]) for p,(_,old,added) in prepared.items()}


def prepare(path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit CPU-only completed-campaign metric preparation required')
    path=Path(path);context=json.loads(path.read_text());require(context.get('schema')==SCHEMA,'explicit selected metric preparation schema required')
    require(context['repo']==str(Path(__file__).resolve().parents[2]) and set(required_sources())<=set(context['source_SHA256']), 'actual complete source-bound metric checkout required')
    os.sched_setaffinity(0,set(context['CPU_affinity']));os.nice(10);source_guard(context)
    check=ClosedInputs(context['canonical_local_root']);check.check(path,path.parent/'file4-closure.json')
    parent,controls,models,cases,shared,images=load_campaign(context,check)
    devices=context['assigned_GPUs'];groups=context['queue_groups'];actual=[g for value in groups.values() for g in value['assigned_GPUs']]
    require(devices and len(devices)==len(set(devices)) and len(actual)==len(set(actual)) and set(devices)==set(actual)
        and set(context['CPU_affinity_per_GPU'])==set(devices),'explicit exact disjoint dispatcher GPU/CPU assignments required')
    for group in groups.values():
        require(live(group['dispatcher']),'actual existing metric dispatcher required')
        import psutil
        process=psutil.Process(group['dispatcher']['PID']);cmd=process.cmdline()
        require('experiments.deformnext_retargeting.matching_v2_queue' in cmd
            and str(Path(cmd[cmd.index('--work-root')+1])/'queue-jobs.json')==group['jobs']
            and str(Path(group['jobs']).parent/'queue-state.json')==group['states']
            and set(group['assigned_GPUs'])<=set(cmd[cmd.index('--gpus')+1].split(','))
            and process.cwd()==context['repo'],'actual metric queue command, source checkout and device scope required')
    require(context['CPU_queue'] in {g['jobs'] for g in groups.values()},'CPU reduction belongs to an actual owned dispatcher')
    pins=PinnedFiles();metric_models,inception=registry_guards(parent,pins)
    root=Path(context['output']);base=Path(context['canonical_local_root'])
    require(root.is_absolute() and root!=base and root.resolve()==root and root.is_relative_to(base) and not root.exists(), 'new owned metric preparation/output namespace required')
    root.mkdir(parents=True);tasks={}
    keys=[(alias,kind) for alias in models for kind in ['perceptual','inception']]+[('reference','inception')]
    for alias,kind in keys:
        reference=alias=='reference';rows=images['zero-source4' if reference else alias];binding=reference_binding(parent,controls) if reference else models[alias]
        output=root/'native'/alias/kind
        native=(dict(scope='final-test508',full_panel=True,metric_models=str(metric_models),output=str(output),inputs=rows)
            if kind=='perceptual' else dict(scope='final-test508',pool_role='shared-original-material-GT' if reference else 'generated-prediction',
                model_binding=binding,observations=feature_observations(rows,reference=reference),expected_sample_ids=parent['population']['sample_ids'],
                **inception,batch_size=32))
        folder=root/'native-plans'/alias/kind;folder.mkdir(parents=True);native_path=folder/'native.json';write(native_path,native)
        close(folder,context,'native-plan-'+alias+'-'+kind)
        tasks[alias+'/'+kind]=dict(alias=alias,kind=kind,model_binding=binding,native_plan=closed_spec(native_path),native_output=str(output))
    folder=root/'inputs';folder.mkdir();manifest_path=folder/'manifest.json'
    manifest=dict(status=MANIFEST_STATUS,scope='final-test508',source_inputs=context['source_inputs'],models=models,
        sample_ids=parent['population']['sample_ids'],assets=508,source_clusters=134,images=images,tasks=tasks,
        reference_binding=reference_binding(parent,controls),actual_completed_native_cases=508*len(models),
        actual_native_child_and_worker_terminals_verified=True,actual_same_scored_RGB_reference_alpha_and_uint8_verified=True,
        no_new_model_selection_or_GPU_forward=True,full_goal_complete=False)
    write(manifest_path,manifest);close(folder,context,'inputs')
    jobs=[];workers={};activation=root/'activation-v1/file4-closure.json'
    for index,(key,task) in enumerate(tasks.items()):
        gpu=devices[index%len(devices)];group=next(g for g in groups.values() if gpu in g['assigned_GPUs']);name=context['name']+'-'+task['alias']+'-'+task['kind']
        folder=root/'worker-plans'/task['alias']/task['kind'];folder.mkdir(parents=True);worker_path=folder/'context.json'
        output=root/'wrappers'/task['alias']/task['kind']
        own={k:deepcopy(context[k]) for k in ['repo','python','canonical_local_root','manifest_directory','closure_command','source_SHA256','source_inputs','assigned_GPUs','CPU_affinity_per_GPU']}
        own.update(schema=WORKER_SCHEMA,name=name,output=str(output),prepared_manifest=closed_spec(manifest_path),**task)
        write(worker_path,own);close(folder,context,'worker-plan-'+task['alias']+'-'+task['kind'])
        descriptor=dict(name=name,resource='exclusive',priority=-99,gpu=gpu,queue=group['jobs'],requires=[str(activation),str(folder/'file4-closure.json')],
            outputs=[str(output/'audit.json'),str(output/'file4-closure.json')],command=[context['python'],'-u','-m','experiments.deformnext_retargeting.run_selected_final_metric','--plan',str(worker_path)])
        jobs.append(descriptor);workers[key]=dict(plan=closed_spec(worker_path),output=str(output),job_name=name,jobs=group['jobs'],states=group['states'],dispatcher=group['dispatcher'])
    collector={k:deepcopy(context[k]) for k in ['repo','python','canonical_local_root','manifest_directory','closure_command','source_SHA256','source_inputs','CPU_affinity']}
    collector.update(schema=COLLECT_SCHEMA,name=context['name']+'-ten-metrics',prepared_manifest=closed_spec(manifest_path),workers=workers,output=str(root/'ten-metrics'))
    folder=root/'collector-plan';folder.mkdir();collector_path=folder/'context.json';write(collector_path,collector);close(folder,context,'collector-plan')
    jobs.append(dict(name=collector['name'],resource='cpu',priority=-98,queue=context['CPU_queue'],requires=[str(activation),str(folder/'file4-closure.json'),
        *[str(Path(w['output'])/'file4-closure.json') for w in workers.values()]],outputs=[str(Path(collector['output'])/'file4-closure.json')],
        command=[context['python'],'-u','-m','experiments.deformnext_retargeting.collect_selected_final_metrics','--plan',str(collector_path)]))
    pins.finish();source_guard(context)
    actual=append_jobs(context,jobs);folder=root/'activation-v1';folder.mkdir()
    result=dict(status='PASS_ACTUAL_SELECTED_PHYSICAL508_METRIC_JOBS_INSTALLED',models=list(models),assets=508,
        prepared_manifest=closed_spec(manifest_path),actual_queues=actual,workers=workers,collector=closed_spec(collector_path),
        official_GPU_jobs=len(workers),new_training_arms=0,actual_GPU_forward_performed_by_preparer=False,full_goal_complete=False)
    write(folder/'audit.json',result);close(folder,context,'activation');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.plan)),flush=True)


if __name__=='__main__':main()
