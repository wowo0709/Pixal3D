"""Collect selected physical508 metrics from actual completed GPU reports."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .cascade_view_roles import ClosedInputs,file_sha256 as sha
from .collect_module_seed_confirmation import read_file,verify_terminal
from .frozen_cascade_metrics import PinnedFiles
from .inception_distribution_metrics import official_distribution_metrics
from .metric_image_identity import validate_metric_image_join
from .prepare_selected_final_metrics import COLLECT_SCHEMA
from .run_selected_final_campaign import close,closed_spec,write
from .run_selected_final_metric import STATUS as WORKER_STATUS,registry_guards,verify_gpu_proof,verify_report
from .selected_campaign_resources import live,source_guard
from .selected_final_metric_inputs import (load_campaign,required_sources,validate_manifest)
from .selected_final_metric_summary import DIST_STATUS,combine_ten,eight_metrics,validate_contract
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


def completed_queue_worker(record,audit,context):
    """A completed native child is insufficient while its actual wrapper lives."""
    identity=audit['worker_identity'];expected=[context['python'],'-u','-m','experiments.deformnext_retargeting.run_selected_final_metric','--plan',record['plan']['path']]
    while True:
        states=json.loads(Path(record['states']).read_text());state=states.get(record['job_name'],{})
        require(state.get('status') not in ['FAILED','HELD'],'actual metric wrapper queue task failed; preserve outputs')
        if state.get('status')=='completed':break
        require(live(record['dispatcher']),'actual metric dispatcher must reap its worker; do not infer completion from output existence')
        time.sleep(1)
    require(state.get('pid')==identity['PID'] and state.get('returncode')==0
        and state.get('command')==expected and state.get('gpu')==audit['physical_GPU'] and not live(identity),
        'actual own metric wrapper must complete successfully and be absent')
    jobs=json.loads(Path(record['jobs']).read_text());rows=[j for j in jobs if j['name']==record['job_name']]
    require(len(rows)==1 and rows[0]['command']==expected,'actual completed descriptor differs from its prepared source-bound plan')
    verify_terminal(dict(**identity,returncode=0,actual_session_drained=True))
    verify_terminal(audit['actual_terminal'])


def collect(path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit CPU-only statistical metric reduction required')
    path=Path(path);context=json.loads(path.read_text());require(context.get('schema')==COLLECT_SCHEMA,'actual selected metric collection context required')
    require(context['repo']==str(Path(__file__).resolve().parents[2]) and set(required_sources())<=set(context['source_SHA256']), 'complete actual selected metric reducer source pins required')
    os.sched_setaffinity(0,set(context['CPU_affinity']));os.nice(10);source_guard(context)
    check=ClosedInputs(context['canonical_local_root']);check.check(path,path.parent/'file4-closure.json')
    parent,controls,models,cases,shared,images=load_campaign(context,check);manifest=read_file(context['prepared_manifest'],check)
    validate_manifest(manifest,parent,controls,models,context['source_inputs'])
    require(exact(images,manifest['images']) and set(context['workers'])==set(manifest['tasks']), 'all actual final pixels and GPU worker jobs must match preparation')
    pins=PinnedFiles();_,inception=registry_guards(parent,pins);official={};banks={};audits={};proofs={}
    for key,record in context['workers'].items():
        worker=read_file(record['plan'],check);audit=read_file(closed_spec(Path(record['output'])/'audit.json'),check)
        require(audit.get('status')==WORKER_STATUS and audit.get('alias')==worker['alias'] and audit.get('kind')==worker['kind']
            and exact(audit.get('executed_plan'),record['plan']) and exact(audit.get('prepared_manifest'),context['prepared_manifest'])
            and exact(audit.get('model_binding'),worker['model_binding']) and audit.get('actual_images')==1524
            and audit.get('CPU_model_forward_or_training_updates')==0 and audit.get('actual_pixel_bytes_verified_before_and_after_GPU') is True,
            'actual completed same-model saved metric worker binding required')
        completed_queue_worker(record,audit,worker);native=read_file(worker['native_plan'],check)
        proof=read_file(audit['actual_GPU_source_proof'],check);verify_gpu_proof(worker,proof,audit['physical_GPU'])
        report=read_file(audit['actual_native_report'],check);root=Path(worker['native_output']);features=None
        if worker['kind']=='inception':
            features=root/'features.npz';check.check(features,features.parent/'file4-closure.json')
        verify_report(worker,native,manifest,parent,report,features=features)
        if worker['kind']=='perceptual':official[worker['alias']]=report
        else:
            with np.load(features,allow_pickle=False) as bank:banks[worker['alias']]=bank['features'].copy()
            audits[worker['alias']]=report
        proofs[key]=dict(wrapper=closed_spec(Path(record['output'])/'audit.json'),actual_native_report=audit['actual_native_report'],
            actual_worker_identity=audit['worker_identity'],actual_native_terminal=audit['actual_terminal'],physical_GPU=audit['physical_GPU'])
    eight=eight_metrics(parent=parent,controls=controls,cases=cases,shared=shared,perceptual=official)
    require(set(banks)==set(models)|{'reference'},'all actual prediction/reference official feature pools required')
    contract=validate_contract(parent);package=Path(inception['package_root']);sys.path.insert(0,str(package))
    import torch
    import torch_fidelity
    torch.set_num_threads(2)
    require(torch_fidelity.__version__=='0.3.0' and Path(torch_fidelity.__file__).resolve().is_relative_to(package.resolve()),'explicit isolated original torch-fidelity0.3.0 statistical routines required')
    distribution=dict(status=DIST_STATUS,models=models,distribution_metrics={},
        feature_image_observations={a:audits[a]['observations'] for a in models},reference_image_observations=audits['reference']['observations'],
        reference_binding=manifest['reference_binding'],metric_registry=parent['evaluation_protocol']['distribution_metric_registry'],
        FID_KID_CI_or_significance_claim=False,full_goal_complete=False)
    validate_metric_image_join(eight,distribution,sample_ids=parent['population']['sample_ids'],clusters=parent['population']['source_clusters_by_sample'],models=models)
    for alias in models:
        distribution['distribution_metrics'][alias]=official_distribution_metrics(banks[alias],banks['reference'],**contract['FID_KID'])
        print(json.dumps(dict(completed_distribution=alias,metrics=distribution['distribution_metrics'][alias])),flush=True)
    result=combine_ten(eight,distribution,parent=parent,controls=controls)
    source_guard(context);pins.finish()
    for p,digest in check.inputs.items():require(sha(p)==digest,'actual closed final metric source/input changed during reduction')
    root=Path(context['output']);base=Path(context['canonical_local_root'])
    require(root.is_absolute() and root!=base and root.resolve()==root and root.is_relative_to(base) and not root.exists(),'new owned final ten-metric report output required')
    root.mkdir(parents=True);write(root/'eight-metrics.json',eight);write(root/'distribution-metrics.json',distribution)
    result.update(source_inputs=context['source_inputs'],prepared_manifest=context['prepared_manifest'],actual_GPU_worker_proofs=proofs,
        input_SHA256=check.inputs,source_SHA256=context['source_SHA256'],all_ten_metrics_collected=True)
    write(root/'audit.json',result)
    table=result['primary_ten_metric_table'];fields=list(next(iter(table.values())))
    with (root/'metrics.csv').open('x') as stream:
        import csv
        writer=csv.writer(stream);writer.writerow(['model',*fields])
        for alias,row in table.items():writer.writerow([alias,*[row[k] for k in fields]])
    close(root,context,'ten-metrics');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();result=collect(args.plan);print(json.dumps(dict(status=result['status'],table=result['primary_ten_metric_table'])),flush=True)


if __name__=='__main__':main()
