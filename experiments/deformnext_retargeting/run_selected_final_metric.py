"""Bound GPU-only official metrics for one actual selected physical508 model."""
import argparse
import json
import os
from pathlib import Path

from .assigned_gpu_entry import validate_assignment
from .cascade_view_roles import ClosedInputs,file_sha256 as sha
from .collect_module_seed_confirmation import read_file
from .frozen_cascade_metrics import PinnedFiles
from .native_final_metric_summary import validate_feature_pool
from .run_selected_final_case import run_child
from .run_selected_final_campaign import close,closed_spec,write
from .selected_campaign_resources import process_identity,source_guard
from .selected_final_metric_inputs import (expected_feature_observations,feature_observations,frozen_inputs,
    required_sources,validate_images,validate_manifest)
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


SCHEMA='selected-physical-final-metric-execution-v1'
STATUS='PASS_ACTUAL_SELECTED_PHYSICAL508_OFFICIAL_METRIC_EXECUTION'


def verify_gpu_proof(context,proof,gpu):
    require(proof.get('actual_entry_success') is True and proof.get('physical_GPU')==gpu
        and proof.get('assigned_GPUs')==context['assigned_GPUs'] and proof.get('original_source_files_unmodified') is True,
        'actual successful same-GPU source-preserving entry proof required')
    if context['kind']=='perceptual':
        entry='experiments.deformnext_retargeting.evaluate_official_perceptual'
        body=proof['adapted_modules'][entry]
        require(proof.get('numerical_AST_unchanged') is True and body.get('all_other_AST_statements_unchanged') is True
            and body.get('changed_membership_literals')==1 and context['source_SHA256'].get(body['original_source'])==body['original_source_SHA256'],
            'unchanged original perceptual numerical AST and exactly one device guard adaptation required')
    else:
        require(proof.get('all_other_AST_statements_unchanged') is True and proof.get('changed_membership_literals')==1
            and context['source_SHA256'].get(proof['original_source'])==proof['original_source_SHA256'],
            'unchanged original Inception numerical AST and exactly one device guard adaptation required')


def registry_guards(parent,pins):
    """Read/check registered bytes only; no CPU NN construction or forward."""
    perceptual=pins.check(parent['evaluation_protocol']['metric_registry']);registry=json.loads(perceptual.read_text())
    require(registry['status']=='PASS_OFFICIAL_PERCEPTUAL_WEIGHT_SOURCE_CACHE_AND_CPU_LOAD','original official perceptual registry required')
    for row in registry['files']:
        path=perceptual.parent/row['path'];require(path.stat().st_size==row['bytes'],'actual metric registered file size differs')
        pins.check(dict(path=str(path),SHA256=row['sha256']))
    distribution=pins.check(parent['evaluation_protocol']['distribution_metric_registry']);inc=json.loads(distribution.read_text())
    require(inc['package']=='torch-fidelity0.3.0' and inc['KID_subsets']==100 and inc['KID_subset_size']==1000
        and inc['KID_rng_seed']==42017,'original official distribution package/settings required')
    for path,digest in inc['source_SHA256'].items():pins.check(dict(path=path,SHA256=digest))
    weights=distribution.parent/'weights/weights-inception-2015-12-05-6726825d.pth'
    pins.check(dict(path=str(weights),SHA256=inc['weights_SHA256']))
    require(weights.stat().st_size==inc['weights_bytes'],'actual registered Inception bytes differ')
    return perceptual.parent,dict(weights=str(weights),weights_SHA256=inc['weights_SHA256'],package_root=inc['isolated_python_root'])


def verify_report(context,native,manifest,parent,report,*,features=None):
    alias=context['alias'];kind=context['kind'];reference=alias=='reference';source='zero-source4' if reference else alias
    rows=manifest['images'][source]
    if kind=='perceptual':
        require(report.get('status')=='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION'
            and report.get('scope')=='final-test508' and report.get('full_panel') is True and report.get('fullframe_primary') is True
            and report.get('metric_registry_SHA256')==parent['evaluation_protocol']['metric_registry']['SHA256']
            and report.get('model_precision')=='float32,TF32off,noAMP' and report.get('no_model_training') is True
            and report.get('GPU_sanity',{}).get('status')=='PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY'
            and report['GPU_sanity'].get('CPU_model_forward_or_training_updates')==0 and len(report['per_view'])==1524,
            'actual registered FP32 GPU-only full508 official DreamSim and separate OpenAI CLIP required')
        for expected,actual in zip(rows,report['per_view']):
            require(all(exact(expected[k],actual[k]) for k in ['arm','sample_id','source_cluster_id','camera_id','shared_npz_SHA256','image_audit']),
                'completed perceptual score differs from actual prepared RGB/reference/crop input')
    else:
        import numpy as np
        require(features is not None,'actual completed feature NPZ required')
        with np.load(features,allow_pickle=False) as bank:
            values=bank['features'];stored=list(zip(bank['sample_id'].tolist(),bank['camera_id'].tolist(),bank['source_cluster_id'].tolist()))
        expected=expected_feature_observations(rows,reference=reference)
        validate_feature_pool(report,values,observations=expected,model_binding=context['model_binding'],weights_SHA256=native['weights_SHA256'])
        require(stored==[(r['sample_id'],r['camera_id'],r['source_cluster_id']) for r in expected]
            and report.get('features_SHA256')==sha(features) and report.get('input_plan_SHA256')==context['native_plan']['SHA256']
            and report.get('pool_role')==native['pool_role'],'actual feature matrix bytes/row order/plan/reference role differ')


def execute(path):
    path=Path(path);context=json.loads(path.read_text());gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    require(context.get('schema')==SCHEMA and context['repo']==str(Path(__file__).resolve().parents[2]),'actual selected final metric context and checkout required')
    validate_assignment(context,gpu,os.environ.get('MATCHING_V2_ALLOWED_GPUS',''));os.sched_setaffinity(0,set(context['CPU_affinity_per_GPU'][gpu]))
    source_guard(context);check=ClosedInputs(context['canonical_local_root']);check.check(path,path.parent/'file4-closure.json')
    parent,controls,models=frozen_inputs(context,check);manifest=read_file(context['prepared_manifest'],check)
    validate_manifest(manifest,parent,controls,models,context['source_inputs']);alias=context['alias'];kind=context['kind']
    require(kind in ['perceptual','inception'] and (alias in models or alias=='reference' and kind=='inception'), 'one explicit actual model or shared-reference metric job required')
    binding=manifest['reference_binding'] if alias=='reference' else models[alias]
    require(exact(context['model_binding'],binding),'metric job belongs to a different physical model')
    task=manifest['tasks'][alias+'/'+kind]
    require(exact(task['native_plan'],context['native_plan']) and task['native_output']==context['native_output'], 'execute precisely the prepared numerical plan')
    native=read_file(context['native_plan'],check);pins=PinnedFiles();metric_models,inception=registry_guards(parent,pins)
    rows=manifest['images']['zero-source4' if alias=='reference' else alias]
    if kind=='perceptual':
        require(native['metric_models']==str(metric_models) and exact(native['inputs'],rows) and native['scope']=='final-test508'
            and native['full_panel'] is True and native['output']==context['native_output'],'actual prepared original perceptual numerical plan required')
    else:
        require(all(native[k]==v for k,v in inception.items()) and native['scope']=='final-test508'
            and exact(native['model_binding'],binding) and native['expected_sample_ids']==parent['population']['sample_ids']
            and exact(native['observations'],feature_observations(rows,reference=alias=='reference')),
            'actual original Inception plan must use this scored RGB bank and registered weights')
    validate_images(rows,alias='zero-source4' if alias=='reference' else alias,parent=parent,check=check)
    root=Path(context['output']);native_output=Path(context['native_output']);base=Path(context['canonical_local_root'])
    require(all(p.is_absolute() and p!=base and p.is_relative_to(base) and p.resolve()==p and not p.exists() for p in [root,native_output])
        and not root.is_relative_to(native_output) and not native_output.is_relative_to(root),'new separate owned metric wrapper/native outputs required')
    root.mkdir(parents=True);terminal=None
    try:
        assigned=['--assignment',str(path),'--assignment-sha256',sha(path),'--runtime-proof',str(root/'GPU-assignment-proof.json')]
        command=[context['python'],'-u','-m']
        if kind=='perceptual':command+=['experiments.deformnext_retargeting.assigned_gpu_entry','--entry','official-perceptual','--plan',str(context['native_plan']['path']),*assigned]
        else:command+=['experiments.deformnext_retargeting.selected_inception_entry','--plan',str(context['native_plan']['path']),'--output',str(native_output),*assigned]
        terminal=run_child(context,root,kind,command)
        verify_gpu_proof(context,json.loads((root/'GPU-assignment-proof.json').read_text()),gpu)
        report_path=native_output/('evaluation.json' if kind=='perceptual' else 'audit.json');report=json.loads(report_path.read_text())
        verify_report(context,native,manifest,parent,report,features=None if kind=='perceptual' else native_output/'features.npz')
        validate_images(rows,alias='zero-source4' if alias=='reference' else alias,parent=parent,check=check)
        pins.finish();source_guard(context)
        for p,d in check.inputs.items():require(sha(p)==d,'actual closed metric inputs changed during GPU forwards')
        close(native_output,context,'native-result')
        audit=dict(status=STATUS,alias=alias,kind=kind,model_binding=binding,prepared_manifest=context['prepared_manifest'],
            parent_freeze=context['source_inputs']['parent_freeze'],control_freeze=context['source_inputs']['control_freeze'],
            executed_plan=closed_spec(path),native_plan=context['native_plan'],actual_native_report=closed_spec(report_path),
            actual_terminal=terminal,worker_identity=process_identity(),physical_GPU=gpu,actual_images=1524,CPU_model_forward_or_training_updates=0,
            actual_GPU_source_proof=closed_spec(root/'GPU-assignment-proof.json'),
            original_numerical_code_unchanged=True,actual_pixel_bytes_verified_before_and_after_GPU=True,
            input_SHA256=check.inputs,source_SHA256=context['source_SHA256'],full_goal_complete=False)
        write(root/'audit.json',audit);close(root,context,'wrapper-result');return audit
    except BaseException as exc:
        write(root/'failure.json',dict(status='FAILED_SELECTED_FINAL_METRIC_PRESERVED',error=repr(exc),actual_terminal=terminal,automatic_retry=False,full_goal_complete=False))
        if not (root/'file4-closure.json').exists():close(root,context,'failed-wrapper')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(execute(args.plan)),flush=True)


if __name__=='__main__':main()
