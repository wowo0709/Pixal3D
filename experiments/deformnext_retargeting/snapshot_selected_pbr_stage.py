"""Snapshot one genuinely completed native PBR stage with its actual TRAIN seed.

Reads saved artifacts only. No model construction, neural forward, resampling,
seed alias, or inference on a different geometric support is performed here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from .cascade_view_roles import ClosedInputs,file_sha256 as sha
from .collect_module_seed_confirmation import read_file,require,verify_terminal
from .collect_selected_pbr_seed_confirmation import (
    validate_trace,training_trace_digest,required_sources as confirmation_sources,
)
from .common_geometry_pbr_contract import validate_stage
from .native_pbr_stage_summary import STEPS,validate_curve
from .pbr_confirmation_initial_identity import read_initial_identity
from .selected_pipeline_model_contract import pbr_binding


SCHEMA='selected-native-PBR-single-stage-snapshot-inputs-v1'


def validate_completion(cfg,mode,proof,*,training_seed):
    require(type(training_seed) is int and training_seed in (101,202)
        and type(cfg['training_seed']) is int and cfg['training_seed']==training_seed,
        'explicit actual integer TRAIN seed must agree with the saved native config')
    validate_stage(cfg,mode,proof['ranks'],training_seed=training_seed)
    a,t,w,p=(proof[k] for k in ['active','terminal','wrapper','progress'])
    require(type(a.get('PID')) is int and a['PID']>0 and type(a.get('birth')) in (int,float)
        and math.isfinite(a['birth']) and a['birth']>0
        and a['PID']==t.get('PID') and a['birth']==t.get('birth')
        and a.get('mode')==mode and a.get('physical_GPUs')==t.get('physical_GPUs')==['0','1','2','3']
        and type(t.get('returncode')) is int and t['returncode']==0
        and p.get('stage')=='completed' and p.get('current')==1900
        and w.get('status')=='PASS_ACTUAL_FRESH_NATIVE_PBR1900_SCIENCE'
        and w.get('mode')==mode and w.get('source_multiview_step')==16850
        and w.get('fresh_optimizer') is True and w.get('actual_optimizer_updates')==1900
        and (w.get('world_size'),w.get('global_batch'),w.get('microbatch'),w.get('accumulation'))==(4,24,6,1)
        and w.get('actual_five_epoch_NN_source_Adam_canonical_archives_verified') is True,
        'own actual successful four-rank native1900 completion and matching child identity required')
    if training_seed==202:
        require(t.get('actual_session_drained') is True,'actual seed202 producer session must be drained')


def validate_runtime(rows,rank):
    require(type(rank) is int and rank in range(4) and len(rows)==1900
        and [r['step'] for r in rows]==list(range(1,1901)), 'actual complete per-rank runtime1900 required')
    for row in rows:
        require(type(row['step']) is int
            and (row['rank'],row['world_size'],row['effective_batch_size'],
                 row['accumulation_packs'],row['microbatch_assets'],row['K'])==(rank,4,24,1,[6],4)
            and math.isfinite(row['loss']['loss']) and math.isfinite(row['compute_seconds'])
            and row['compute_seconds']>=0,
            'same actual DDP4/global24/micro6/accum1/MV4 runtime and finite loss required')


def snapshot(plan,check,job):
    require(plan.get('schema')==SCHEMA and type(plan.get('training_seed')) is int,
        'one explicit actual native stage snapshot plan required')
    require(job.get('status')=='completed' and type(job.get('returncode')) is int
        and job['returncode']==0,'actual successful science job must precede its snapshot')
    mode,seed=plan['mode'],plan['training_seed']
    stage=read_file(plan['stage_plan'],check)
    require(stage['mode']==mode,'actual native science plan has a different mode')
    native,wrapper=Path(stage['output']),Path(stage['wrapper_output'])
    def file(path,receipt=None):
        path=Path(path);receipt=path.parent/'file4-closure.json' if receipt is None else Path(receipt)
        return dict(path=str(path),SHA256=check.check(path,receipt),receipt=str(receipt))
    def read(path,receipt=None,*,lines=False):
        spec=file(path,receipt)
        return read_file(spec,check,lines=lines),spec
    nr,wr=native/'file4-closure.json',wrapper/'file4-closure.json'
    cfg,config_spec=read(native/'config.json',nr)
    declared,_=read(stage['config'])
    require(cfg==dict(declared,config=stage['config']) and cfg['output_dir']==str(native),
        'actual saved config must equal this original scientific run plan')
    proof={};completion_specs={}
    for name,path in [('active',wrapper/'active.json'),('terminal',wrapper/'terminal.json'),
                      ('wrapper',wrapper/'audit.json'),('progress',native/'progress.json')]:
        proof[name],completion_specs[name]=read(path,wr if name!='progress' else nr)
    ranks={}
    proof['ranks']=[]
    for rank in range(4):
        value,spec=read(native/f'stage-completed.rank{rank}.json',nr)
        proof['ranks'].append(value);ranks[str(rank)]=spec
    validate_completion(cfg,mode,proof,training_seed=seed)
    absence=verify_terminal(proof['terminal'])
    rows,validation_spec=read(native/'validation.jsonl',nr,lines=True)
    curve=validate_curve(rows)
    initial,evidence=read_initial_identity(native,plan['stage_plan']['path'],rows[0],check)
    model_specs={'config':config_spec,'conditioner_checkpoint':None}
    model_files={}
    archive=native/'checkpoint-archives/step0001900'
    descriptor,_=read(archive/'checkpoint.json')
    require(descriptor['mode']==mode and descriptor['training_seed']==seed and descriptor['step']==1900,
        'actual final archive descriptor must retain this native TRAIN seed and mode')
    for name in ['denoiser','misc']+([] if mode=='vanilla' else ['deform_conditioner']):
        path=native/'ckpts'/f'{name}_step0001900.pt'
        spec=file(path,archive/'file4-closure.json')
        require(str(path.relative_to(check.root)) in descriptor['checkpoint_files'],
            'actual final checkpoint must be named in its own epoch archive')
        model_files[name]=dict(path=str(path),SHA256=spec['SHA256'],bytes=path.stat().st_size)
        if name!='misc':model_specs['checkpoint' if name=='denoiser' else 'conditioner_checkpoint']=spec
    hashes={spec['path']:spec['SHA256'] for spec in model_specs.values() if spec is not None}
    binding=pbr_binding(cfg,{str(row['rank']):row for row in proof['ranks']},hashes)
    traces,trace_specs,runtime={},{},{}
    for rank in range(4):
        suffix='' if rank==0 else f'.rank{rank}'
        values,spec=read(native/f'training-noise-trace{suffix}.jsonl',nr,lines=True)
        validate_trace(values);traces[str(rank)]=training_trace_digest(values);trace_specs[str(rank)]=spec
        values,_=read(native/f'runtime_batch{suffix}.jsonl',nr,lines=True)
        validate_runtime(values,rank)
        runtime[str(rank)]=dict(actual_updates=1900,
            median_last10_compute_seconds=statistics.median(r['compute_seconds'] for r in values[-10:]),
            peak_allocated_GiB=max(r['peak_allocated_gib'] for r in values),
            peak_reserved_GiB=max(r['reserved_gib'] for r in values))
    from PIL import Image
    pixels={}
    for step,row in zip(STEPS,rows):
        per={}
        for index,event in enumerate(row['quick_per_asset']):
            path=native/'validation-panels'/f'step{step:07d}'/f'case{index:03d}.png'
            file(path,nr)
            with Image.open(path) as opened:
                image=opened.convert('RGB');width,height=image.size
                require(width==3*height,'actual GT/Pred/Difference panel layout required')
                per[event['sample_id']]=hashlib.sha256(image.crop((0,0,height,height)).tobytes()).hexdigest()
        pixels[str(step)]=per
        require(per==pixels['0'],'actual fixed GT PNG pixels changed between epochs')
    record=dict(status='PASS_ACTUAL_NATIVE_PBR1900_FIXED4_STAGE_SNAPSHOT',
        schema='native-PBR-stage-explicit-training-seed-snapshot-v2',mode=mode,training_seed=seed,
        native_output=str(native),science_job=plan['science_job'],science_plan=plan['stage_plan']['path'],
        actual_science_job_terminal=job,completion_proof=proof,actual_terminal_evidence=absence,
        model_binding=binding,model_files=model_files,validation_rows=rows,curve_summary=curve,
        runtime_summary=runtime,training_trace_digests=traces,training_trace_inputs=trace_specs,
        GT_PNG_pixel_SHA256=pixels,initial_render_identity=initial,initial_identity_evidence=evidence,
        same_fixed_GT_shape_RGB_noise_and_camera_all_six_epochs=True,
        PSNR_is_pooled_MSE_not_mean_asset_dB=True,source_preservation=False,full_cascade=False,
        no_NN_forward_backward_or_training_update=True,verified_inputs=dict(check.inputs),
        quality_gain_proven=False,model_adopted=False,full_goal_complete=False)
    item=dict(model_specs,snapshot=None,completion_files=completion_specs,rank_completions=ranks,
        training_traces=trace_specs,validation_log=validation_spec)
    return record,item


def required_sources():
    root=Path(__file__).resolve().parent
    return sorted(set(confirmation_sources())|{str(root/(name+'.py')) for name in ['snapshot_selected_pbr_stage',
        'pbr_confirmation_initial_identity','pbr_initial_render_identity','collect_selected_pbr_seed_confirmation',
        'collect_module_seed_confirmation','selected_pipeline_model_contract','common_geometry_pbr_contract',
        'native_pbr_stage_summary','cascade_view_roles','native_final_freeze','pipeline_population_contract_v2']})


def execute(plan_path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit CPU-only stage reader required')
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text())
    check=ClosedInputs(plan['canonical_local_root']);check.check(plan_path,plan_path.parent/'file4-closure.json')
    require(set(required_sources())<=set(plan['source_SHA256']),'pin every actual stage reader dependency')
    for path,digest in plan['source_SHA256'].items():require(sha(path)==digest,'stage snapshot source changed')
    jobs=json.loads(Path(plan['queue_state']).read_text())
    require(plan['science_job'] in jobs,'actual scientific job missing from the recorded queue')
    result,item=snapshot(plan,check,jobs[plan['science_job']])
    for path,digest in check.inputs.items():require(sha(path)==digest,'closed input changed during snapshot')
    for path,digest in plan['source_SHA256'].items():require(sha(path)==digest,'stage source changed during snapshot')
    output=Path(plan['output']);output.relative_to(check.root);output.mkdir(parents=True,exist_ok=False)
    path=output/'audit.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    item['snapshot']=dict(path=str(path),SHA256=sha(path),receipt=str(output/'file4-closure.json'))
    (output/'model-inputs.json').write_text(json.dumps(item,indent=2,allow_nan=False)+'\n')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();output=execute(args.plan)
    print(json.dumps(dict(output=str(output),actual_file4_closure_still_required=True,full_goal_complete=False)))


if __name__=='__main__':main()
