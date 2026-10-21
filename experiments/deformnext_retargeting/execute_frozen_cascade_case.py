"""Execute one frozen native cascade through bounded file4 raw-archive IO.

Each numerical phase runs in its own original native subprocess. Large NPZ
archives go straight to the explicitly assigned immutable store. The caller
provides the same canonical closure command used for its other campaign jobs.
Every phase and final case get separate immutable completion receipts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from .evaluate_bound_population import pinned, sha
from .frozen_cascade_metrics import PinnedFiles, verify_texture_replay, verify_test_PBR_binding
from .immutable_result_store import relative_key
from .bind_generated_pbr_dataset import load_QA, validate_binding
from .summarize_frozen_inception_pools import validate_frozen_models


def io_binding(phase, native, *, store, prefix, compressed):
    """Bind only completed producer manifests; never predict their checksums."""
    inputs={}
    for root in phase['input_archive_roots']:
        path=Path(root)/'remote-npz.json'
        record=json.loads(path.read_text())
        if record['remote_root']!=store['remote_root'] or record['schema']!='native-cascade-remote-npz-v1':
            raise ValueError('completed native producer store/schema differs')
        inputs[root]=sha(path)
    result=dict(stage=phase['remote_stage'],native_plan=phase['native_plan']['path'],
        native_plan_SHA256=phase['native_plan']['SHA256'],native_source_SHA256=phase['native_source']['SHA256'],
        store=store,output_prefix=relative_key(prefix),compressed=compressed,input_manifests=inputs)
    if phase['remote_stage']=='PBR-replay':
        module='experiments.deformnext_retargeting.pbr_remote_results'
    elif phase['remote_stage']=='zero-generation' and 'bound_zero_shot' in native:
        module='experiments.deformnext_retargeting.bound_zero_shot_cascade'
        result['bound_zero_adapter_source_SHA256']=sha(Path(__file__).with_name('bound_zero_shot_cascade.py'))
    else:
        module='experiments.deformnext_retargeting.cascade_remote_results'
    return result,module


def close(folder, *, execution, tag):
    base=Path(execution['canonical_local_root']).resolve()
    manifest=folder/(tag+'-files.json')
    entries=[str(p.relative_to(base)) for p in folder.rglob('*') if p.is_file() and p.name!='file4-closure.json']
    # The closure input listing itself is metadata, not an archive to recursively
    # include while it is being written.
    entries=[p for p in entries if base/p!=manifest]
    manifest.write_text(json.dumps(entries,indent=2)+'\n')
    subprocess.run([*execution['closure_command'],'--manifest',str(manifest),
        '--receipt',str((folder/'file4-closure.json').relative_to(base)),
        '--scope','Frozen final508 native cascade '+tag+'; no model selection or training'],check=True)


def execute(case_spec, execution):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if (not visible.isdigit() or visible not in execution['allowed_physical_GPUs']
            or visible not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(',')):
        raise ValueError('one explicitly assigned physical GPU required')
    if not isinstance(execution['compressed'],bool):raise ValueError('explicit archive compression policy required')
    if Path(execution['source_root']).resolve()!=Path(__file__).resolve().parents[2]:
        raise ValueError('execute inside the explicitly compiled native source checkout')
    if pinned(execution['executor_source']).resolve()!=Path(__file__).resolve():
        raise ValueError('explicit compiled executor source binding required')
    if 'CPU_affinity_by_GPU' in execution:
        os.sched_setaffinity(0,set(execution['CPU_affinity_by_GPU'][visible]))
    case_path=pinned(case_spec);case=json.loads(case_path.read_text())
    if case['source_inputs']['scope']!='final-test508':raise ValueError('explicit frozen final-test508 case required')
    pinned(case['model_freeze'])
    dependency=case['reference_dependency']
    if dependency:
        proof=json.loads(Path(dependency).read_text())
        if proof['status']!='PASS' or not proof['local_equals_actual_file4_SHA256']:
            raise ValueError('the shared original-material reference is not canonically completed')
    pins=PinnedFiles();terminals=[]
    freeze=json.loads(pins.check(case['model_freeze']).read_text())
    ids,clusters,_=validate_frozen_models(freeze,freeze['models'])
    if (case['sample_id'] not in ids or case['source_cluster_id']!=clusters[case['sample_id']]
            or case['model_freeze']!=case['source_inputs']['model_freeze']):
        raise ValueError('case must belong to the exact pre-test frozen model population')
    load_QA(case['source_inputs'])
    model=freeze['models'][case['alias']]
    def frozen_files(value):
        if isinstance(value,dict):
            if {'path','SHA256'}.issubset(value):
                pins.check(value)
            else:
                for child in value.values():frozen_files(child)
        elif isinstance(value,list):
            for child in value:frozen_files(child)
    frozen_files(model)
    if model.get('pbr_dataset_binding') is not None:
        binding=validate_binding(model['pbr_dataset_binding'],conditioner_config=model['pbr_conditioner_config'],
            **{key:case['source_inputs'][key] for key in ['cache_QA','cache_QA_plan','manifest']})
        frozen_files(binding)
    pins.check(case['source_inputs']['SS_evaluations'][case['alias']])
    # Per-sample output root is explicit in the first native generation plan.
    first=json.loads(pins.check(case['phases'][0]['native_plan']).read_text())
    root=Path(first['output']).parent
    if not root.resolve().is_relative_to(Path(execution['canonical_local_root']).resolve()):
        raise ValueError('native outputs must stay inside the assigned canonical results cache')
    if root.exists():raise FileExistsError('preserve existing case/failed outputs')
    root.mkdir(parents=True,exist_ok=False)
    input_guard=case['source_inputs']
    for key in ['model_freeze','manifest','cache_QA','cache_QA_plan','asset_packet']:pins.check(input_guard[key])
    started=time.monotonic()
    for phase in case['phases']:
        path=pins.check(phase['native_plan']);pins.check(phase['native_source']);pins.check(phase['execution_source'])
        native=json.loads(path.read_text());destination=Path(native['output'])
        if not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError('native phase output must belong to its own case')
        if destination.exists():raise FileExistsError('preserve previous native phase output')
        for source,digest in {**native.get('inputs_SHA256',{}),**native.get('input_SHA256',{}),**native.get('source_SHA256',{})}.items():
            pins.check(dict(path=source,SHA256=digest))
        destination.parent.mkdir(parents=True,exist_ok=True)
        module=phase['execution_module'];actual_plan=path;binding_path=None
        if phase['remote_stage'] is not None:
            prefix=execution['raw_prefix']+'/'+case['alias']+'/'+case['sample_id']+'/'+phase['name']
            bound,module=io_binding(phase,native,store=execution['store'],prefix=prefix,compressed=execution['compressed'])
            if module!=phase['execution_module']:raise ValueError('compiled native archive adapter differs')
            binding_path=root/(phase['name']+'-IO.json');binding_path.write_text(json.dumps(bound,indent=2)+'\n')
            actual_plan=binding_path
        command=[execution['python'],'-u','-m',module,'--plan',str(actual_plan)]
        with (root/(phase['name']+'.log')).open('xb') as stream:
            child=subprocess.Popen(command,cwd=execution['source_root'],env=dict(os.environ,
                PYTHONPATH=execution['source_root'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                OPENCV_IO_ENABLE_OPENEXR='1'),stdout=stream,stderr=subprocess.STDOUT)
            birth=psutil.Process(child.pid).create_time()
            code=child.wait()
        terminal=dict(phase=phase['name'],native_PID=child.pid,birth=birth,returncode=code,physical_GPU=visible,command=command,
                      native_source=phase['native_source'],native_plan=phase['native_plan'])
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
        terminals.append(terminal)
        if code:
            (root/'failure.json').write_text(json.dumps(dict(failed_phase=phase['name'],terminal=terminal),indent=2)+'\n')
            raise RuntimeError('native phase failed; outputs preserved: '+phase['name'])
        result=json.loads((destination/phase['result']).read_text())
        if not result['status'].startswith('PASS_'):raise ValueError('native phase did not pass: '+phase['name'])
        if phase['name']=='generation':
            if (result['actual_resolution']!=1024 or result['sample_id']!=case['sample_id']
                    or result['GT_geometry_or_coords_used_in_generation'] or result['per_output_alignment']):
                raise ValueError('actual native predicted geometry/sample differs')
        elif phase['name']=='PBR':
            original=json.loads((Path(native['generated_shape_root'])/'generation.json').read_text())
            freeze_value=freeze['population']['texture_seeds_by_sample'][case['sample_id']]
            verify_texture_replay(native,result,original,model,seed=freeze_value)
            verify_test_PBR_binding(native,model,**{key:input_guard[key] for key in ['cache_QA','cache_QA_plan','manifest']})
        elif phase['name']=='render':
            if (len(result['per_view'])!=3 or any(v['GT_geometry_RGBA_containment']<.99 for v in result['per_view'])
                    or result['prediction_geometry_postprocess']['render_faces']>1000000):
                raise ValueError('native camera alignment or shared render geometry budget failed')
        close(destination,execution=execution,tag=phase['name'])
    pins.finish();pinned(case_spec);pinned(execution['executor_source'])
    result=dict(status='PASS_ACTUAL_FROZEN_NATIVE_CASCADE_CASE',sample_id=case['sample_id'],alias=case['alias'],
        source_cluster_id=case['source_cluster_id'],terminals=terminals,elapsed_seconds=time.monotonic()-started,
        model_freeze=case['model_freeze'],no_training_or_model_selection=True,full_goal_complete=False)
    (root/'case-completed.json').write_text(json.dumps(result,indent=2)+'\n')
    close(root,execution=execution,tag='case')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--execution',type=Path,required=True);parser.add_argument('--execution-sha256',required=True)
    args=parser.parse_args()
    execution=json.loads(pinned(dict(path=str(args.execution),SHA256=args.execution_sha256)).read_text())
    result=execute(dict(path=str(args.plan),SHA256=args.plan_sha256),execution)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
