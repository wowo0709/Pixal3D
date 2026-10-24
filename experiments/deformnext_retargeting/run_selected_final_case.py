"""Run one actual selected-pipeline TEST case with closed native phase plans.

Raw data remain available for every declared geometry consumer. A completed
case does not release its disk reservation; the campaign must first verify and
retire only its canonical local duplicates after all consumers have finished.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

from .assigned_gpu_entry import validate_assignment
from .bind_generated_pbr_dataset import load_QA
from .bind_native_surface_reference import SurfaceReferences
from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .collect_module_seed_confirmation import read_file, verify_terminal
from .frozen_cascade_metrics import PinnedFiles
from .metric_image_identity import scored_image_observation
from .native_final_case_integrity import local_surface_join
from .native_run_disk_budget import DiskBudget
from .process_session_drain import drain_owned_session
from .selected_final_case_integrity import CASE_STATUS, validate_case_reports, validate_geometry_reuse
from .selected_final_case_plans import case_plans, required_sources as case_sources
from .selected_final_phase import SCHEMA as PHASE_SCHEMA, ENTRIES
from .selected_final_population import admit_all_ss, compose_models
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


SCHEMA = 'selected-physical-final-case-execution-v1'


def required_sources():
    root = Path(__file__).resolve().parent
    return sorted(set(case_sources()) | {str(root/(name+'.py')) for name in
        ['run_selected_final_case','selected_final_phase','selected_final_case_integrity',
         'native_final_case_integrity','native_run_disk_budget','checkpoint_cache_retirement',
         'process_session_drain','bind_native_surface_reference','metric_image_identity',
         'frozen_cascade_metrics','collect_module_seed_confirmation','bind_generated_pbr_dataset',
         'selected_pipeline_joint_summary_v2','original_zero_population','pbr_replay_artifacts',
         'native_population_contract','evaluate_bound_population','cascade_view_roles','prepare_frozen_cascade_cases']})


def spec(value):
    return {k:value[k] for k in ['path','SHA256']}


def write(path, value):
    with Path(path).open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False)


def validate_execution(context, *, gpu, allowed):
    require(context.get('schema') == SCHEMA,'one explicit selected physical final-case context required')
    validate_assignment(context,gpu,allowed)
    require(context.get('repo') == str(Path(__file__).resolve().parents[2]), 'execute this explicit actual source checkout')
    root = Path(context['canonical_local_root']); output = Path(context['output'])
    require(root.is_absolute() and output.is_absolute() and output.resolve() != root.resolve()
        and output.resolve().is_relative_to(root.resolve()), 'one owned output below the explicit canonical root required')
    require(all(isinstance(context[k],str) and Path(context[k]).is_absolute()
        for k in ['python','manifest_directory','disk_reservations']), 'explicit absolute runtime paths required')
    cpus = context['CPU_affinity_per_GPU']
    require(set(cpus) == set(context['assigned_GPUs'])
        and all(isinstance(v,list) and v and len(v) == len(set(v))
            and all(type(i) is int and i >= 0 for i in v) for v in cpus.values()), 'explicit CPU assignment required')
    for key in ['minimum_free_GPU_bytes','raw_budget_bytes','workspace_bytes']:
        require(type(context.get(key)) is int and context[key] > 0,'positive measured resource admission required')
    require(set(required_sources()) <= set(context['source_SHA256']), 'pin the complete actual final-case executor/source chain')
    closure = context['closure_command']
    require(isinstance(closure,list) and len(closure) == 2 and closure[0] == context['python']
        and Path(closure[1]).is_absolute() and closure[1] in context['source_SHA256'],
        'explicit pinned canonical file4 closure helper required')


def load_case(context, check, pins):
    """Derive plans from actual frozen files; never trust inline model/row aliases."""
    inputs = context['source_inputs']
    parent = read_file(inputs['parent_freeze'],check); controls = read_file(inputs['control_freeze'],check)
    models = compose_models(parent,controls); alias = context['alias']; sid = context['sample_id']
    require(alias in models and sid in parent['population']['sample_ids'], 'one actual frozen model/sample required')
    frozen = parent['evaluation_protocol']
    require(all(frozen['source_SHA256'].get(p) == d for p,d in context['source_SHA256'].items()
        if p != context['closure_command'][1]) and set(required_sources()) <= set(frozen['source_SHA256']),
        'the DEV freeze must pin this actual final executor before any held-out prediction')
    for p,d in frozen['source_SHA256'].items():pins.check(dict(path=p,SHA256=d))
    for model in models.values():
        for stage in [model['ss64'],model['pbr1024'],*model['geometry_stages'].values()]:
            for key in ['native_config','config','checkpoint','conditioner_checkpoint']:
                if stage.get(key) is not None:pins.check(stage[key])
    for value in parent['development_evidence']:pins.check(value)
    require(exact(spec(inputs['manifest']),parent['population']['manifest'])
        and exact(spec(inputs['cache_QA']),parent['development_evidence_roles']['cache_QA']),
        'use the actual frozen manifest and original cache QA')
    rows = read_file(inputs['manifest'],check,lines=True)
    for key in ['cache_QA','cache_QA_plan']:read_file(inputs[key],check)
    load_QA(inputs)
    require([r['sample_id'] for r in rows] == parent['population']['sample_ids']
        and {r['sample_id']:r['source_cluster_id'] for r in rows} == parent['population']['source_clusters_by_sample'],
        'actual ordered508/134-source rows must match the DEV freeze')
    packet = read_file(inputs['asset_packet'],check)
    require(packet['schema'] == 'native-final-cascade-asset-inputs-v1'
        and packet['assets'] == 508 and packet['source_clusters'] == 134
        and [r['sample_id'] for r in packet['records']] == parent['population']['sample_ids'],
        'actual complete sealed508 asset packet required')
    require(set(inputs['SS_evaluations']) == set(models),'all distinct final SS inputs and original controls required')
    records = {name:read_file(value,check) for name,value in inputs['SS_evaluations'].items()}
    admitted = admit_all_ss(records,parent=parent,controls=controls)
    ss_spec = inputs['SS_evaluations'][alias]; artifact = admitted[alias][sid]
    artifact_path = Path(ss_spec['path']).parent/artifact['artifact']
    require(artifact_path.resolve().is_relative_to(Path(ss_spec['path']).parent.resolve()), 'own archived SS artifact required')
    require(check.check(artifact_path,ss_spec['receipt']) == artifact['artifact_SHA256'], 'actual own SS archive bytes changed')
    artifact = dict(artifact,path=str(artifact_path),SHA256=artifact['artifact_SHA256'])
    templates = frozen['final_case_templates']
    require(exact(spec(inputs['geometry_template']),frozen['generation_template'])
        and all(exact(spec(inputs[k+'_template']),templates[k]) for k in ['render','shared']),
        'the original numerical and common rendering templates must be frozen before TEST')
    loaded = {k:read_file(inputs[k+'_template'],check) for k in ['geometry','render','shared']}
    index = parent['population']['sample_ids'].index(sid); row = rows[index]; asset = packet['records'][index]
    model = models[alias]; group = model['geometry_group']
    first = next(name for name,value in models.items() if value['geometry_group'] == group)
    reference = None; owner = None
    if alias != first:
        owner = read_file(inputs['geometry_owner'],check)
        require(owner.get('status') == CASE_STATUS and owner.get('actual_completed_native_case') is True
            and owner.get('alias') == first and owner.get('sample_id') == sid
            and exact(owner.get('parent_freeze'),spec(inputs['parent_freeze']))
            and exact(owner.get('control_freeze'),spec(inputs['control_freeze'])), 'completed own frozen geometry owner required')
        reference = dict(alias=first,sample_id=sid,generation_root=owner['geometry_source_root'],
            SS_evaluation=spec(ss_spec),SS_prediction_voxels_SHA256=artifact['prediction_voxels_SHA256'])
    else:
        require('geometry_owner' not in inputs,'canonical geometry producer must run its own original Shape cascade')
    args = dict(parent=parent,controls=controls,alias=alias,row=row,asset=asset,
        file4_cache_root=packet['file4_cache_root'],assignment=context,SS_evaluation=spec(ss_spec),SS_artifact=artifact,
        geometry_template=loaded['geometry'],geometry_template_spec=spec(inputs['geometry_template']),
        render_template=loaded['render'],shared_template=loaded['shared'],output=context['output'],
        reference_root=context['reference_shared_root'],reference_render_root=context['reference_render_root'],
        geometry_reference=reference)
    if model['kind'] == 'selected-trained':
        mode = model['PBR_mode']
        require(exact(spec(inputs['PBR_template']),templates['PBR_by_mode'][mode]), 'own frozen selected PBR template required')
        args.update(PBR_template=read_file(inputs['PBR_template'],check),
            PBR_config=json.loads(pins.check(model['pbr1024']['native_config']).read_text()),
            PBR_inputs=context['PBR_input_SHA256'],PBR_closed_inputs=context['PBR_closed_inputs'])
    case = case_plans(**args)
    if alias != 'zero-source4':
        material = read_file(inputs['material_reference'],check)
        require(material.get('status') == CASE_STATUS and material.get('alias') == 'zero-source4'
            and material.get('sample_id') == sid and material.get('source_cluster_id') == case['source_cluster_id']
            and material.get('actual_completed_native_case') is True
            and exact(material.get('parent_freeze'),spec(inputs['parent_freeze']))
            and exact(material.get('control_freeze'),spec(inputs['control_freeze']))
            and Path(context['reference_shared_root']) == Path(material['case_output'])/'shared'
            and Path(context['reference_render_root']) == Path(material['case_output'])/'render',
            'one actual completed same-sample original M1 material reference required')
        for terminal in material['terminals']:verify_terminal(terminal)
        for folder in [Path(context['reference_shared_root']),Path(context['reference_render_root'])]:
            for name in ['render-evaluation.json',*[f'camera{cid:02d}.npz' for cid in [8,9,10]]]:
                check.check(folder/name,folder/'file4-closure.json')
    return parent,controls,records[alias],case,owner


def close(folder, context, *, label, failed=False):
    folder = Path(folder); base = Path(context['canonical_local_root']).resolve()
    manifest = Path(context['manifest_directory'])/(context['name']+'-'+label+'-files.json')
    files = [p for p in sorted(folder.rglob('*')) if p.is_file() and p != folder/'file4-closure.json']
    require(not any(p.is_symlink() for p in files),'canonical phase results cannot contain symlinked artifacts')
    write(manifest,[str(p.relative_to(base)) for p in files])
    receipt = folder/('failure-file4-closure.json' if failed else 'file4-closure.json')
    subprocess.run([*context['closure_command'],'--manifest',str(manifest),'--receipt',str(receipt.relative_to(base)),
        '--scope','Actual selected physical final native case phase; no model selection or complete experiment claim'],check=True)


def run_child(context, output, label, command):
    """Keep ownership until the real child session has drained, including failure."""
    output = Path(output); child = None; birth = None; bound = None; error = None; draining = False
    previous = {}
    def interrupted(signum,frame):
        if not draining:raise KeyboardInterrupt('owned selected final phase interrupted '+str(signum))
    for sig in [signal.SIGINT,signal.SIGTERM]:previous[sig] = signal.signal(sig,interrupted)
    env = dict(os.environ,PYTHONPATH=context['repo'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENCV_IO_ENABLE_OPENEXR='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        bound = time.time()
        with (output/(label+'.log')).open('xb') as stream:
            child = subprocess.Popen(command,cwd=context['repo'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        birth = psutil.Process(child.pid).create_time()
        write(output/(label+'-active.json'),dict(PID=child.pid,birth=birth,phase=label,command=command))
        child.wait()
    except BaseException as exc:
        error = exc
        if child is not None and child.poll() is None:
            try:child.send_signal(signal.SIGINT)
            except ProcessLookupError:pass
    finally:
        draining = True
        try:
            if child is not None:
                child.wait(); drain_owned_session(child.pid,birth if birth is not None else bound)
                terminal = dict(PID=child.pid,birth=birth,session_ID=child.pid,returncode=child.returncode,phase=label,
                    command=command,physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),actual_session_drained=True,
                    owner_birth_verified=birth is not None,parent_error=None if error is None else repr(error))
                write(output/(label+'-terminal.json'),terminal)
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
    if error is not None:raise error
    require(child is not None and child.returncode == 0 and birth is not None,'actual selected final phase failed; preserve all outputs')
    verify_terminal(terminal)
    return terminal


def execute(path):
    path = Path(path); context = json.loads(path.read_text()); gpu = os.environ.get('CUDA_VISIBLE_DEVICES','')
    validate_execution(context,gpu=gpu,allowed=os.environ.get('MATCHING_V2_ALLOWED_GPUS',''))
    os.sched_setaffinity(0,set(context['CPU_affinity_per_GPU'][gpu]))
    check = ClosedInputs(context['canonical_local_root']); check.check(path,path.parent/'file4-closure.json')
    pins = PinnedFiles()
    for p,d in context['source_SHA256'].items():pins.check(dict(path=p,SHA256=d))
    parent,controls,ss,case,owner = load_case(context,check,pins)
    root = Path(context['output']); require(not root.exists(),'preserve any previous completed or failed case')
    free = int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20
    require(free >= context['minimum_free_GPU_bytes'],'actual free GPU memory does not meet measured admission')
    disk = DiskBudget(context['disk_reservations'],workspace_bytes=context['workspace_bytes'])
    admission = disk.reserve(context['name'],checkpoint_budget_bytes=context['raw_budget_bytes'],native_output=str(root),
        wrapper_output=str(root),owner=dict(PID=os.getpid(),birth=psutil.Process(os.getpid()).create_time()))
    root.mkdir(parents=True,exist_ok=False); write(root/'disk-admission.json',admission); write(root/'case-plan.json',case)
    reports = {}; terminals = []; surface = None; started = time.monotonic()
    try:
        for phase in case['phases']:
            label = phase['name']; native = deepcopy(phase['plan'])
            if phase['kind'] == 'reuse-closed-native-guides':
                require(owner is not None,'actual geometry owner required')
                for terminal in owner['terminals']:verify_terminal(terminal)
                folder = Path(case['geometry_source_root'])
                for filename in ['generation.json','prediction.npz','geometry-evaluation.npz','native-shape-guides.pt']:
                    check.check(folder/filename,folder/'file4-closure.json')
                report = json.loads((folder/'generation.json').read_text())
                validate_geometry_reuse(case,case['model'],owner,models=compose_models(parent,controls),
                    prediction_SHA256=report['SS_prediction_SHA256'],guide_SHA256=sha(folder/'native-shape-guides.pt'))
            else:
                directory = root/'runtime-plans'/label; directory.mkdir(parents=True,exist_ok=False)
                if label == 'PBR':
                    for name in ['generation.json','prediction.npz','geometry-evaluation.npz','native-shape-guides.pt']:
                        source = Path(case['geometry_source_root'])/name
                        digest = check.check(source,source.parent/'file4-closure.json')
                        native['closed_inputs'][str(source)] = str(source.parent/'file4-closure.json')
                        native['input_SHA256'][str(source)] = digest
                native_path = directory/'native.json'; write(native_path,native)
                kind = phase['kind']
                if label == 'shared':kind = 'original-shared-material' if case['alias'] == 'zero-source4' else 'fixed-shared-material'
                entry = Path(__file__).with_name(ENTRIES[kind]+'.py')
                binding = dict(schema=PHASE_SCHEMA,kind=kind,native_plan=dict(path=str(native_path),SHA256=sha(native_path)),
                    native_source=dict(path=str(entry),SHA256=context['source_SHA256'][str(entry)]),
                    phase_source_SHA256=context['source_SHA256'][str(Path(__file__).with_name('selected_final_phase.py'))])
                if kind == 'selected-zero-bound-entry':
                    binding.update(**{k:spec(context['source_inputs'][k]) for k in ['parent_freeze','control_freeze']},
                        SS_evaluation=case['SS_evaluation'])
                entry_path = directory/'entry.json'; write(entry_path,binding); close(directory,context,label=label+'-plans')
                command = [context['python'],'-u','-m','experiments.deformnext_retargeting.selected_final_phase','--plan',str(entry_path)]
                terminals.append(run_child(context,root,label,command))
                folder = Path(native['output']); report = json.loads((folder/phase['result']).read_text())
            reports[label] = report
            if label == 'geometry':
                references = SurfaceReferences(context['GT_surface_reference'],receipt=context['GT_surface_reference_receipt'],
                    closed=check,ids=parent['population']['sample_ids'],clusters=parent['population']['source_clusters_by_sample'])
                surface = local_surface_join(native,report,geometry_path=folder/'geometry-evaluation.npz',references=references)
                if not case['geometry_reference']:write(folder/'shared-surface-binding.json',surface)
            if phase['kind'] != 'reuse-closed-native-guides':close(folder,context,label=label+'-result')
        final = reports.get('PBR',reports['geometry'])
        audit = validate_case_reports(case,parent=parent,controls=controls,SS_evaluation=ss,geometry=reports['geometry'],
            final=final,render=reports['render'],shared=reports['shared'],surface=surface,terminals=terminals,geometry_reference_audit=owner)
        from PIL import Image
        import numpy as np
        images = []; observations = []
        for cid in [8,9,10]:
            source = root/'shared'/f'camera{cid:02d}.npz'; check.check(source,source.parent/'file4-closure.json')
            with np.load(source,allow_pickle=False) as arrays:
                pixels = {k:arrays[k].copy() for k in ['prediction','prediction_alpha','GT_shared_renderer','GT_alpha']}
            observations.append(scored_image_observation(pixels,sample_id=case['sample_id'],camera_id=cid,
                source_cluster_id=case['source_cluster_id'],archive=str(source),archive_SHA256=sha(source)))
            gt,pred = pixels['GT_shared_renderer'],pixels['prediction']; images.append(np.concatenate([gt,pred,abs(gt-pred)],axis=2))
        Image.fromarray((np.clip(np.concatenate(images,axis=1).transpose(1,2,0),0,1)*255).astype('uint8')).save(root/'GT-Pred-Diff.png')
        for p,d in check.inputs.items():require(sha(p) == d,'closed input changed during actual native case')
        pins.finish(); check.check(path,path.parent/'file4-closure.json')
        audit.update(parent_freeze=spec(context['source_inputs']['parent_freeze']),control_freeze=spec(context['source_inputs']['control_freeze']),
            case_output=str(root),raw_archive_roots=case['raw_archive_roots'],final_generation_root=case['final_generation_root'],
            scored_image_observations=observations,actual_shared_GT_material_pixel_identity_in_observations=True,
            elapsed_seconds=time.monotonic()-started,verified_inputs=check.inputs,source_SHA256=context['source_SHA256'],
            disk_reservation_release_deferred_until_all_raw_consumers_and_canonical_retirement=True)
        write(root/'audit.json',audit); close(root,context,label='case')
        return audit
    except BaseException as exc:
        write(root/'failure.json',dict(status='FAILED_SELECTED_FINAL_CASE_PRESERVED',error=repr(exc),terminals=terminals,
            automatic_retry=False,disk_reservation_released=False,full_goal_complete=False))
        close(root,context,label='failure',failed=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',type=Path,required=True)
    args = parser.parse_args(); result = execute(args.plan)
    print(json.dumps(dict(status=result['status'],alias=result['alias'],sample_id=result['sample_id'],full_goal_complete=False)),flush=True)


if __name__ == '__main__': main()
