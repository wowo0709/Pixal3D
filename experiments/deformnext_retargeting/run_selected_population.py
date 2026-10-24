"""Own one physical-family population GPU child and close its actual outputs.

The child uses the existing physical-family evaluator unchanged. GPU/CPU
assignment and the canonical archive command are explicit closed inputs.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .collect_module_seed_confirmation import read_file, validate_file
from .native_population_contract import ENGINEERING, FINAL, validate_capture
from .pipeline_population_contract_v2 import required_sources as population_sources
from .process_session_drain import drain_owned_session
from .selected_pipeline_model_contract import require, validate_ss_binding, validate_ss_evaluation


SCHEMA = 'selected-physical-SS-population-execution-v1'
MODULE = 'experiments.deformnext_retargeting.evaluate_pipeline_bound_population_v2'


def required_sources(arm):
    root = Path(__file__).resolve().parent
    return sorted(set(population_sources(arm)) | {str(root/(name+'.py')) for name in
        ['run_selected_population','process_session_drain','selected_pipeline_model_contract']})


def validate_execution(context, population, *, gpu, allowed):
    require(context.get('schema') == SCHEMA, 'explicit physical population execution context required')
    arm = population['module_arm']; validate_ss_binding(population['model_binding'], arm)
    require(population['population_scope'] in (ENGINEERING, FINAL)
        and type(population['expected_assets']) is int
        and population['expected_assets'] == (36 if population['population_scope'] == ENGINEERING else 508)
        and type(population['generation_batch_size']) is int and population['generation_batch_size'] == 8
        and population['deterministic_algorithms'] is True and population['mode'] == 'bound-population',
        'original validation36 or final508 population/sampler required')
    devices = context['assigned_GPUs']; affinity = context['CPU_affinity_per_GPU']
    require(isinstance(devices, list) and devices and len(devices) == len(set(devices))
        and all(isinstance(d, str) and d.isdecimal() for d in devices)
        and set(devices) <= set(population['assigned_GPUs'])
        and isinstance(affinity, dict) and set(affinity) == set(devices)
        and all(isinstance(v, list) and v and len(v) == len(set(v))
            and all(type(c) is int and c >= 0 for c in v) for v in affinity.values())
        and gpu in devices and gpu in allowed.split(',') and gpu.isdecimal(),
        'one actually authorized GPU and explicit CPU affinity required')
    root = Path(context['canonical_local_root'])
    require(root.is_absolute() and context['repo'] == population['repo']
        and context['python'] == population['python']
        and context['canonical_local_root'] == population['canonical_local_root'],
        'same explicit source checkout, Python environment and canonical root required')
    for key in ['repo','python','manifest_directory']:
        require(isinstance(context[key], str) and Path(context[key]).is_absolute(), 'absolute execution paths required')
    require(context['repo'] == str(Path(__file__).resolve().parents[2]), 'execute this actual source checkout')
    for key in ['output','wrapper_output']:
        require(Path(population[key]).is_absolute() and Path(population[key]).is_relative_to(root),
                'all native and wrapper outputs must stay in the assigned canonical root')
    require(population['output'] != population['wrapper_output'], 'separate actual native and wrapper outputs required')
    for key in (['reference_evaluation'] if population['population_scope'] == ENGINEERING else
                ['model_freeze','cache_QA','cache_QA_plan','engineering_parity']):
        validate_file(population[key])
    closure = context['closure_command']
    require(isinstance(closure, list) and len(closure) == 2 and closure[0] == context['python']
        and isinstance(closure[1], str) and Path(closure[1]).is_absolute()
        and closure[1] in context['source_SHA256'], 'pin the exact canonical closure helper command')
    require(type(context['minimum_free_GPU_bytes']) is int and context['minimum_free_GPU_bytes'] > 0,
            'positive measured minimum free GPU memory required')
    require(set(required_sources(arm)) <= set(context['source_SHA256'])
        and set(population_sources(arm)) <= set(population['source_SHA256']),
        'pin the complete physical evaluator and original numerical source chain')
    require(all(context['source_SHA256'][p] == digest for p, digest in population['source_SHA256'].items()
        if p in context['source_SHA256']), 'execution context and population must pin the same numerical sources')
    return [context['python'], '-u', '-m', MODULE, '--plan', context['population_plan']['path']]


def execute(path):
    path = Path(path); context = json.loads(path.read_text()); check = ClosedInputs(context['canonical_local_root'])
    check.check(path, path.parent/'file4-closure.json')
    population = read_file(context['population_plan'], check)
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    command = validate_execution(context, population, gpu=gpu, allowed=os.environ.get('MATCHING_V2_ALLOWED_GPUS', ''))
    os.sched_setaffinity(0, set(context['CPU_affinity_per_GPU'][gpu]))
    native, wrapper = Path(population['output']), Path(population['wrapper_output'])

    def guard():
        check.check(path, path.parent/'file4-closure.json')
        require(read_file(context['population_plan'], check) == population, 'actual population execution plan changed')
        for source, digest in {**context['source_SHA256'], **population['source_SHA256'],
                               **population['input_SHA256']}.items():
            require(sha(source) == digest, 'actual population source/input bytes changed')
        for source, receipt in population['closed_inputs'].items(): check.check(source, receipt)

    def write(p, value):
        with p.open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False)

    def close(folder, label, *, failed=False):
        manifest = Path(context['manifest_directory'])/(context['name']+'-'+label+'-files.json')
        write(manifest, [str(p.relative_to(check.root)) for p in sorted(folder.rglob('*'))
            if p.is_file() and not p.is_symlink() and 'wandb' not in p.parts])
        receipt = folder/('failure-file4-closure.json' if failed else 'file4-closure.json')
        subprocess.run([*context['closure_command'], '--manifest', str(manifest), '--receipt',
            str(receipt.relative_to(check.root)), '--scope',
            'Actual own physical SS population child and drained outputs; no full-cascade or final experiment completion claim'],
            check=True)

    guard()
    require(not native.exists() and not wrapper.exists(), 'preserve existing native and wrapper outputs')
    free = int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True).strip())*2**20
    require(free >= context['minimum_free_GPU_bytes'], 'actual free GPU memory does not meet measured admission')
    wrapper.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=context['repo'], OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    child = None; birth = None; launch_bound = None; error = None; draining = False

    def interrupted(signum, frame):
        if not draining: raise KeyboardInterrupt('owned population evaluation interrupted '+str(signum))

    previous = {s: signal.signal(s, interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    try:
        launch_bound = time.time()
        with (wrapper/'native.log').open('xb') as stream:
            child = subprocess.Popen(command, cwd=context['repo'], env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        birth = psutil.Process(child.pid).create_time()
        write(wrapper/'active.json', dict(PID=child.pid,birth=birth,physical_GPUs=[gpu],command=command))
        child.wait()
    except BaseException as exc:
        error = exc
        if child is not None and child.poll() is None:
            try: child.send_signal(signal.SIGINT)
            except ProcessLookupError: pass
    finally:
        draining = True
        try:
            if child is not None:
                child.wait(); drain_owned_session(child.pid, birth if birth is not None else launch_bound)
                write(wrapper/'terminal.json', dict(PID=child.pid,birth=birth,session_ID=child.pid,
                    physical_GPUs=[gpu],returncode=child.returncode,actual_session_drained=True,
                    session_started_at_bound=launch_bound,owner_birth_verified=birth is not None,
                    parent_error=None if error is None else repr(error)))
        finally:
            for signum, handler in previous.items(): signal.signal(signum, handler)
    if error is not None or child is None or child.returncode != 0:
        write(wrapper/'failure.json', dict(status='FAILED_PHYSICAL_POPULATION_PRESERVED',error=repr(error),
            returncode=None if child is None else child.returncode,automatic_retry=False,full_goal_complete=False))
        if native.exists(): close(native, 'native-failure', failed=True)
        close(wrapper, 'wrapper-failure', failed=True)
        if error is not None: raise error
        raise RuntimeError('actual population child failed; outputs preserved')
    result = json.loads((native/'evaluation.json').read_text())
    audit = json.loads((native/'population-audit.json').read_text())
    completed = json.loads((native/'completed-bound.json').read_text())
    validate_ss_evaluation(result, population['model_binding'], population['module_arm'])
    validate_capture(result, result['native_SS_artifacts']['artifacts'], population['expected_assets'])
    require(result['frozen_native_model_binding'] == population['model_binding']
        and audit['model_binding'] == population['model_binding']
        and completed['status'] == audit['status'] == 'PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION'
        and completed['optimizer_updates'] == audit['optimizer_updates'] == 0
        and audit['assets'] == population['expected_assets'], 'actual own complete population output required')
    guard()
    write(wrapper/'audit.json', dict(status='PASS_ACTUAL_OWN_PHYSICAL_POPULATION_WORKER',
        module_arm=population['module_arm'],population_scope=population['population_scope'],
        assets=population['expected_assets'],model_binding=population['model_binding'],
        actual_owned_session_drained=True,verified_inputs=check.inputs,source_SHA256=context['source_SHA256'],
        full_goal_complete=False))
    close(native, 'native'); close(wrapper, 'wrapper')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(execute(args.plan)), flush=True)


if __name__ == '__main__': main()
