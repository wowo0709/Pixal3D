"""Read closed SS replication artifacts and compare actual frozen tensor data.

This entry performs no model construction, neural forward, training, selection,
or GPU launch. It reduces the measured candidate and deduplicated controls.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re

import psutil

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .pipeline_training_seed_summary_v2 import models_for_selected, summarize, validate_evaluations
from .pipeline_population_contract_v2 import model_binding, required_sources as population_sources


SCHEMA = 'selected-pipeline-SS-two-seed-closed-inputs-v2'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_file(spec, *, closed=True):
    fields = {'path', 'SHA256', 'receipt'} if closed else {'path', 'SHA256'}
    require(isinstance(spec, dict) and set(spec) == fields
        and isinstance(spec['path'], str) and Path(spec['path']).is_absolute()
        and isinstance(spec['SHA256'], str)
        and re.fullmatch('[0-9a-f]{64}', spec['SHA256']) is not None,
        'explicit absolute file and lowercase actual SHA256 required')
    if closed:
        require(isinstance(spec['receipt'], str) and Path(spec['receipt']).is_absolute(),
                'explicit actual file4 receipt required')


def read_file(spec, check, *, lines=False):
    validate_file(spec)
    require(check.check(spec['path'], spec['receipt']) == spec['SHA256'],
            'declared file hash differs from actual closed bytes')
    data = Path(spec['path']).read_text()
    return [json.loads(line) for line in data.splitlines() if line.strip()] if lines else json.loads(data)


def verify_terminal(terminal):
    """Verify child absence without guessing a legacy child's session ID."""
    require(type(terminal.get('returncode')) is int and terminal['returncode'] == 0
        and type(terminal.get('PID')) is int and terminal['PID'] > 0
        and type(terminal.get('birth')) in (int, float) and math.isfinite(terminal['birth'])
        and terminal['birth'] > 0 and terminal.get('actual_session_drained', True) is True,
        'successful owned terminal with actual PID/birth required')
    pid = terminal['PID']
    session = terminal.get('session_ID')
    require(session is None or type(session) is int and session > 0,
            'explicit recorded session ID must be a positive integer')
    for process in psutil.process_iter(['pid', 'create_time']):
        try:
            require(not (process.pid == pid and process.create_time() == terminal['birth']),
                    'reported terminal still has its actual live child')
            if session is not None:
                require(os.getsid(process.pid) != session, 'recorded owned session still has a process')
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
    return dict(PID=pid, birth=terminal['birth'], actual_child_absent=True,
                recorded_session_ID=session, actual_recorded_session_absent=None if session is None else True,
                producer_recorded_session_drain=terminal.get('actual_session_drained'),
                legacy_session_absence_not_inferred_from_child_PID=session is None)


def frozen_base_comparison(original, checkpoint):
    """Compare mmap-loaded CPU tensor data; permit learned LoRA changes only."""
    import torch
    require(isinstance(original, dict) and isinstance(checkpoint, dict)
        and set(checkpoint) == set(original) and original,
        'original and final denoiser parameter keys must be identical')
    frozen = [key for key in original if 'lora_' not in key]
    learned = [key for key in original if 'lora_' in key]
    require(frozen and learned, 'actual base and LoRA tensors required')
    for key, value in original.items():
        other = checkpoint[key]
        require(isinstance(value, torch.Tensor) and isinstance(other, torch.Tensor)
            and value.device.type == other.device.type == 'cpu'
            and value.dtype == other.dtype and value.shape == other.shape,
            'actual CPU tensor structure and dtype must equal original: '+key)
        if key in frozen:
            require(torch.equal(value, other), 'frozen original NN tensor changed: '+key)
    return dict(actual_original_frozen_NN_base_tensors_exact=True,
                frozen_tensors=len(frozen), LoRA_tensors=len(learned),
                CPU_neural_forward_or_training_updates=0)


def collect_seed(records, *, training_seed, selected, check, original_spec, original):
    """Read one actually finished seed; this alone proves no replication."""
    arms = models_for_selected(selected)
    require(type(training_seed) is int and training_seed in (101, 202)
        and set(records) == set(arms), 'own seed and exact deduplicated model set required')
    configs, evaluations, traces, starts, evidence = ({} for _ in range(5))
    import torch
    for arm in arms:
        record = records[arm]
        cfg = read_file(record['config'], check)
        bundle = read_file(record['bundle'], check)
        require(type(cfg['training_seed']) is int and cfg['training_seed'] == training_seed
            and cfg['trainer']['args']['continuation_ckpt']['denoiser'] == original_spec['path'],
            'same actual original multiview initializer and own training seed required')
        hashes = {}
        for key in ('native_config', 'checkpoint', 'conditioner_checkpoint'):
            spec = record['config' if key == 'native_config' else key]
            require((spec is None) == (key == 'conditioner_checkpoint' and arm == 'V1'),
                    'physical source checkpoint must match actual module arm')
            if spec is not None:
                validate_file(spec)
                require(spec['path'] == bundle[key]
                    and check.check(spec['path'], spec['receipt']) == spec['SHA256'],
                    'bundle must use the actual own closed checkpoint/config bytes')
                hashes[spec['path']] = spec['SHA256']
        own = model_binding(bundle, cfg, hashes, arm=arm)
        starts[arm] = read_file(record['startup'], check)
        traces[arm] = read_file(record['training_trace'], check, lines=True)
        terminal_proofs = [verify_terminal(read_file(record['training_terminal'], check))]
        require(len(record['evaluations']) == 3, 'three actual generation-noise records required')
        rows = []
        for item in record['evaluations']:
            result = read_file(item['evaluation'], check)
            terminal_proofs.append(verify_terminal(read_file(item['terminal'], check)))
            require(result['native_task_checkpoint_bundle'] == record['bundle']['path'],
                    'evaluation belongs to another checkpoint bundle')
            for path, digest in hashes.items():
                require(result.get('verified_inputs', {}).get(path) == digest,
                        'evaluation claims different checkpoint/source/config bytes')
            rows.append(result)
        validate_evaluations(rows, cfg, arm)
        current = torch.load(record['checkpoint']['path'], map_location='cpu', weights_only=True, mmap=True)
        proof = frozen_base_comparison(original, current)
        require(proof['LoRA_tensors'] == 360, 'actual native denoiser requires all360 LoRA tensors')
        del current
        configs[arm], evaluations[arm] = cfg, rows
        evidence[arm] = dict(model_binding=own, frozen_base=proof, terminal_proofs=terminal_proofs)
    return evaluations, configs, traces, starts, evidence


def required_sources(selected):
    directory = Path(__file__).resolve().parent
    paths = {path for arm in models_for_selected(selected) for path in population_sources(arm)}
    paths.update(str(directory/(name+'.py')) for name in ('collect_pipeline_seed_confirmation_v2',
        'pipeline_training_seed_summary_v2', 'pipeline_confirmation_recipe_v2',
        'module_confirmation_recipe'))
    return sorted(paths)


def collect(plan, check):
    require(plan.get('schema') == SCHEMA and set(plan['models']) == {'101', '202'},
            'both actual training seeds in a closed confirmation plan required')
    selected = plan['selected_arm']
    models_for_selected(selected)
    original_spec = plan['original_checkpoint']
    validate_file(original_spec, closed=False)
    require(sha(original_spec['path']) == original_spec['SHA256'], 'original checkpoint bytes changed')
    import torch
    torch.set_num_threads(2)
    original = torch.load(original_spec['path'], map_location='cpu', weights_only=True, mmap=True)
    collected = {seed: collect_seed(plan['models'][str(seed)], training_seed=seed,
        selected=selected, check=check, original_spec=original_spec, original=original) for seed in (101, 202)}
    del original
    results, configs, traces, starts, evidence = ({seed: collected[seed][i] for seed in collected}
                                                for i in range(5))
    statistics = summarize(results, configs, traces, starts, selected=selected)
    require(sha(original_spec['path']) == original_spec['SHA256'], 'original checkpoint changed during audit')
    for path, digest in check.inputs.items():
        require(sha(path) == digest, 'closed input changed during reduction')
    return dict(status='PASS_ACTUAL_CLOSED_SELECTED_PIPELINE_SS_TWO_SEED_DEV128X3',
        statistics=statistics, actual_frozen_base_and_model_bytes_verified=True,
        original_checkpoint=original_spec, model_evidence=evidence, verified_inputs=dict(check.inputs),
        CPU_neural_forward_or_training_updates=0, candidate_selection_not_proven_here=True,
        no_GPU_execution_or_TEST_authorization=True, full_goal_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit CPU-only metadata process required')
    plan = json.loads(args.plan.read_text())
    check = ClosedInputs(plan['canonical_local_root'])
    check.check(args.plan, args.plan.parent/'file4-closure.json')
    require(set(required_sources(plan['selected_arm'])) <= set(plan['source_SHA256']),
            'collector, reducer and all own numerical family sources must be pinned')
    for path, digest in plan['source_SHA256'].items():
        require(sha(path) == digest, 'bound source changed: '+path)
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    result = collect(plan, check)
    for path, digest in plan['source_SHA256'].items():
        require(sha(path) == digest, 'bound source changed during reduction: '+path)
    (output/'statistics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
