"""Resource-only placement of pending, source-bound correspondence SS evaluations.

No files, processes, numerical models, or queue state are changed here.
Callers must retire the original pending producers before activating successors.
"""
from copy import deepcopy
from pathlib import Path


VARIANTS = ('M_FIXED_SPATIAL', 'M_FIXED_RAW', 'M_FIXED_D1_COS')
EVALUATOR = 'experiments.deformnext_retargeting.evaluate_fixed_correspondence_checkpoint'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(value):
    return isinstance(value, str) and Path(value).is_absolute() and '..' not in Path(value).parts


def relocate_plan(plan, *, original_path, replacement_path, devices, cpu_affinity):
    """Change explicit device/CPU admission and the self-referencing CLI only."""
    require(plan.get('phase') == 'evaluation' and plan.get('mode') == 'final128'
        and plan.get('variant') in VARIANTS and plan.get('expected_assets') == 128
        and plan.get('generation_batch_size') == 8
        and plan.get('deterministic_algorithms') is True,
        'own deterministic correspondence final128 evaluation required')
    require(absolute(original_path) and absolute(replacement_path)
        and original_path != replacement_path, 'distinct explicit absolute plan paths required')
    require(isinstance(devices, list) and bool(devices)
        and all(isinstance(g, str) and g.isdecimal() for g in devices)
        and len(devices) == len(set(devices)) and set(cpu_affinity) == set(devices),
        'explicit distinct assigned devices and matching CPU map required')
    cpus = [c for g in devices for c in cpu_affinity[g]]
    require(all(isinstance(cpu_affinity[g], list) and cpu_affinity[g] for g in devices)
        and all(type(c) is int and c >= 0 for c in cpus) and len(cpus) == len(set(cpus)),
        'nonempty disjoint explicit integer CPU assignments required')
    command = plan.get('command')
    require(command == [plan.get('python'), '-u', '-m', EVALUATOR, '--plan', original_path],
        'exact own evaluator and self-referencing plan CLI required')
    require(plan.get('output') == plan.get('native_output')
        and all(absolute(plan.get(k)) for k in ('output', 'wrapper_output', 'bundle', 'panel', 'seeds'))
        and Path(plan['wrapper_output']).parent == Path(plan['output']).parent
        and plan['wrapper_output'] != plan['output']
        and all(isinstance(plan.get(k), dict) and plan[k]
                for k in ('source_SHA256', 'input_SHA256', 'closed_inputs')),
        'unchanged own model, complete panel/noise inputs and paired outputs required')
    result = deepcopy(plan)
    result['assigned_GPUs'] = list(devices)
    result['CPU_affinity_per_GPU'] = deepcopy(cpu_affinity)
    result['command'][-1] = replacement_path
    normalized = deepcopy(result)
    normalized['assigned_GPUs'] = deepcopy(plan['assigned_GPUs'])
    normalized['CPU_affinity_per_GPU'] = deepcopy(plan['CPU_affinity_per_GPU'])
    normalized['command'][-1] = original_path
    require(normalized == plan, 'resource placement must preserve every numerical/input/output field')
    return result


def successor_job(job, *, state, name, replacement_path, relocated_plan,
                  producer_outputs, main_training_job, prelaunch_receipt,
                  activation_receipt, priority=-20):
    """Convert auxiliary completion dependencies to their explicit file receipts."""
    require(state.get('status') == 'pending', 'only never-started pending jobs can move')
    require(job.get('resource') == 'exclusive'
        and not any(k in job for k in ('gpu', 'gpus', 'gate_file', 'attached_pid')),
        'original unpinned exclusive one-GPU evaluation required')
    require(isinstance(name, str) and name and name != job.get('name')
        and isinstance(main_training_job, str) and bool(main_training_job)
        and type(priority) is int, 'explicit successor identity, training gate and priority required')
    require(all(absolute(p) for p in (replacement_path, prelaunch_receipt, activation_receipt)),
        'explicit canonical placement and activation receipts required')
    outputs = [str(Path(relocated_plan[k])/'file4-closure.json')
               for k in ('native_output', 'wrapper_output')]
    require(job.get('outputs') == outputs, 'same native/wrapper output identities required')
    dependencies = job.get('depends_on', [])
    require(bool(dependencies) and all(d in producer_outputs and producer_outputs[d]
        and set(producer_outputs[d]) <= set(job.get('requires', [])) for d in dependencies),
        'each auxiliary dependency must already have explicit required output receipts')
    command = job.get('command', [])
    require(command.count('--plan') == 1 and command.index('--plan') + 1 < len(command),
        'exact one plan argument required')
    result = deepcopy(job)
    result.update(name=name, priority=priority, depends_on=[main_training_job])
    result['command'][command.index('--plan') + 1] = replacement_path
    result['requires'] = list(dict.fromkeys([*job['requires'], prelaunch_receipt, activation_receipt]))
    require(result['resource'] == 'exclusive' and result['outputs'] == job['outputs'],
        'no overlap with ongoing training or output substitution allowed')
    return result
