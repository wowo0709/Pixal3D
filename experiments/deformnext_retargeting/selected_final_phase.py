"""Execute one pinned final-case phase through its original numerical entry."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from .assigned_gpu_entry import adapt_guard, validate_assignment
from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .evaluate_bound_population import pinned
from .native_final_freeze import validate_spec
from .selected_pipeline_model_contract import require


SCHEMA = 'selected-physical-final-native-phase-v1'
ENTRIES = {
    'native-guide-capture': 'cascade_from_ss_artifact',
    'native-guide-replay': 'replay_r_pbr_on_generated_shape',
    'selected-zero-bound-entry': 'cascade_zero_shot_from_ss_artifact',
    'assigned-native-render': 'render_cascade_artifact_v2',
    'original-shared-material': 'score_cascade_shared_material',
    'fixed-shared-material': 'score_cascade_fixed_reference',
}


def validate_binding(binding, native):
    require(binding.get('schema') == SCHEMA and binding.get('kind') in ENTRIES,
            'one explicit supported original numerical phase required')
    kind = binding['kind']; root = Path(__file__).resolve().parent
    for key in ['native_source','native_plan']:validate_spec(binding[key])
    entry = root/(ENTRIES[kind]+'.py')
    require(binding.get('native_source') == dict(path=str(entry),SHA256=native['source_SHA256'].get(str(entry))),
            'pin the actual original phase source in both binding and native plan')
    require(native['source_SHA256'].get(str(Path(__file__).resolve())) == binding.get('phase_source_SHA256'),
            'pin the actual generic phase adapter before execution')
    if kind == 'selected-zero-bound-entry':
        for name in ['parent_freeze','control_freeze','SS_evaluation']:
            require(name in binding,'actual zero metadata binding required')
    return kind,entry


def run_original(kind, path, native, binding):
    """Use existing guide/zero wrappers, or adapt only the GPU literal."""
    if kind == 'selected-zero-bound-entry':
        from .selected_original_zero_cascade_population import execute
        adapter = Path(__file__).with_name('selected_original_zero_cascade_population.py')
        return execute(dict(native_plan=binding['native_plan'],
            **{k:binding[k] for k in ['parent_freeze','control_freeze','SS_evaluation']},
            native_source_SHA256=binding['native_source']['SHA256'],metadata_adapter_SHA256=sha(adapter)))
    old_argv = sys.argv
    try:
        if kind in ('native-guide-capture','native-guide-replay'):
            from .native_guide_entry import main
            sys.argv = ['native_guide_entry','--kind','capture' if kind == 'native-guide-capture' else 'replay',
                        '--plan',str(path)]
            main()
            return dict(actual_existing_native_guide_entry_used=True)
        source = Path(binding['native_source']['path'])
        module = importlib.import_module('experiments.deformnext_retargeting.'+ENTRIES[kind])
        tree,proof = adapt_guard(source.read_text(),native['assigned_GPUs'])
        namespace = dict(module.__dict__); exec(compile(tree,str(source),'exec'),namespace)
        sys.argv = [str(source),'--plan',str(path)]; namespace['main']()
        return dict(numerical_entry_AST_device_literal_only=proof)
    finally:
        sys.argv = old_argv


def execute(path):
    path = Path(path); binding = json.loads(path.read_text())
    native_path = pinned(binding['native_plan']); native = json.loads(native_path.read_text())
    kind,entry = validate_binding(binding,native)
    validate_assignment(native,os.environ.get('CUDA_VISIBLE_DEVICES',''),os.environ.get('MATCHING_V2_ALLOWED_GPUS',''))
    check = ClosedInputs(native['canonical_local_root'])
    def guard():
        check.check(path,path.parent/'file4-closure.json')
        check.check(native_path,native_path.parent/'file4-closure.json')
        require(sha(__file__) == binding['phase_source_SHA256'],'actual phase adapter changed')
        pinned(binding['native_plan']); pinned(binding['native_source'])
        for p,d in {**native.get('source_SHA256',{}),**native.get('inputs_SHA256',{}),**native.get('input_SHA256',{})}.items():
            require(sha(p) == d,'actual phase source/input changed: '+p)
        for p,r in native.get('closed_inputs',{}).items():check.check(p,r)
    guard()
    proof = run_original(kind,native_path,native,binding)
    guard()
    result = dict(status='PASS_ACTUAL_ORIGINAL_NUMERICAL_SELECTED_CASE_PHASE',kind=kind,
        native_source=binding['native_source'],native_plan=binding['native_plan'],physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],
        actual_original_entry_executed=True,proof=proof,full_goal_complete=False)
    with (Path(native['output'])/'selected-phase-entry.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',type=Path,required=True)
    args = parser.parse_args(); print(json.dumps(execute(args.plan)),flush=True)


if __name__ == '__main__': main()
