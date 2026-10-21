"""Original zero-shot numerics with an explicit frozen full508 data guard.

The old zero-shot entry requires a prior32 reference replay. Held-out data has
no prior model prediction to replay, so validate its actual508 bound SS archive
instead. The native numerical AST is unchanged outside the existing archive IO
adapter; only this named metadata-validation function is replaced in memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .evaluate_bound_population import pinned, sha
from .frozen_cascade_metrics import verify_test_SS_binding
from .summarize_frozen_inception_pools import validate_frozen_models
from .zero_shot_cascade_inputs import zero_shot_inputs


def validate_bound_zero_shot(plan, evaluation):
    if 'bound_zero_shot' not in plan:
        raise ValueError('explicit frozen full508 zero-shot binding required')
    binding=plan['bound_zero_shot']
    freeze=json.loads(pinned(binding['model_freeze']).read_text())
    ids,_,_=validate_frozen_models(freeze,freeze['models'])
    for evidence in freeze['development_evidence']:pinned(evidence)
    manifest=pinned(binding['manifest'])
    rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if ([r['sample_id'] for r in rows]!=ids or any(r['split']!='test' for r in rows)
            or binding['manifest']['SHA256']!=freeze['population']['manifest_SHA256']):
        raise ValueError('the exact frozen test508 population is required')
    model=freeze['models'][binding['model_alias']]
    if (not model['original_zero_shot_stack'] or model['texture_protocol']!='native-inline'
            or model['pbr_conditioner_checkpoint'] is not None or model['ss64']['step']!=4210
            or model['ss64']['conditioner_checkpoint'] is not None):
        raise ValueError('all four original multiview stages without deformation updates required')
    verify_test_SS_binding(evaluation,model,freeze_spec=binding['model_freeze'],manifest_spec=binding['manifest'],
                          seed_map_SHA256=freeze['population']['seed_map_SHA256'])
    protocol=zero_shot_inputs(plan['row'],plan['conditioning_mode'],plan['RGB_path_map'])
    if protocol['conditioning_mode']!=model['ss64']['conditioning_mode']:
        raise ValueError('frozen original source-only or edited-RGB input mode differs')
    for key in ['input_states','RGB_paths']:
        if plan[key]!=protocol[key]:raise ValueError('actual original four RGB inputs differ')
    for key in ['conditioning_mode','input_states','input_camera_ids','target_modality']:
        if evaluation[key]!=protocol[key]:raise ValueError('SS and downstream original RGB protocols differ')
    if (evaluation['status']!='PASS' or evaluation['stage']!='ss64' or evaluation['weights']!='raw'
            or evaluation['step']!=4210 or evaluation['conditioner_checkpoint'] is not None
            or evaluation['checkpoint']!=str(pinned(model['ss64']['checkpoint']))
            or evaluation['checkpoint']!=plan['original_SS_checkpoint']
            or evaluation['optimizer_updates_during_evaluation']!=0 or evaluation['diagnostic_only']
            or evaluation['native_SS_artifacts']['status']!='PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_BOUND_POPULATION'
            or [r['sample_id'] for r in evaluation['native_SS_artifacts']['artifacts']]!=ids):
        raise ValueError('actual full508 original SS4210 archive required, without a fabricated prior reference')
    if plan['row']['sample_id'] not in ids:raise ValueError('sample outside the frozen population')
    for name,step in [('shape512',4210),('shape1024',16850),('pbr1024',16850)]:
        stage=plan['stages'][name]
        if (stage['original_multiview_step']!=step
                or stage['checkpoint']!=str(pinned(model['stages'][name]['checkpoint']))
                or stage['config']!=str(pinned(model['stages'][name]['config']))
                or stage['checkpoint']!=plan['original_stage_checkpoints'][name]):
            raise ValueError('actual original native stage weights/config differ: '+name)
    return protocol


def run(binding):
    # Import the unchanged archive adapter only at execution, not for CPU guards.
    from .cascade_remote_results import CascadeRemoteResults, remote_io_tree
    from .immutable_result_store import ImmutableResultStore
    if binding['stage']!='zero-generation' or binding['bound_zero_adapter_source_SHA256']!=sha(__file__):
        raise ValueError('explicit pinned full508 zero-shot guard adapter required')
    source_path=Path(__file__).with_name('cascade_zero_shot_from_ss_artifact.py')
    native_path=Path(binding['native_plan'])
    if sha(source_path)!=binding['native_source_SHA256'] or sha(native_path)!=binding['native_plan_SHA256']:
        raise ValueError('original native numerical source/plan bytes differ')
    native=json.loads(native_path.read_text())
    tree,contract=remote_io_tree(source_path.read_text(),stage='zero-generation')
    contract.update(metadata_validation_adapter=__file__,metadata_validation_adapter_SHA256=sha(__file__),
        numerical_native_AST_reverses_exact=True,no_fake_reference_subset_or_step_metadata=True)
    store=ImmutableResultStore(binding['store']['ssh_argv'],binding['store']['remote_root'])
    io=CascadeRemoteResults(store,output=native['output'],prefix=binding['output_prefix'],stage='zero-generation',
        input_manifests=binding['input_manifests'],source_contract=contract,compressed=binding['compressed'])
    namespace=dict(__name__=__package__+'._bound_zero_native',__package__=__package__,__file__=str(source_path),_remote_results=io)
    exec(compile(tree,str(source_path),'exec'),namespace)
    namespace['validate_zero_shot_cascade']=validate_bound_zero_shot
    previous=sys.argv
    try:
        sys.argv=[str(source_path),'--plan',str(native_path)]
        namespace['main']()
    finally:sys.argv=previous


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();run(json.loads(args.plan.read_text()))


if __name__=='__main__':main()
