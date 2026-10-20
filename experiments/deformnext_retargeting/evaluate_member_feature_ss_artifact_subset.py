"""Exact32-of128 SS prediction archive for the sealed four-head runtimes."""
import argparse
import importlib
import json
from pathlib import Path
import sys

from .evaluate_ss_artifact_subset import capture_ss_artifacts,validate_subset_reference,sha
from .ss_pooled_support_metrics import artifact_support_metrics


RUNTIMES={
    'member_feature_runtime':dict(physical=0,source_tensors=19,input_channels=2048,arm='feature-direct-nodq',
        marker='member-feature-runtime.json',status='PASS_COMPLETED_MEMBER_FEATURE_RUNTIME_PHASE'),
    'member_feature_deformation_runtime':dict(physical=2048,source_tensors=20,input_channels=4096,arm='feature-direct-dq',
        marker='member-feature-deformation-runtime.json',status='PASS_COMPLETED_MEMBER_FEATURE_DEFORMATION_RUNTIME_PHASE'),
}


def validate_archive_arguments(arguments):
    expected={'--phase':'evaluation','--stage':'ss64','--step':'480',
        '--expected-assets':'32','--panel-size':'32','--generation-batch':'8','--source-value-control':'normal'}
    for flag,value in expected.items():
        if arguments.count(flag)!=1:
            raise ValueError('one explicit read-only SS archive '+flag+' required')
        index=arguments.index(flag)+1
        if index>=len(arguments) or arguments[index]!=value:
            raise ValueError('fixed read-only SS archive protocol changed '+flag)


def validate_member_replay(actual,reference,captured,runtime):
    validate_subset_reference(actual,reference,captured)
    spec=RUNTIMES[runtime]
    if (len(reference['per_asset'])!=128
            or len({row['sample_id'] for row in reference['per_asset']})!=128
            or len({row['sample_id'] for row in actual['per_asset']})!=32):
        raise ValueError('unique32 prefix of original unique128 development evaluation required')
    for result in [actual,reference]:
        source=result['native_prior_source_ray']
        if (result['step']!=480 or result['diagnostic_only']
                or result['conditioning_mode']!='edited-rgb' or result['target_modality']!='rgb'
                or result['input_states']!=['M1','M0','M0','M0']
                or result['input_camera_ids']!=[8,9,10,11]
                or result['generation_batch_size']!=8 or result['sampling_steps']!=12 or result['guidance']!=3.
                or source['arm']!=spec['arm'] or source['source_value_control']!='normal'
                or source['physical_deformation_query_channels']!=spec['physical']
                or source['physical_retargeter_input_channels']!=spec['input_channels']
                or source['actual_QKV_feature_tensors']!=9 or source['actual_optimizer_states']!=0
                or not source['target_carrier_is_native_four_view_V1']):
            raise ValueError('actual four-head nativeMV4 NoDQ/DQ archive identity or protocol differs')


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--evaluation-module',choices=list(RUNTIMES),required=True)
    parser.add_argument('--reference-evaluation',type=Path,required=True)
    local,remaining=parser.parse_known_args()
    validate_archive_arguments(remaining)
    from . import quick_metrics as quick
    output=Path(remaining[remaining.index('--output')+1])
    if output.exists():raise FileExistsError('fresh four-head prediction archive namespace required')
    reference=json.loads(local.reference_evaluation.read_text())
    entry=importlib.import_module('.'+local.evaluation_module,__package__)
    from . import ss_pooled_support_metrics as support
    bound={str(Path(path).resolve()):sha(path) for path in [__file__,quick.__file__,entry.__file__,
        support.__file__,local.reference_evaluation]}
    captured=[];previous=sys.argv
    try:
        with capture_ss_artifacts(quick,output,captured):
            sys.argv=[entry.__file__,*remaining];entry.main()
    finally:sys.argv=previous
    path=output/'evaluation.json';actual=json.loads(path.read_text())
    validate_member_replay(actual,reference,captured,local.evaluation_module)
    spec=RUNTIMES[local.evaluation_module]
    marker=json.loads((output/spec['marker']).read_text())
    if (marker['status']!=spec['status'] or marker['phase']!='evaluation'
            or marker['actual_QKV_feature_head_tensors']!=9
            or marker['actual_source_tensors']!=spec['source_tensors']
            or marker['physical_deformation_query_channels']!=spec['physical']):
        raise ValueError('actual completed four-head evaluation runtime marker required')
    if any(sha(path)!=digest for path,digest in bound.items()):
        raise ValueError('archive runtime/source or exact reference changed')
    for artifact in captured:
        artifact['native_SS32_support']=artifact_support_metrics(output/artifact['artifact'])
    actual['native_SS_artifacts']=dict(status='PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_EXACT_REFERENCE_SUBSET_REPLAY',
        original_prefix32_all_records_and_occupancy_counts_exact=True,artifacts=captured,
        reference_evaluation=str(local.reference_evaluation),reference_SHA256=sha(local.reference_evaluation),
        source_SHA256=bound,actual_evaluation_runtime=marker,reference_asset_count=128,archived_asset_count=32,
        reference_full_panel_quick_metrics_not_compared_to_subset=True,
        original_reference_had_no_prediction_grid_hash_so_not_claimed=True,
        predicted_SS_is_available_for_full_cascade=True,GT_SS_is_for_scoring_only=True,
        decoded_GT_and_prediction_support32_metrics_recorded=True,
        full_cascade_generated=False,optimizer_updates=0)
    path.write_text(json.dumps(actual,indent=2)+'\n')
    print(json.dumps(dict(status=actual['native_SS_artifacts']['status'],assets=32,physical_DQ_channels=spec['physical'])),flush=True)


if __name__=='__main__':main()
