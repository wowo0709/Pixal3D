"""Archive native SS128 predictions during the original joint evaluation.

The sampler/decoder executes once. A later DEV32 cascade subset can copy the
actual saved bytes on CPU, without loading another model or sampling again.
"""
import json
from pathlib import Path
import sys

from .cascade_view_roles import file_sha256
from .evaluate_bound_population import argument
from .evaluate_ss_artifact_subset import capture_ss_artifacts


def validate_arguments(arguments):
    expected={'--stage':'ss64','--step':'480','--expected-assets':'128',
              '--panel-size':'128','--generation-batch':'8','--source-value-control':'normal'}
    for key,value in expected.items():
        if argument(arguments,key)!=value:
            raise ValueError('native development128 archive protocol differs: '+key)
    for flag in ['--arm-config','--checkpoint','--conditioner-checkpoint','--completed-training-root',
                 '--source-input-design','--panel-file','--seed-map-file','--output']:
        argument(arguments,flag)
    if any(flag in arguments for flag in ['--conditioning-mode','--transport-control',
           '--r9-path-control','--task-value-control','--source-teacher-init-spec','--pilot-variant']):
        raise ValueError('single actual joint480 edited-RGB evaluation without combined diagnostics required')


def validate_captured(result,captured):
    rows=result['per_asset'];source=result['source_input_joint']
    if (result['status']!='PASS' or result['stage']!='ss64' or result['step']!=480
            or result['optimizer_updates_during_evaluation']!=0 or result['diagnostic_only']
            or result['generation_batch_size']!=8 or result['sampling_steps']!=12 or result['guidance']!=3.
            or result['input_states']!=['M1','M0','M0','M0'] or result['input_camera_ids']!=[8,9,10,11]
            or result['target_modality']!='rgb'
            or len(rows)!=128 or len(captured)!=128 or len({r['sample_id']for r in rows})!=128
            or source['completed_native_joint_updates']!=480 or source['source_tensors']!=19
            or source['variant'] not in {'original-NoDQ','source-contrast-NoDQ'}
            or not source['actual_joint_SS_LoRA_and_source_finetuned']
            or not source['source_and_denoiser_same_completed_run']
            or not source['actual_joint_source_state_exact_after_evaluation']):
        raise ValueError('actual same-run joint480/source19 native128 prediction archive required')
    for record,artifact in zip(rows,captured):
        if (record['sample_id']!=artifact['sample_id'] or record['seed']!=artifact['seed']
                or record['target_voxels_sha256']!=artifact['target_voxels_SHA256']
                or any(record[k]!=artifact['occupancy_counts'][k]for k in ['tp','fp','fn'])):
            raise ValueError('actual SS decode/archive and native voxel records differ')


def main():
    arguments=sys.argv[1:];validate_arguments(arguments)
    output=Path(argument(arguments,'--output'))
    if output.exists():raise FileExistsError('preserve existing native archive/evaluation output')
    from . import quick_metrics as quick,evaluate_source_input_joint as entry
    sources=[Path(__file__),Path(entry.__file__),Path(__file__).with_name('evaluate_ss_artifact_subset.py')]
    bound={str(p):file_sha256(p)for p in sources};captured=[];argv=sys.argv
    try:
        with capture_ss_artifacts(quick,output,captured):
            sys.argv=[entry.__file__,*arguments];entry.main()
    finally:sys.argv=argv
    path=output/'evaluation.json';result=json.loads(path.read_text());validate_captured(result,captured)
    if any(file_sha256(p)!=digest for p,digest in bound.items()):
        raise ValueError('source archive/native entry changed during evaluation')
    # Preserve the complete original scientific result before adding export
    # provenance. No per-asset metric, latent, occupancy or model is changed.
    (output/'source-input-core-evaluation.json').write_bytes(path.read_bytes())
    result['native_SS_artifacts']=dict(status='PASS_ACTUAL_JOINT_SS128_SINGLE_EVALUATION_ARCHIVE',
        scope='development-validation128',artifacts=captured,archived_asset_count=128,
        actual_prediction_and_GT_grid_hashes_recorded=True,
        actual_native_decode_counts_and_GTnoise_records_exact=True,
        single_sampling_and_decode_execution=True,predicted_SS_is_available_for_full_cascade=True,
        GT_SS_is_for_scoring_only=True,additional_inference_or_optimizer_updates=0,
        original_native_arguments=arguments,underlying_evaluation_module='evaluate_source_input_joint',
        source_SHA256=bound,model_adopted=False,full_goal_complete=False)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['native_SS_artifacts']['status'],assets=128,
        source_variant=result['source_input_joint']['variant'],additional_inference=0)),flush=True)


if __name__=='__main__':main()
