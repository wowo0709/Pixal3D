"""Original zero-shot cascade with genuine bound508 SS archive metadata.

Only the named metadata validator is scoped to the new population protocol;
the original cascade's model loading, numerics, and post-generation GT use run
from their existing unchanged module.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from .evaluate_bound_population import pinned, sha
from .native_population_contract import FINAL
from .original_zero_population import MODES
from .zero_shot_cascade_inputs import zero_shot_inputs


def validate(plan, evaluation, *, parent, controls):
    alias=plan['original_control_alias']
    if alias not in MODES or controls.get('status')!='FROZEN_ORIGINAL_ZERO_RGB_CONTROLS_BEFORE_TEST':
        raise ValueError('explicit pre-test frozen original control required')
    model=controls['models'][alias]
    population=parent['population'];ids=population['sample_ids']
    if (parent.get('status')!='FROZEN_FROM_DEVELOPMENT_ONLY'
            or parent.get('heldout_quality_read_before_freeze')is not False
            or controls.get('heldout_model_quality_read')is not False
            or len(ids)!=508 or len(set(ids))!=508
            or len(set(population['source_clusters_by_sample'].values()))!=134
            or population['rendered_cameras']!=[8,9,10]
            or evaluation.get('population_scope')!=FINAL
            or evaluation.get('status')!='PASS' or evaluation.get('stage')!='ss64'
            or evaluation.get('step')!=0 or evaluation.get('original_multiview_SS_step')!=4210
            or evaluation.get('deformation_finetuning_updates')!=0
            or evaluation.get('frozen_native_model_binding')!=model
            or evaluation.get('checkpoint')!=model['checkpoint']['path']
            or evaluation.get('conditioner_checkpoint')is not None
            or evaluation.get('optimizer_updates_during_evaluation')!=0
            or evaluation.get('diagnostic_only')is not False
            or evaluation.get('quick_panel_ids')!=ids
            or evaluation.get('actual_original_full_base_and_LoRA_validation',{}).get(
                'original_full_base_and_LoRA_exact_before_evaluation')is not True):
        raise ValueError('actual full508 original zero-update SS4210 archive required')
    artifacts=evaluation['native_SS_artifacts']
    if (artifacts.get('status')!='PASS_ACTUAL_ORIGINAL_ZERO_SS_BOUND_POPULATION_ARCHIVE'
            or artifacts.get('archived_asset_count')!=508
            or [a['sample_id']for a in artifacts['artifacts']]!=ids
            or [r['sample_id']for r in evaluation['per_asset']]!=ids
            or {r['sample_id']:r['seed']for r in evaluation['per_asset']}!=population['generation_seeds']):
        raise ValueError('complete prescribed508 SS artifacts and noise required, without a fake prior replay')
    row=plan['row'];sid=row['sample_id']
    if (sid not in ids or row['source_cluster_id']!=population['source_clusters_by_sample'][sid]
            or row['split']!='test' or plan['conditioning_mode']!=model['conditioning_mode']
            or plan['original_SS_checkpoint']!=model['checkpoint']['path']):
        raise ValueError('actual frozen sample/cluster/input mode required')
    protocol=zero_shot_inputs(row,plan['conditioning_mode'],plan['RGB_path_map'])
    for key in ['input_states','RGB_paths']:
        if plan[key]!=protocol[key]:raise ValueError('actual four-RGB zero-shot inputs differ: '+key)
    for key in ['conditioning_mode','input_states','input_camera_ids','target_modality']:
        if evaluation[key]!=protocol[key]:raise ValueError('SS and downstream original RGB protocol differ: '+key)
    if set(plan['stages'])!={'shape512','shape1024','pbr1024'}:
        raise ValueError('complete original downstream stage composition required')
    for name,step in [('shape512',4210),('shape1024',16850),('pbr1024',16850)]:
        own=controls['downstream_original_stages'][name];actual=plan['stages'][name]
        parent_stage=parent['original_PBR_for_SS_isolation'] if name=='pbr1024'else parent['models']['vanilla'][name]
        if (own['step']!=step or actual['original_multiview_step']!=step
                or own['checkpoint']['SHA256']!=parent_stage['checkpoint']['SHA256']
                or actual['checkpoint']!=own['checkpoint']['path']
                or actual['checkpoint']!=plan['original_stage_checkpoints'][name]
                or actual['config']!=own['config']['path']):
            raise ValueError('original multi-view stage/config changed: '+name)
    return protocol


@contextmanager
def metadata_validator(core, callback):
    original=core.validate_zero_shot_cascade;calls=[]
    def checked(plan,evaluation):
        calls.append(1)
        if len(calls)!=1:raise ValueError('one actual original cascade metadata-validation site required')
        return callback(plan,evaluation)
    core.validate_zero_shot_cascade=checked
    try:
        yield
        if calls!=[1]:raise ValueError('the original numerical entry did not use its metadata guard')
    finally:core.validate_zero_shot_cascade=original


def execute(binding):
    path=pinned(binding['native_plan']);plan=json.loads(path.read_text())
    parent=json.loads(pinned(binding['parent_freeze']).read_text())
    controls=json.loads(pinned(binding['control_freeze']).read_text())
    ss=pinned(binding['SS_evaluation']);evaluation=json.loads(ss.read_text())
    if plan['SS_evaluation']!=str(ss):raise ValueError('native SS input differs from the bound original control archive')
    validate(plan,evaluation,parent=parent,controls=controls)
    for evidence in parent['development_evidence']:pinned(evidence)
    for stage in controls['downstream_original_stages'].values():
        for key in ['config','checkpoint']:pinned(stage[key])
    pinned(controls['models'][plan['original_control_alias']]['checkpoint'])
    from . import cascade_zero_shot_from_ss_artifact as core
    source=dict(path=core.__file__,SHA256=binding['native_source_SHA256']);pinned(source)
    if sha(__file__)!=binding['metadata_adapter_SHA256']:raise ValueError('original zero metadata adapter changed')
    old=sys.argv
    try:
        with metadata_validator(core,lambda p,r:validate(p,r,parent=parent,controls=controls)):
            sys.argv=[core.__file__,'--plan',str(path)];core.main()
    finally:sys.argv=old
    for spec in [source,binding['native_plan'],binding['parent_freeze'],binding['control_freeze'],binding['SS_evaluation']]:pinned(spec)
    output=Path(plan['output']);result=json.loads((output/'generation.json').read_text())
    if (result.get('status')!='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
            or not result.get('original_zero_shot_full_stack')
            or not result.get('GT_loaded_only_after_generation')
            or result.get('GT_geometry_or_coords_used_in_generation')
            or result.get('deformation_finetuning_updates')!=0):
        raise ValueError('actual unchanged original full-cascade execution failed')
    proof=dict(status='PASS_ACTUAL_ORIGINAL_ZERO_CASCADE_BOUND_TEST508_SAMPLE',
        original_control_alias=plan['original_control_alias'],sample_id=result['sample_id'],
        source_cluster_id=plan['row']['source_cluster_id'],parent_freeze=binding['parent_freeze'],
        control_freeze=binding['control_freeze'],SS_evaluation=binding['SS_evaluation'],
        native_source=source,numerical_source_unchanged=True,no_fabricated_reference_or_step_metadata=True,
        original_multi_view_steps=result['original_multiview_steps'],optimizer_updates=0,full_goal_complete=False)
    (output/'zero-population-binding.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    args=p.parse_args();print(json.dumps(execute(json.loads(args.plan.read_text()))),flush=True)


if __name__=='__main__':main()
