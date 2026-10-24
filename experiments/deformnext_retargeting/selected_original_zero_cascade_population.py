"""Use original zero-shot numerics with a selected-pipeline V2 population guard."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from .assigned_gpu_entry import adapt_guard, validate_assignment
from .archived_native_guides import POLICY
from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .evaluate_bound_population import pinned
from .selected_final_population import admit_ss, compose_models
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require
from .zero_shot_cascade_inputs import zero_shot_inputs


def required_sources():
    root = Path(__file__).resolve().parent
    return [str(root/(name+'.py')) for name in ['selected_original_zero_cascade_population',
        'selected_final_population','selected_pipeline_model_contract','selected_pipeline_freeze_v2',
        'assigned_gpu_entry','cascade_zero_shot_from_ss_artifact','zero_shot_cascade_inputs',
        'archived_native_guides','native_guide_capture_context']]


def validate(plan, evaluation, *, parent, controls):
    alias = plan['original_control_alias']; models = compose_models(parent, controls)
    require(alias in ('zero-source4','zero-edited4'), 'one actual original four-RGB control required')
    model = models[alias]
    admit_ss(alias, model, evaluation, parent=parent, controls=controls,
        manifest_SHA256=parent['population']['manifest']['SHA256'])
    row = plan['row']; sid = row['sample_id']
    require(sid in parent['population']['sample_ids'] and row['split'] == 'test'
        and row['source_cluster_id'] == parent['population']['source_clusters_by_sample'][sid]
        and plan['original_SS_checkpoint'] == model['ss64']['checkpoint']['path']
        and plan['conditioning_mode'] == model['ss64']['conditioning_mode'],
        'actual frozen sample/source/SS checkpoint and input mode required')
    protocol = zero_shot_inputs(row, plan['conditioning_mode'], plan['RGB_path_map'])
    require(all(plan[k] == protocol[k] for k in ['input_states','RGB_paths'])
        and all(evaluation[k] == protocol[k] for k in ['conditioning_mode','input_states','input_camera_ids','target_modality']),
        'SS and original cascade must use the same four calibrated RGB inputs')
    require(set(plan['stages']) == {'shape512','shape1024','pbr1024'}, 'all three original downstream stages required')
    for name, own in model['geometry_stages'].items():
        actual = plan['stages'][name]
        require(actual['original_multiview_step'] == own['step']
            and actual['config'] == own['config']['path'] and actual['checkpoint'] == own['checkpoint']['path']
            and plan['original_stage_checkpoints'][name] == own['checkpoint']['path']
            and exact(own, parent['original_stages'][name]), 'original stage/config differs: '+name)
    return protocol


def invoke_original(core, plan_path, plan, evaluation, *, parent, controls, devices):
    """Scope only the metadata callback and the explicit GPU membership literal."""
    source = Path(core.__file__); tree, proof = adapt_guard(source.read_text(), devices)
    namespace = dict(core.__dict__); exec(compile(tree, str(source), 'exec'), namespace)
    previous = namespace['validate_zero_shot_cascade']; calls = []
    def checked(actual_plan, actual_evaluation):
        require(exact(actual_plan, plan) and exact(actual_evaluation, evaluation), 'original entry loaded another case or SS archive')
        calls.append(1); require(len(calls) == 1, 'one original metadata validation call required')
        return validate(actual_plan, actual_evaluation, parent=parent, controls=controls)
    old_argv = sys.argv; namespace['validate_zero_shot_cascade'] = checked
    try:
        sys.argv = [str(source), '--plan', str(plan_path)]; namespace['main']()
        require(calls == [1], 'the original numerical entry must use the population metadata guard')
    finally:
        namespace['validate_zero_shot_cascade'] = previous; sys.argv = old_argv
    return proof


def execute(binding):
    path = pinned(binding['native_plan']); plan = json.loads(path.read_text())
    parent = json.loads(pinned(binding['parent_freeze']).read_text())
    controls = json.loads(pinned(binding['control_freeze']).read_text())
    ss = pinned(binding['SS_evaluation']); evaluation = json.loads(ss.read_text())
    require(plan['SS_evaluation'] == str(ss), 'actual original SS archive differs from bound input')
    devices = validate_assignment(plan, os.environ.get('CUDA_VISIBLE_DEVICES',''), os.environ.get('MATCHING_V2_ALLOWED_GPUS',''))
    check = ClosedInputs(plan['canonical_local_root']); check.check(path, path.parent/'file4-closure.json')
    require(set(required_sources()) <= set(plan['source_SHA256']), 'pin each original numerical and selected metadata validator')
    for source, digest in {**plan['source_SHA256'], **plan.get('inputs_SHA256',{})}.items():
        require(sha(source) == digest, 'actual original source/input changed')
    for source, receipt in plan.get('closed_inputs',{}).items(): check.check(source, receipt)
    validate(plan, evaluation, parent=parent, controls=controls)
    specs = [binding[k] for k in ['native_plan','parent_freeze','control_freeze','SS_evaluation']]
    specs.extend(parent['development_evidence'])
    for stage in parent['original_stages'].values(): specs.extend(stage[k] for k in ['config','checkpoint'])
    actual_SS = controls['models'][plan['original_control_alias']]
    specs.extend(actual_SS[k] for k in ['native_config','checkpoint'])
    for spec in specs: pinned(spec)
    core = importlib.import_module('experiments.deformnext_retargeting.cascade_zero_shot_from_ss_artifact')
    require(sha(core.__file__) == binding['native_source_SHA256']
        and sha(__file__) == binding['metadata_adapter_SHA256'], 'pin actual original numerical source and metadata adapter')
    require(plan.get('texture_support_policy') == POLICY, 'capture actual original native guides required')
    from .native_guide_capture_context import capture_native_guides
    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
    archive = Path(plan['output'])/'native-shape-guides.pt'
    with capture_native_guides(Pixal3DImageTo3DPipeline, archive) as events:
        proof = invoke_original(core, path, plan, evaluation, parent=parent, controls=controls, devices=devices)
    require(len(events) == 1 and events[0]['actual_guide_count'] == 4,
            'one actual original Shape decode and four unchanged native guides required')
    for spec in specs: pinned(spec)
    require(sha(core.__file__) == binding['native_source_SHA256']
        and sha(__file__) == binding['metadata_adapter_SHA256'], 'original source changed during execution')
    for source, digest in {**plan['source_SHA256'], **plan.get('inputs_SHA256',{})}.items():
        require(sha(source) == digest, 'original source or input changed during execution')
    result = json.loads((Path(plan['output'])/'generation.json').read_text())
    require(result.get('status') == 'PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
        and result.get('original_zero_shot_full_stack') is True and result.get('GT_loaded_only_after_generation') is True
        and result.get('GT_geometry_or_coords_used_in_generation') is False
        and result.get('deformation_finetuning_updates') == 0, 'actual original full-cascade execution failed')
    result.update(texture_support_policy=POLICY,actual_native_guide_archive_SHA256=sha(archive),
        actual_native_guide_events=events,actual_native_guide_logits_order_scale_and_dead_branches_preserved=True,
        original_zero_cascade_mathematics_and_returned_guide_objects_unchanged=True,
        numerical_entry_binding=proof,shape_decoder_recomputed=True)
    (Path(plan['output'])/'generation.json').write_text(json.dumps(result,indent=2)+'\n')
    record = dict(status='PASS_ACTUAL_ORIGINAL_ZERO_SELECTED_BOUND_TEST508_SAMPLE',
        original_control_alias=plan['original_control_alias'],sample_id=result['sample_id'],
        source_cluster_id=plan['row']['source_cluster_id'],parent_freeze=binding['parent_freeze'],
        control_freeze=binding['control_freeze'],SS_evaluation=binding['SS_evaluation'],
        numerical_entry_AST_device_literal_only=proof,numerical_model_and_sampling_math_unchanged=True,
        original_multiview_steps=result['original_multiview_steps'],optimizer_updates=0,full_goal_complete=False)
    (Path(plan['output'])/'selected-zero-population-binding.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(execute(json.loads(args.plan.read_text()))), flush=True)


if __name__ == '__main__': main()
