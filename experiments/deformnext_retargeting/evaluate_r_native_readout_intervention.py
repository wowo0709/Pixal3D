"""Saved native480 source readout diagnostics, distinct from retrained ablations."""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path

from . import evaluate_r_native_checkpoint as loader
from .cascade_view_roles import ClosedInputs, file_sha256
from .r_mediated_interventions import CONTROLS, SavedRMediatedIntervention


MODE = 'native480-readout-intervention32'
IDENTITY = ('sample_id', 'source_cluster_id', 'seed', 'target_voxels_sha256',
            'target_voxel_shape', 'target_representation')
MODEL = ('native_variant', 'checkpoint', 'conditioner_checkpoint', 'arm_config',
         'step', 'actual_loaded_trainable_SHA256')


def validate_protocol(plan, bundle):
    if (plan.get('mode') != MODE or plan.get('expected_assets') != 32
            or plan.get('intervention') not in CONTROLS
            or plan.get('deterministic_algorithms') is not True
            or plan.get('generation_batch_size') != 8
            or bundle.get('variant') != 'R_raw_topk' or bundle.get('step') != 480
            or bundle.get('source_teacher_updates') != 0):
        raise ValueError('actual native raw-TopK480, deterministic DEV32/batch8 and named intervention required')


def compare_reference(actual, reference, *, exact):
    if (reference.get('native_variant') != 'R_raw_topk' or reference.get('step') != 480
            or reference.get('source_teacher_updates') != 0
            or reference.get('deterministic_algorithms') is not True
            or len(reference['per_asset']) != 32):
        raise ValueError('actual deterministic native480 DEV32 reference required')
    if (actual['quick_panel_ids'] != reference['quick_panel_ids']
            or any(actual[k] != reference[k] for k in MODEL)
            or len(actual['per_asset']) != 32):
        raise ValueError('same actual native480 model and ordered32 population required')
    for row, expected in zip(actual['per_asset'], reference['per_asset']):
        if any(row[k] != expected[k] for k in IDENTITY):
            raise ValueError('intervention changed paired source/GT/noise identity')
    if exact and (actual['per_asset'] != reference['per_asset']
                  or actual['quick_metrics'] != reference['quick_metrics']):
        raise ValueError('normal shadow did not reproduce actual native480 reference exactly')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    check = ClosedInputs(plan['canonical_local_root'])
    check.check(args.plan, args.plan.parent/'file4-closure.json')
    bundle = json.loads(Path(plan['bundle']).read_text())
    validate_protocol(plan, bundle)
    reference_path = Path(plan['intervention_reference'])
    check.check(reference_path, reference_path.parent/'file4-closure.json')
    reference = json.loads(reference_path.read_text())
    control = plan['intervention']
    if control != 'normal':
        proof = Path(plan['normal_shadow_proof'])
        check.check(proof, proof.parent/'file4-closure.json')
        observed = json.loads(proof.read_text())
        if (observed.get('status') != 'PASS_ACTUAL_NATIVE480_READOUT_DIAGNOSTIC'
                or observed.get('control') != 'normal'
                or not observed.get('normal_reference_records_exact')
                or observed.get('reference_SHA256') != file_sha256(reference_path)):
            raise ValueError('own actual native480 normal-shadow proof required before perturbation')
    original_panel, original_binding = loader.validate_panel, loader.cold_source_binding
    captured, consumer_events = [], []

    class ObservedIntervention(SavedRMediatedIntervention):
        def after(self, module, module_args, kwargs, result):
            super().after(module, module_args, kwargs, result)
            from .r_native_signal_probe import matching_signal
            self.events[-1]['matching_signal'] = matching_signal(module, result)

    def panel(p, b, c):
        validate_protocol(p, b)
        contract = copy.deepcopy(p)
        contract['mode'] = 'archive32'
        return original_panel(contract, b, c)

    @contextmanager
    def binding(variant):
        with original_binding(variant) as modules:
            yield modules
            if len(modules) != 1:
                raise ValueError('one actual native source module required')
            captured.append(ObservedIntervention(modules[0], control))

    from pixal3d.trainers.flow_matching import member_posterior_source_ray as consumer
    from .r_native_signal_probe import consumer_signal
    original_add = consumer.add_source_ray_correction

    def add(native, delta):
        result = original_add(native, delta)
        consumer_events.append(consumer_signal(native, delta, result))
        return result

    loader.validate_panel, loader.cold_source_binding = panel, binding
    consumer.add_source_ray_correction = add
    try:
        loader.main()
    finally:
        loader.validate_panel, loader.cold_source_binding = original_panel, original_binding
        consumer.add_source_ray_correction = original_add
        for item in captured:
            item.close()
    if len(captured) != 1 or len(captured[0].events) != 4 or len(consumer_events) != 4:
        raise ValueError('four actual batch8 module and consumer forwards required')
    output = Path(plan['output'])
    actual = json.loads((output/'evaluation.json').read_text())
    compare_reference(actual, reference, exact=(control == 'normal'))
    record = dict(status='PASS_ACTUAL_NATIVE480_READOUT_DIAGNOSTIC', control=control,
        native_flow_training_updates=480, source_teacher_updates=0, optimizer_updates=0,
        actual_model={k: actual[k] for k in MODEL}, assets=32,
        reference=str(reference_path), reference_SHA256=file_sha256(reference_path),
        normal_reference_records_exact=(control == 'normal'),
        paired_GT_source_noise_and_model_identities_verified=True,
        shadow_events=captured[0].events, consumer_events=consumer_events,
        semantic_prior_only_keeps_matching_score_and_removes_learned_weight_residual=True,
        top1_feature_only_is_inference_intervention_not_retrained_physical_Top1_control=True,
        wrong_values_permute_within_semantic_support_and_keep_Top1_carrier=True,
        no_uniform_NoScore_or_spatial_NoCorr_claim=True,
        no_new_quality_gain_or_model_adoption=True, full_goal_complete=False)
    actual.update(diagnostic_only=True, saved_native480_readout_intervention=record)
    (output/'evaluation.json').write_text(json.dumps(actual, indent=2)+'\n')
    (output/'intervention.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(status=record['status'], control=control, full_goal_complete=False)), flush=True)


if __name__ == '__main__':
    main()
