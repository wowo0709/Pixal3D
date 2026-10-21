"""Evaluate a declared source-teacher variant on a fixed confirmation subset."""
import argparse
import json
from pathlib import Path
import runpy
import sys

import torch


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--pilot-variant', choices=['original-NoDQ', 'source-reference-NoDQ', 'source-contrast-NoDQ'], required=True)
    parser.add_argument('--require-native-identity', action='store_true')
    parser.add_argument('--expected-assets', type=int, choices=[72, 128], required=True)
    local, remaining = parser.parse_known_args()
    if remaining[remaining.index('--stage')+1] != 'ss64':
        raise ValueError('source-feature pilot supports native SS only')
    if any(flag in remaining for flag in ['--conditioning-mode', '--transport-control',
                                         '--r9-path-control', '--task-value-control']):
        raise ValueError('unchanged M1target8/M0source9,10,11 RGB input required')
    from pixal3d import trainers
    from pixal3d.modules.source_contrast_member_feature import SourceContrastMemberFeatureConditioner
    from pixal3d.trainers.flow_matching.member_feature_source_ray import verify_utility
    from pixal3d.trainers.flow_matching.member_posterior_source_ray import (
        MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer as Parent,
    )
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    from .member_feature_recipe import validate_feature_recipe

    cfg_path = Path(remaining[remaining.index('--arm-config')+1])
    cfg = json.loads(cfg_path.read_text()); validate_feature_recipe(cfg)
    from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
    from pixal3d.modules.source_referenced_member_feature import SourceReferencedMemberFeatureConditioner
    module_class = {
        "original-NoDQ": MemberFeatureSourceRayConditioner,
        "source-reference-NoDQ": SourceReferencedMemberFeatureConditioner,
        "source-contrast-NoDQ": SourceContrastMemberFeatureConditioner,
    }[local.pilot_variant]
    parameter_count = 20 if local.pilot_variant == "source-reference-NoDQ" else 19
    captured, identity = [], []

    class Contrasted(Parent):
        def init_models_and_more(self, *, ray_feature_utility, **kwargs):
            digest = verify_utility(ray_feature_utility)
            if self.image_cond_model is None: self._init_image_cond_model()
            if self.deform_conditioner is None:
                self.deform_conditioner = module_class(**self.deform_conditioner_config).cuda()
                self.deform_conditioner.load_utility(torch.load(ray_feature_utility['checkpoint'],
                    map_location='cpu', weights_only=True))
                self.models['deform_conditioner'] = self.deform_conditioner
            super().init_models_and_more(**kwargs)
            if sum(p.requires_grad for p in self.deform_conditioner.parameters()) != parameter_count:
                raise ValueError('complete nineteen-tensor source-contrast conditioner required')
            self._ray_feature_utility_initialization = dict(SHA256=digest, epochs=7,
                utility_updates=112, source_teacher_variant=local.pilot_variant, GT_inference_input=False)

    selected = Contrasted

    class Evaluation(selected):
        def init_models_and_more(self, **kwargs):
            super().init_models_and_more(**kwargs); captured.append(self)

        def _encode_deform_views(self, images, camera, transport, **kwargs):
            result, negative = super()._encode_deform_views(images, camera, transport, **kwargs)
            if local.require_native_identity:
                cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
                native, native_negative = ImageConditionedProjMixin._encode_deform_views(self, images, camera, None)
                equality = {key: bool(torch.equal(value, native[key])) for key, value in result.items()}
                equality.update({'negative_'+key: bool(torch.equal(value, native_negative[key]))
                                 for key, value in negative.items()})
                if (not all(equality.values()) or not torch.equal(cpu_rng, torch.random.get_rng_state())
                        or not torch.equal(cuda_rng, torch.cuda.get_rng_state())):
                    raise ValueError('cold module must preserve actual native global/proj/negative/RNG exactly')
                identity.append(equality)
            return result, negative

    name = cfg['trainer']['name']; previous = trainers.__dict__.get(name)
    argv = sys.argv
    entry = Path(__file__).with_name('evaluate_deterministic_checkpoint.py')
    try:
        setattr(trainers, name, Evaluation)
        sys.argv = [str(entry), *remaining]
        runpy.run_path(str(entry), run_name='__main__')
    finally:
        sys.argv = argv
        if previous is None: delattr(trainers, name)
        else: setattr(trainers, name, previous)
    root = Path(remaining[remaining.index('--output')+1]); path = root/'evaluation.json'
    result = json.loads(path.read_text())
    if (len(captured) != 1 or len(captured[0].optimizer.state) != 0
            or (local.pilot_variant == 'source-contrast-NoDQ' and int(captured[0].deform_conditioner.retargeter.source_contrast_schema.item()) != 1)
            or len(result['per_asset']) != local.expected_assets or result['input_camera_ids'] != [8, 9, 10, 11]
            or result['input_states'] != ['M1', 'M0', 'M0', 'M0']
            or result['sampling_steps'] != 12 or result['guidance'] != 3.
            or (local.require_native_identity and not identity)):
        raise ValueError('actual requested subset/noise native frozen-SS evaluation without updates required')
    result['source_feature_teacher_pilot'] = dict(variant=local.pilot_variant,
        source_tensors=parameter_count,
        extra_source_reference_channels=1024 if local.pilot_variant == 'source-reference-NoDQ' else 0,
        retargeter_input_mode=('matched_plus_head_change_minus_native_source' if local.pilot_variant == 'source-contrast-NoDQ' else 'matched_plus_head_change_and_target'),
        source_contrast_checkpoint_marker=local.pilot_variant == 'source-contrast-NoDQ',
        actual_subset_assets=local.expected_assets,
        training_teacher_not_loaded_during_inference=True, original_multiview_SS_checkpoint_step=4210,
        original_SS_denoiser_not_jointly_finetuned=True, generation_uses_live_RGB_BF16_native_condition=True,
        cold_native_tensor_identity_required=local.require_native_identity, actual_cold_identity_events=identity,
        process_local_trainer_binding=True, frozen_registry_and_running_jobs_unchanged=True,
        full9149_joint_experiment_not_replaced=True, model_adopted=False, full_goal_complete=False)
    path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
