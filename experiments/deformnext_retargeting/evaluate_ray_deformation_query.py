"""Read-only pinned SS evaluation of physical target-ray deformation query."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import torch


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--DQ-identity-proof',type=Path,required=True)
    parser.add_argument('--expected-assets',type=int,required=True)
    local, remaining = parser.parse_known_args()
    from pixal3d import trainers
    from pixal3d.modules.ray_deformation_query import RayDeformationQueryConditioner
    from pixal3d import datasets
    from pixal3d.datasets.deformnext_deformation_query import DeformNeXtDeformationQuerySS64
    datasets.DeformNeXtDeformationQuerySS64 = DeformNeXtDeformationQuerySS64
    from pixal3d.trainers.flow_matching.flow_matching import ImageConditionedProjFlowMatchingCFGTrainer

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    proof = json.loads(local.DQ_identity_proof.read_text())
    if proof['status'] != 'PASS_ACTUAL_SS_PHYSICAL_DEFORMATION_QUERY_IDENTITY_AND_TEN_TENSOR_GRADIENT':
        raise ValueError('actual SS feature identity/gradient proof required')
    bound = dict(proof['source_SHA256'])
    bound[str(Path(__file__).resolve())] = sha(__file__)
    if any(sha(path) != digest for path,digest in bound.items()):
        raise ValueError('feature identity or evaluator source changed')
    captured = []

    class RayDeformationEvaluationTrainer(ImageConditionedProjFlowMatchingCFGTrainer):
        def init_models_and_more(self, *, anchor_reference_checkpoint,
                                 feature_utility_checkpoint, **kwargs):
            if set(self.models) != {'denoiser'} or self.deform_conditioner is not None:
                raise ValueError('complete native frozen SS evaluation required')
            denoiser = self.models.pop('denoiser')
            denoiser.requires_grad_(False)
            module = RayDeformationQueryConditioner(**self.deform_conditioner_config).cuda()
            module.load_reference(torch.load(anchor_reference_checkpoint,map_location='cpu',weights_only=True))
            module.ray_readout.score_selector.load_state_dict(
                torch.load(feature_utility_checkpoint,map_location='cpu',weights_only=True),strict=True)
            self.deform_conditioner = module
            self.models['deform_conditioner'] = module
            super().init_models_and_more(**kwargs)
            self.models['denoiser'] = denoiser
            self.training_models['denoiser'] = denoiser
            if len(self.model_params) != 10:
                raise ValueError('exactly nine attention tensors plus one deformation projection may be registered')
            captured.append(self)

    if '--stage' not in remaining or remaining[remaining.index('--stage')+1] != 'ss64':
        raise ValueError('this verified entry only supports SS64')
    config_path = Path(remaining[remaining.index('--arm-config')+1])
    config = json.loads(config_path.read_text())
    if config['trainer']['name'] != 'RayDeformationEvaluationTrainer':
        raise ValueError('explicit evaluation constructor required')
    bound[str(config_path)] = sha(config_path)
    output = Path(remaining[remaining.index('--output')+1])
    if output.exists():
        raise FileExistsError('fresh evaluation namespace required')
    trainers.RayDeformationEvaluationTrainer = RayDeformationEvaluationTrainer
    previous = sys.argv
    try:
        entry = Path(__file__).with_name('evaluate_deterministic_checkpoint.py')
        sys.argv = [str(entry), *remaining]
        runpy.run_path(str(entry),run_name='__main__')
    finally:
        sys.argv = previous
        delattr(trainers,'RayDeformationEvaluationTrainer')
    result_path = output/'evaluation.json'
    result = json.loads(result_path.read_text())
    if (len(captured) != 1 or len(result['per_asset']) != local.expected_assets
            or result['optimizer_updates_during_evaluation'] != 0
            or any(sha(path) != digest for path,digest in bound.items())):
        raise ValueError('complete read-only pinned evaluation required')
    trainer = captured[0]
    result['ray_deformation_query'] = dict(query='edited-target-patch-ray',
        original_nearest_carrier=True,scalar_gate=False,feature_tensors=9,deformation_tensors=1,
        physical_deformation_channels=2048,retargeter_input_channels=trainer.deform_conditioner.retargeter.input_dim,
        deformation_projection_nonzero=int(torch.count_nonzero(trainer.deform_conditioner.retargeter.deformation_projection.weight)),
        output_projection_nonzero=int(torch.count_nonzero(trainer.deform_conditioner.ray_readout.output.weight)),
        initial_identity_proof=str(local.DQ_identity_proof),source_SHA256=bound,
        source_registration='explicit process-local;frozen registry unchanged',
        optimizer_updates=0,GT_in_model_inputs=False)
    result_path.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
