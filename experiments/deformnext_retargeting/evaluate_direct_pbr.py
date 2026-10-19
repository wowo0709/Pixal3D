"""Read-only streamed PBR evaluation of the physical-query/direct module.

Retain the existing paired RGB/noise/GT Shape renderer. Adapt only the metric
dispatch name, so the frozen streaming evaluator retains its own input audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--native-PBR-proof',type=Path,required=True)
    local,remaining = parser.parse_known_args()
    from pixal3d import datasets,trainers
    from pixal3d.datasets.deformnext_deformation_query_pbr import DeformNeXtDeformationQueryPBR1024
    from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner
    from pixal3d.trainers.flow_matching.sparse_flow_matching import ImageConditionedProjSparseFlowMatchingCFGTrainer
    from . import quick_metrics, evaluate_pbr_streaming

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    proof = json.loads(local.native_PBR_proof.read_text())
    if (proof['status'] != 'PASS_ACTUAL_DIRECT_PBR_PHYSICAL_QUERY_IDENTITY_FIRST_DINO_NAF_AND16_GRADIENTS'
            or not proof['initial_evidence_exact'] or proof['physical_retargeter_input_channels']!=5120):
        raise ValueError('actual native Direct PBR proof required')
    bound = dict(proof['source_SHA256'])
    bound[str(Path(__file__).resolve())] = sha(__file__)
    if any(sha(path)!=digest for path,digest in bound.items()):
        raise ValueError('native PBR proof or evaluator source changed')
    captured = []

    class DirectPBREvaluationTrainer(ImageConditionedProjSparseFlowMatchingCFGTrainer):
        def init_models_and_more(self,*,direct_utility_checkpoint,**kwargs):
            if set(self.models)!={'denoiser'} or self.deform_conditioner is not None:
                raise ValueError('complete native PBR NN evaluation required')
            denoiser = self.models.pop('denoiser')
            denoiser.requires_grad_(False)
            module = DirectRayRetargetingConditioner(**self.deform_conditioner_config).cuda()
            module.ray_readout.score_selector.load_state_dict(
                torch.load(direct_utility_checkpoint,map_location='cpu',weights_only=True),strict=True)
            module.retargeter.requires_grad_(True)
            module.fusion_gate.requires_grad_(True)
            self.deform_conditioner = module
            self.models['deform_conditioner'] = module
            super().init_models_and_more(**kwargs)
            self.models['denoiser'] = denoiser
            self.training_models['denoiser'] = denoiser
            if len(self.model_params)!=16 or module.retargeter.input_dim!=5120:
                raise ValueError('real physical5120 PBR module with16 evaluation-only registered tensors required')
            captured.append(self)

    if remaining[remaining.index('--stage')+1]!='pbr1024':
        raise ValueError('this entry supports PBR1024 only')
    path = Path(remaining[remaining.index('--arm-config')+1])
    config = json.loads(path.read_text())
    if (config['trainer']['name']!='DirectPBREvaluationTrainer'
            or config['dataset']['name']!='DeformNeXtDeformationQueryPBR1024'
            or config['validation']['dataset_name']!='DeformNeXtDeformationQueryPBR1024'):
        raise ValueError('explicit Direct PBR model/dataset schema required')
    bound[str(path)] = sha(path)
    datasets.DeformNeXtDeformationQueryPBR1024 = DeformNeXtDeformationQueryPBR1024
    trainers.DirectPBREvaluationTrainer = DirectPBREvaluationTrainer
    original = quick_metrics.evaluate_quick

    def physical_dispatch(trainer,dataset,config):
        if config['dataset_name']!='DeformNeXtDeformationQueryPBR1024' or dataset.deformation_query_channels!=2048:
            raise ValueError('this PBR evaluation must retain the actual physical query dataset')
        return original(trainer,dataset,dict(config,dataset_name='DeformNeXtPBR1024'))

    quick_metrics.evaluate_quick = physical_dispatch
    previous = sys.argv
    try:
        sys.argv = [str(evaluate_pbr_streaming.__file__),*remaining]
        evaluate_pbr_streaming.main()
    finally:
        sys.argv = previous
        quick_metrics.evaluate_quick = original
    output = Path(remaining[remaining.index('--output')+1])
    path = output/'evaluation.json'
    result = json.loads(path.read_text())
    if len(captured)!=1 or result['optimizer_updates_during_evaluation']!=0:
        raise ValueError('one frozen native PBR evaluation with zero updates required')
    if any(sha(path)!=digest for path,digest in bound.items()):
        raise ValueError('Direct PBR source/config changed during evaluation')
    module = captured[0].deform_conditioner
    result['direct_physical_PBR'] = dict(physical_input_channels=5120,deformation_channels=2048,
        ray_query='edited-target-patch-ray',semantic_Top1_carrier_retained=True,
        direct_feature_bound=.25,matching_temperature=module.score_temperature,
        head_output_projection_nonzero=int(torch.count_nonzero(module.ray_readout.output.weight)),
        DQ_projection_nonzero=int(torch.count_nonzero(module.retargeter.deformation_projection.weight)),
        source_SHA256=bound,optimizer_updates=0,GT_matching_input=False,
        standard_GT_Shape_condition=True,source_preservation_metric=False,full_cascade=False)
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
