"""Pinned native-prior saved-weight evaluation using the existing SS protocol."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    from . import evaluate_target_first_observed as evaluation
    from .train_native_prior_source_ray import validate_native_prior_recipe
    from pixal3d.trainers.flow_matching import target_first_observed_query as original
    from pixal3d.trainers.flow_matching.native_prior_source_ray import NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer as Trainer
    saved_class=original.ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer
    saved_validate=evaluation.validate_recipe
    original.ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer=Trainer
    evaluation.validate_recipe=validate_native_prior_recipe
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        evaluation.main()
    finally:
        original.ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer=saved_class
        evaluation.validate_recipe=saved_validate
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=digest:
        raise ValueError('native-prior evaluator changed during execution')
    path=Path(sys.argv[sys.argv.index('--output')+1])/'evaluation.json'
    result=json.loads(path.read_text());native=result.pop('observed_query_target_first')
    native.pop('native_target_global_and_negative_condition')
    native.update(native_MV4_global_projection_negative_initialization=True,
        target_carrier_is_native_four_view_V1=True,
        no_query_fallback='Native pretrained MV4 prior, zero source correction',
        source_correction='sum_v Proj_v[u*g*(E-original_own_view_S)+u*R]/4',
        ffn_chunk_assets=24,FFN_eval_no_grad_native_unmodified=True,
        native_prior_adapter_SHA256=digest)
    result['native_prior_source_ray']=native
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
