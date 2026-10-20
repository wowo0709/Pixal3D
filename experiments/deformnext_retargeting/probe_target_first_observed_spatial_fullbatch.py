"""Run the unchanged disposable SS96 profile with native spatial attention."""


def main():
    from pixal3d.trainers.flow_matching import target_first_observed_query as original
    from pixal3d.trainers.flow_matching.target_first_observed_spatial import (
        ObservedQuerySpatialTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer,
    )
    from . import probe_target_first_fullbatch as proof

    saved = original.ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer
    original.ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer = (
        ObservedQuerySpatialTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer)
    try:
        proof.main()
    finally:
        original.ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer = saved


if __name__ == '__main__':
    main()
