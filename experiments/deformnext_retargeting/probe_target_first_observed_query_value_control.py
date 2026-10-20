"""Unchanged native fixed-use intervention with the observed-query trainer."""


def main():
    from pixal3d.trainers.flow_matching import target_first_ray_v2 as original
    from pixal3d.trainers.flow_matching.target_first_observed_query import (
        ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer,
    )
    from . import probe_target_first_value_control as proof

    saved = original.TargetFirstRayImageConditionedProjFlowMatchingCFGTrainerV2
    original.TargetFirstRayImageConditionedProjFlowMatchingCFGTrainerV2 = (
        ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer)
    try:
        proof.main()
    finally:
        original.TargetFirstRayImageConditionedProjFlowMatchingCFGTrainerV2 = saved


if __name__ == '__main__':
    main()
