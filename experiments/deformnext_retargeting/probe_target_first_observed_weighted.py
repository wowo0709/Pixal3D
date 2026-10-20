"""Run the unchanged observed-query native proof with fixed-score readout."""


def main():
    from pixal3d.trainers.flow_matching import target_first_observed_query as original
    from pixal3d.trainers.flow_matching.target_first_observed_weighted import (
        ObservedQueryWeightedTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer,
    )
    from . import probe_target_first_observed_query as proof

    saved = original.ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer
    original.ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer = (
        ObservedQueryWeightedTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer)
    try:
        proof.main()
    finally:
        original.ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer = saved


if __name__ == '__main__':
    main()
