"""Native32 saved-weight checks through an isolated process-local constructor.

The existing source evaluator, trainer registry, checkpoint loader, calibrated
RGB inputs and sampler remain the same. The constructor is restored on exit.
"""
import argparse
import json
from pathlib import Path
import sys

from . import r_mediated_source_pilot as pilot
from .r_mediated_interventions import CONTROLS, SavedRMediatedIntervention


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--intervention', choices=CONTROLS, required=True)
    local, remaining = parser.parse_known_args()
    if remaining[remaining.index('--appearance-mode')+1] != 'raw-topk-plus-feature':
        raise ValueError('fixed trained raw-TopK appearance required')
    if '--require-native-identity' in remaining or remaining[remaining.index('--step')+1] != '112':
        raise ValueError('trained step112; normal-shadow proof is distinct from cold identity')
    previous = pilot.build_conditioner
    captured = []

    def build(*args, **kwargs):
        module = previous(*args, **kwargs)
        captured.append(SavedRMediatedIntervention(module, local.intervention))
        return module

    argv = sys.argv
    pilot.build_conditioner = build
    try:
        from .evaluate_r_mediated_source_teacher_pilot import main as native_main
        sys.argv = [argv[0], *remaining]
        native_main()
        if len(captured) != 1 or not captured[0].events:
            raise ValueError('actual native conditioner forwards were not observed')
        path = Path(remaining[remaining.index('--output')+1]) / 'evaluation.json'
        result = json.loads(path.read_text())
        result['diagnostic_only'] = True
        result['saved_R_mediated_intervention'] = dict(control=local.intervention,
            same_weights_step112=True, inference_only_not_retrained_ablation=True,
            NoCorr_spatial_training_arm_not_implemented_by_this_diagnostic=True,
            semantic_prior_only_retains_matching_score_not_uniform=True,
            Top1_feature_only_selects_one_correspondence_not_uniform=True,
            shadow_events=captured[0].events, source_teacher_or_GT_features_not_loaded=True,
            new_training_updates=0, model_adopted=False, full_goal_complete=False)
        path.write_text(json.dumps(result, indent=2)+'\n')
    finally:
        pilot.build_conditioner = previous
        sys.argv = argv
        for item in captured:
            item.close()


if __name__ == '__main__':
    main()
