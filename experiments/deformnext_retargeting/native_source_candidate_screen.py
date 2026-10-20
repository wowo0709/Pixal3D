"""Development-only screen for an independent training-seed confirmation.

Three generation noises are paired repeats, not independent training runs.
This screen retains the earlier requirement of a positive source-cluster CI
against V1 and matched Top1 controls, with positive effects at every noise.
It never adopts a full model or reads held-out test predictions.
"""
from .compare_ss_noise_panels import compare


def screen(evaluations, candidates, *, draws=4000):
    """Screen named SS480 evaluations using explicit, frozen control lists.

    ``evaluations[name]`` contains the three noise reports. Candidate specs
    declare ``required_controls`` and may declare ``diagnostic_controls``.
    Diagnostic comparisons remain separate from eligibility: for example,
    outperforming semantic Top1 does not establish a semantic-versus-spatial
    correspondence effect, or a matching decoder's isolated contribution.
    """
    if not evaluations or not candidates or draws <= 0:
        raise ValueError('nonempty evaluations/candidates and positive draws required')
    for name, reports in evaluations.items():
        if len(reports) != 3:
            raise ValueError(f'exactly three generation-noise reports required: {name}')
        for report in reports:
            if (report.get('status') != 'PASS' or report.get('stage') != 'ss64'
                    or report.get('step') != 480
                    or report.get('optimizer_updates_during_evaluation') != 0):
                raise ValueError(f'actual saved SS480 evaluation required: {name}')
    evidence = {}
    for name, spec in candidates.items():
        required = spec['required_controls']
        diagnostic = spec.get('diagnostic_controls', [])
        controls = required + diagnostic
        if (name not in evaluations or not required or 'v1' not in required
                or len(controls) != len(set(controls))
                or any(c == name or c not in evaluations for c in controls)):
            raise ValueError(f'distinct present candidate/controls including V1 required: {name}')
        comparisons = {}
        candidate_iou = None
        for control in controls:
            result = compare([(str(noise), evaluations[control][noise], evaluations[name][noise])
                              for noise in range(3)], draws=draws)
            if candidate_iou is not None and candidate_iou != result['candidate_iou']:
                raise ValueError('candidate population differs between controls')
            candidate_iou = result['candidate_iou']
            comparisons[control] = dict(
                result=result,
                positive_source_cluster_CI=result['paired_cluster_bootstrap_ci95'][0] > 0,
                positive_at_all_three_noises=all(row['gain'] > 0 for row in result['per_noise']),
                required_for_eligibility=control in required,
            )
        eligible = all(comparisons[c]['positive_source_cluster_CI']
                       and comparisons[c]['positive_at_all_three_noises'] for c in required)
        evidence[name] = dict(
            eligible_for_independent_training_confirmation=eligible,
            candidate_micro_IoU=candidate_iou,
            required_controls=list(required), diagnostic_controls=list(diagnostic),
            comparisons=comparisons,
        )
    eligible_names = [name for name, row in evidence.items()
                      if row['eligible_for_independent_training_confirmation']]
    # Highest fixed-final micro IoU, with a stable name tie-break; no best epoch.
    selected = min(eligible_names, key=lambda name: (-evidence[name]['candidate_micro_IoU'], name)) if eligible_names else None
    return dict(
        status=('PROMISING_NATIVE_SOURCE_COMBINATION_FOR_SEED_CONFIRMATION'
                if selected else 'NO_PROMISING_NATIVE_SOURCE_COMBINATION_YET'),
        selected_candidate=selected, candidate_evidence=evidence,
        selection_rule='Positive source-cluster CI against V1 and every declared matched Top1/additive control; positive at all3 generation noises; highest fixed SS480 micro IoU, stable name tie-break.',
        source_training_seed=101, generation_noise_repeats=3,
        intervals_exploratory_uncorrected_for_multiple_comparisons=True,
        independent_training_confirmation=False, heldout_test_predictions_read=False,
        full_cascade_quality_gain_proven=False, isolated_matching_decoder_effect_proven=False,
        model_adopted=False, full_goal_complete=False,
        if_no_candidate='Preserve all negative evidence and continue model diagnosis/design; do not mark the goal complete.',
    )
