"""CPU checkpoint evidence about learning and frozen matching, without inference claims."""
import torch


def audit_source_weights(initial, current, head_names):
    if len(head_names) != 9 or len(set(head_names)) != 9:
        raise ValueError('nine actual learned Q/K/V/output/utility tensors required')
    frozen = [name for name in initial if name.startswith('q2c.') or name.startswith('ray_readout.initial_')
              or name == 'ray_readout.residual_logit_bound']
    if (not any(n.startswith('q2c.') for n in frozen)
            or len([n for n in frozen if n.startswith('ray_readout.initial_')]) != 7
            or set(initial) != set(current)
            or any(initial[n].dtype != current[n].dtype or not torch.equal(initial[n], current[n]) for n in frozen)):
        raise ValueError('frozen matching decoder/score or actual seven reference anchors changed')
    rows = {}
    for name in head_names:
        before, after = initial[name].float(), current[name].float()
        if before.shape != after.shape or not torch.isfinite(after).all():
            raise ValueError('finite same-shaped learned head weights required')
        delta = after - before
        norm = float(before.norm())
        rows[name] = dict(initial_L2=norm, current_L2=float(after.norm()), delta_L2=float(delta.norm()),
            relative_delta_L2=None if norm == 0 else float(delta.norm()) / norm,
            changed_elements=int((before != after).sum()), total_elements=before.numel())
    return dict(status='PASS_ACTUAL_NATIVE_SOURCE_CHECKPOINT_WEIGHT_AUDIT',
        frozen_matching_decoder_score_and_buffers_exact=True, frozen_head_reference_anchors_exact=True,
        learned_head_tensors=rows, learned_attention_tensors_changed=sum(r['changed_elements'] > 0 for r in rows.values()),
        R_output_projection_L2=float(current['retargeter.net.2.weight'].float().norm()),
        semantic_fusion_scale=float(current['fusion_gate'].float().tanh()),
        maximum_declared_pairwise_log_odds_change=2 * float(current['ray_readout.residual_logit_bound']),
        attention_distributions_not_evaluated=True, matching_accuracy_not_evaluated=True,
        decoder_execution_or_causal_quality_effect_not_proven_by_weights=True,
        native_denoiser_checkpoint_not_loaded=True, optimizer_updates=0, quality_gain_proven=False, full_goal_complete=False)
