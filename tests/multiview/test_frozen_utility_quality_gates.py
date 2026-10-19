"""Reject quality decisions based on unfrozen, partial, or unlogged runs."""
import copy

import pytest

from experiments.deformnext_retargeting.summarize_frozen_utility_followup import validate_saved_audits
from experiments.deformnext_retargeting.frozen_matcher_utility import validate_profile


@pytest.mark.parametrize('cfg', [{'ckpt': None, 'load_dir': None},
                                 {'ckpt': 'none', 'load_dir': None},
                                 {'ckpt': 'latest', 'load_dir': ''}])
def test_invalid_fresh_resume_options_fail_before_loading_any_model(cfg, tmp_path):
    with pytest.raises(ValueError, match='fresh utility recipe'):
        validate_profile(cfg, tmp_path)


@pytest.fixture
def complete_evidence():
    final = {
        'status': 'PASS_ACTUAL_FINAL_TENSORS_AND_OPTIMIZER',
        'frozen_base_tensors_exact': 701, 'LoRA_tensors_updated': 360,
        'matching_tensors_present': {'decoder': 18, 'score': 6},
        'matching_tensors_changed': {'decoder': [], 'score': []},
        'physical_deformation_query_channels': 0, 'retargeter_input_channels': 2048,
        'optimizer_states': 371, 'actual_optimizer_step': 480,
        'all1061_denoiser_tensors_finite': True, 'all_conditioner_tensors_finite': True,
        'all_optimizer_moments_finite': True,
        'task_attention_tensors_changed': ['content_output.weight'],
    }
    qa = {
        'status': 'PASS_TRAINED_SS_CACHE_AMP_CFG_CONTRACT', 'optimizer_updates': 0,
        'saved_frozen_head_exact_to_selected_checkpoint': True,
        'frozen_head_state_keys': list(range(24)),
        'actual_trainable_matcher_saved_state_changed_keys': [],
    }
    epochs = [{
        'status': 'PASS_ACTUAL_EPOCH_OPTIMIZER_LOGGING_AND_BATCH',
        'epoch': number, 'step': 96 * number, 'training_seed': 101,
        'global_batch': 96, 'world_size': 2, 'microbatch_per_rank': 48,
        'accumulation': 1, 'actual_optimizer_states': 371,
        'all_optimizer_steps': 96 * number,
        'actual_remote_four_numeric_fields_verified': True,
        'actual_remote_PNG_byte_exact': True,
    } for number in range(1, 6)]
    return final, qa, epochs


def test_complete_saved_evidence_can_reach_quality_comparison(complete_evidence):
    validate_saved_audits(*complete_evidence)


@pytest.mark.parametrize('change', ['unfrozen', 'no-utility-learning', 'partial-epochs',
                                   'wrong-budget', 'local-only-logging', 'GPU-proof-with-SGD', 'nonfinite'])
def test_incomplete_or_different_execution_cannot_claim_quality(complete_evidence, change):
    final, qa, epochs = copy.deepcopy(complete_evidence)
    if change == 'unfrozen':
        final['matching_tensors_changed']['decoder'] = ['query_projection.weight']
    elif change == 'no-utility-learning':
        final['task_attention_tensors_changed'] = ['query.weight']
    elif change == 'partial-epochs':
        epochs.pop()
    elif change == 'wrong-budget':
        epochs[2]['microbatch_per_rank'] = 24
    elif change == 'local-only-logging':
        epochs[-1]['actual_remote_PNG_byte_exact'] = False
    elif change == 'GPU-proof-with-SGD':
        qa['optimizer_updates'] = 1
    elif change == 'nonfinite':
        final['all_optimizer_moments_finite'] = False
    with pytest.raises(ValueError):
        validate_saved_audits(final, qa, epochs)
