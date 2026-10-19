"""Synthetic quality gates and queue recipes; never execute a GPU experiment."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting.frozen_utility_confirmation import (
    CONTROLS, NAME, QUEUE_JOB, SUMMARY_JOB, confirmation_jobs,
    confirmation_recipe, decision_supported, queue,
    validate_matched_recipe,
)


def development(positive=True):
    contrast = {'gain': .002, 'paired_cluster_bootstrap_ci95': [.001, .003],
                'per_noise': [{'gain': .002}] * 3}
    contrasts = {key: copy.deepcopy(contrast) for key in ['v1', *CONTROLS]}
    if not positive:
        contrasts['static-top1-c1']['paired_cluster_bootstrap_ci95'][0] = -.001
    return {'candidate': NAME, 'model_adopted': False, 'full_goal_complete': False,
            'first_round_negative_preserved': True, 'contrasts': contrasts, 'inputs': [],
            'status': 'PROMISING_FROZEN_UTILITY_REQUIRES_CONFIRMATION' if positive
                      else 'NO_FROZEN_UTILITY_IMPROVEMENT_CONFIRMED',
            'eligible_for_training_seed_confirmation': positive}


def source(root):
    return {'ckpt': 'none', 'load_dir': '', 'num_gpus': 2,
            'num_nodes': 1, 'models': {'denoiser': 'native'},
            'training_data': 'world-v4-train9149', 'data_dir': 'world-v4-roots',
            'dataset': {'args': {'task_matching_features': True, 'transport_arm': 'c1'}},
            'validation': {'dataset_args': {'task_matching_features': True, 'transport_arm': 'c1'},
                           'every_steps': 96},
            'default_output_dir': str(root / 'runs_ss64' / NAME),
            'trainer': {'args': {'max_steps': 480, 'batch_size_per_gpu': 48,
                                'batch_split': 1, 'deform_conditioner': {
                'implementation': 'r9-native-v5', 'arm': 'c1', 'score_temperature': .005,
                'task_attention': {'decoder': True, 'score_prior': True,
                                  'train_decoder': False, 'train_score': False,
                                  'content_mode': 'residual-zero', 'chunk_size': 64}}}},
            'experiment': {'arm': NAME, 'required_microbatch_per_gpu': 48}, 'training_seed': 101}


def test_point_estimate_without_positive_same_retargeter_CI_cannot_confirm():
    assert decision_supported(development()) is True
    assert decision_supported(development(False)) is False


@pytest.mark.parametrize('control', ['v1', *CONTROLS])
def test_each_missing_positive_control_rejects_claimed_eligibility(control):
    value = development()
    value['contrasts'][control]['per_noise'][0]['gain'] = -.001
    with pytest.raises(ValueError, match='eligibility'):
        decision_supported(value)


def test_fresh_seed_recipe_preserves_original_matching_module_and_budget(tmp_path):
    original = source(tmp_path)
    before = copy.deepcopy(original)
    cfg = confirmation_recipe(original, tmp_path)
    assert original == before and cfg['training_seed'] == 202
    assert cfg['ckpt'] == 'none' and cfg['load_dir'] == ''
    assert cfg['trainer'] == original['trainer']
    assert cfg['master_port'] == '30443'
    assert cfg['wandb_id'] == 'ss-frozen-utility-v1-c1-s202'
    assert Path(cfg['default_output_dir']) == tmp_path / 'seed-confirmation-v1/runs_ss64' / NAME


def test_confirmation_keeps_frozen_entrypoint_and_both_matched_seed_controls(tmp_path):
    jobs = confirmation_jobs(tmp_path / 'F', tmp_path / 'N', tmp_path / 'panels', ['0', '1'])
    assert len(jobs) == 6 and jobs[0]['gpus'] == ['0', '1']
    assert 'experiments.deformnext_retargeting.frozen_matcher_utility' in jobs[0]['command']
    assert jobs[0]['depends_on'] == [QUEUE_JOB]
    summary = next(job for job in jobs if job['name'] == SUMMARY_JOB)
    assert len(summary['requires']) == 18
    for control in CONTROLS:
        assert f'functional-QA-s202-{control}' in summary['depends_on']
        for noise in range(3):
            assert f'confirm-seed202-final-{control}-noise{noise}' in summary['depends_on']
    assert not any('additional-validation' in str(job) for job in jobs)


def matched_control(candidate):
    control = copy.deepcopy(candidate)
    spec = control['trainer']['args']['deform_conditioner']
    spec.pop('task_attention')
    spec['candidate_aggregation'] = 'top1'
    for key, args in [('dataset', 'args'), ('validation', 'dataset_args')]:
        control[key][args].pop('task_matching_features')
    return control


def close_synthetic_decision(root):
    path = root / 'fixed-final-decision-v1.json'
    (root / 'fixed-final-report-file4.json').write_text(json.dumps({
        'status': 'PASS', 'local_equals_actual_file4_SHA256': True,
        'files': [{'relative_path': str(path.relative_to(root.parent)),
                   'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}],
    }))


def test_extra_context_cache_is_allowed_without_changing_RGB_protocol(tmp_path):
    cfg = confirmation_recipe(source(tmp_path), tmp_path)
    validate_matched_recipe(cfg, matched_control(cfg), 'static-top1-c1')


def test_top1_dormant_temperature_and_default_semantic_mode_are_allowed(tmp_path):
    cfg = confirmation_recipe(source(tmp_path), tmp_path)
    cfg['trainer']['args']['deform_conditioner']['correspondence_mode'] = 'semantic'
    control = matched_control(cfg)
    control['trainer']['args']['deform_conditioner']['score_temperature'] = .05
    control['trainer']['args']['deform_conditioner'].pop('correspondence_mode')
    validate_matched_recipe(cfg, control, 'static-top1-c1')


def test_no_retargeter_control_has_its_declared_b2_dataset_consumer(tmp_path):
    cfg = confirmation_recipe(source(tmp_path), tmp_path)
    control = matched_control(cfg)
    control['trainer']['args']['deform_conditioner']['arm'] = 'b2'
    for key, args in [('dataset', 'args'), ('validation', 'dataset_args')]:
        control[key][args]['transport_arm'] = 'b2'
    validate_matched_recipe(cfg, control, 'top1-reference')
    control['validation']['dataset_args']['transport_arm'] = 'c1'
    with pytest.raises(ValueError, match='transport arm'):
        validate_matched_recipe(cfg, control, 'top1-reference')


@pytest.mark.parametrize('change', ['seed', 'RGB-source', 'model', 'batch', 'validation', 'retargeter'])
def test_confirmation_rejects_unmatched_training_or_generation(change, tmp_path):
    cfg = confirmation_recipe(source(tmp_path), tmp_path)
    control = matched_control(cfg)
    if change == 'seed':
        control['training_seed'] = 101
    elif change == 'RGB-source':
        control['training_data'] = 'different-rows'
    elif change == 'model':
        control['models']['denoiser'] = 'other-LoRA'
    elif change == 'batch':
        control['trainer']['args']['batch_size_per_gpu'] = 24
    elif change == 'validation':
        control['validation']['every_steps'] = 48
    else:
        control['trainer']['args']['deform_conditioner']['g_max'] = .5
    with pytest.raises(ValueError):
        validate_matched_recipe(cfg, control, 'static-top1-c1')


@pytest.mark.parametrize('devices', [['0', '0'], ['6', '7'], ['0'], ['0', '1', '2']])
def test_confirmation_cannot_use_unassigned_or_wrong_gpu_count(tmp_path, devices):
    with pytest.raises(ValueError, match='assigned physical GPUs'):
        confirmation_jobs(tmp_path, tmp_path, tmp_path, devices)


def test_negative_branch_records_failure_without_queue_or_recipe_writes(tmp_path):
    root, work = tmp_path / 'F', tmp_path / 'W'
    root.mkdir()
    work.mkdir()
    (root / 'fixed-final-decision-v1.json').write_text(json.dumps(development(False)))
    close_synthetic_decision(root)
    original_queue = b'[]\n'
    (work / 'queue-jobs.json').write_bytes(original_queue)
    args = SimpleNamespace(root=root, work_root=work, campaign_root=tmp_path / 'N',
                           previous_work=tmp_path / 'old', gpus='0,1')
    queue(args)
    assert (work / 'queue-jobs.json').read_bytes() == original_queue
    assert not (root / 'seed-confirmation-v1').exists()
    result = json.loads((root / 'seed202-queue-decision-v1.json').read_text())
    assert result['status'] == 'NO_F1_CONFIRMATION_QUEUED' and not result['full_goal_complete']
    with pytest.raises(FileExistsError):
        queue(args)


def test_positive_branch_appends_only_after_exact_control_dependencies_exist(tmp_path):
    root, work = tmp_path / 'F', tmp_path / 'W'
    (root / 'configs').mkdir(parents=True)
    work.mkdir()
    (root / 'fixed-final-decision-v1.json').write_text(json.dumps(development()))
    close_synthetic_decision(root)
    (root / 'configs' / (NAME + '.json')).write_text(json.dumps(source(root)))
    names = [QUEUE_JOB] + [f'confirm-seed202-final-{arm}-noise{n}' for arm in CONTROLS for n in range(3)]
    names += ['functional-QA-s202-' + arm for arm in CONTROLS]
    original = [{'name': name} for name in names]
    (work / 'queue-jobs.json').write_text(json.dumps(original))
    queue(SimpleNamespace(root=root, work_root=work, campaign_root=tmp_path / 'N',
                          previous_work=tmp_path / 'old', gpus='0,1'))
    jobs = json.loads((work / 'queue-jobs.json').read_text())
    assert jobs[:len(original)] == original and len(jobs) == len(original) + 6
    assert json.loads((root / 'seed-confirmation-v1/configs' / (NAME + '.json')).read_text())['training_seed'] == 202
    result = json.loads((root / 'seed202-queue-decision-v1.json').read_text())
    assert result['status'] == 'FROZEN_UTILITY_CONFIRMATION_QUEUED'
    assert not result['model_adopted'] and not result['full_goal_complete']


def test_changed_decision_cannot_queue_even_when_declaring_positive_evidence(tmp_path):
    root = tmp_path / 'F'
    root.mkdir()
    path = root / 'fixed-final-decision-v1.json'
    path.write_text(json.dumps(development(False)))
    close_synthetic_decision(root)
    path.write_text(json.dumps(development()))
    with pytest.raises(ValueError, match='actual file4 development decision'):
        queue(SimpleNamespace(root=root, campaign_root=tmp_path / 'N'))
