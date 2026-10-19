"""Synthetic confirmation/config checks only; never start a real GPU job."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting import frozen_utility_confirmation as common
from experiments.deformnext_retargeting.frozen_utility_strength import NAME, NAMESPACE, recipe, validate_profile
from experiments.deformnext_retargeting.utility_strength_confirmation import (
    AUDIT_JOB, QUEUE_JOB, SUMMARY_JOB, STOP_NAME, auditor_source,
    confirmation_jobs, confirmation_recipe, queue,
)


def source(root):
    value = {'ckpt': 'none', 'load_dir': '', 'num_gpus': 2, 'num_nodes': 1,
        'models': {'denoiser': 'native'}, 'training_data': 'world-v4-train9149',
        'data_dir': 'world-v4-roots', 'training_seed': 101,
        'dataset': {'args': {'task_matching_features': True, 'transport_arm': 'c1'}},
        'validation': {'dataset_args': {'task_matching_features': True, 'transport_arm': 'c1'}, 'every_steps': 96},
        'default_output_dir': str(root / 'runs_ss64/frozen-utility-c1'),
        'trainer': {'args': {'max_steps': 480, 'batch_size_per_gpu': 48, 'batch_split': 1,
            'continuation_ckpt': {'denoiser': 'original-SS4210.pt'},
            'deform_conditioner': {'implementation': 'r9-native-v5', 'arm': 'c1', 'score_temperature': .005,
                'q2c_checkpoint': 'original-R9.pt',
                'task_attention': {'decoder': True, 'score_prior': True, 'train_decoder': False,
                    'train_score': False, 'content_mode': 'residual-zero', 'chunk_size': 64}}}},
        'experiment': {'arm': 'frozen-utility-c1', 'required_microbatch_per_gpu': 48}}
    return recipe(value, root)


def development(positive=True):
    contrast = {'gain': .002, 'paired_cluster_bootstrap_ci95': [.001, .003],
                'per_noise': [{'gain': .002}] * 3}
    values = {key: copy.deepcopy(contrast) for key in ['v1', *common.CONTROLS]}
    if not positive:
        values['static-top1-c1']['paired_cluster_bootstrap_ci95'][0] = -.001
    return {'candidate': NAME, 'model_adopted': False, 'full_goal_complete': False,
        'first_round_negative_preserved': True, 'contrasts': values, 'inputs': [],
        'status': 'PROMISING_FROZEN_UTILITY_REQUIRES_CONFIRMATION' if positive
                  else 'NO_FROZEN_UTILITY_IMPROVEMENT_CONFIRMED',
        'eligible_for_training_seed_confirmation': positive}


def close_decision(root, campaign):
    path = root / 'fixed-final-decision-v1.json'
    (root / 'fixed-final-report-file4.json').write_text(json.dumps({
        'status': 'PASS', 'local_equals_actual_file4_SHA256': True,
        'files': [{'relative_path': str(path.relative_to(campaign.parent)),
                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}]}))


def control(candidate, arm):
    result = copy.deepcopy(candidate)
    spec = result['trainer']['args']['deform_conditioner']
    spec.pop('task_attention')
    spec['candidate_aggregation'] = 'top1'
    spec['score_temperature'] = .05
    spec['arm'] = 'c1' if arm == 'static-top1-c1' else 'b2'
    for key, arg in [('dataset', 'args'), ('validation', 'dataset_args')]:
        result[key][arg].pop('task_matching_features')
        result[key][arg]['transport_arm'] = spec['arm']
    return result


def test_seed202_recipe_preserves_model_optimizer_and_original_continuation(tmp_path):
    root = tmp_path / NAMESPACE
    original = source(root)
    before = copy.deepcopy(original)
    cfg = confirmation_recipe(original, root)
    assert original == before and cfg['trainer'] == original['trainer']
    assert cfg['training_seed'] == cfg['experiment']['training_seed'] == 202
    assert cfg['master_port'] == '30445'
    assert cfg['wandb_id'] == 'ss-frozen-utility-scale16-c1-s202'
    assert cfg['config'] == cfg['experiment']['recipe_path'] == str(root / 'seed-confirmation-v1/configs' / (NAME + '.json'))
    assert cfg['output_dir'] == cfg['default_output_dir'] == str(root / 'seed-confirmation-v1/runs_ss64' / NAME)
    validate_profile(cfg, root / 'seed-confirmation-v1')
    for arm in common.CONTROLS:
        common.validate_matched_recipe(cfg, control(cfg, arm), arm)


@pytest.mark.parametrize('change', ['parent-root', 'metadata-seed', 'wrong-root', 'seed101-in-confirmation'])
def test_seed202_never_silently_replaces_the_seed101_profile(tmp_path, change):
    root = tmp_path / NAMESPACE
    cfg = confirmation_recipe(source(root), root)
    selected = root / 'seed-confirmation-v1'
    if change == 'parent-root':
        selected = root
    elif change == 'metadata-seed':
        cfg['experiment']['training_seed'] = 101
    elif change == 'wrong-root':
        selected = tmp_path / 'other/seed-confirmation-v1'
    else:
        cfg['training_seed'] = cfg['experiment']['training_seed'] = 101
    with pytest.raises(ValueError):
        validate_profile(cfg, selected)


def test_development_gate_is_bound_to_candidate_and_same_retargeter_gain():
    assert common.decision_supported(development(), candidate_name=NAME)
    assert not common.decision_supported(development(False), candidate_name=NAME)
    with pytest.raises(ValueError):
        common.decision_supported(development())
    value = development()
    value['contrasts']['static-top1-c1']['per_noise'][0]['gain'] = -.001
    with pytest.raises(ValueError):
        common.decision_supported(value, candidate_name=NAME)


def test_replica_jobs_require_both_controls_real_epoch_closures_and_deferred_final_audit(tmp_path):
    jobs = confirmation_jobs(tmp_path / NAMESPACE, tmp_path, tmp_path / 'old', ['2', '3'],
        allowed_gpus=['0', '1', '2', '3'], work_root=tmp_path / 'work',
        audit_driver=tmp_path / 'driver.py', audit_implementation=tmp_path / 'audit.py', audit_sha='a' * 64)
    assert len(jobs) == 7 and jobs[0]['gpus'] == ['2', '3']
    assert jobs[0]['depends_on'] == [QUEUE_JOB]
    assert 'experiments.deformnext_retargeting.frozen_utility_strength' in jobs[0]['command']
    audit = next(j for j in jobs if j['name'] == AUDIT_JOB)
    assert audit['resource'] == 'cpu' and audit['depends_on'] == ['confirm-seed202-train-' + NAME]
    assert any('epoch05/file4-closure.json' in value for value in audit['requires'])
    assert str(tmp_path / 'work' / STOP_NAME) in audit['command']
    summary = next(j for j in jobs if j['name'] == SUMMARY_JOB)
    assert AUDIT_JOB in summary['depends_on'] and len(summary['requires']) == 18
    for arm in common.CONTROLS:
        assert 'functional-QA-s202-' + arm in summary['depends_on']
        assert all(f'confirm-seed202-final-{arm}-noise{n}' in summary['depends_on'] for n in range(3))
    assert all('additional121' not in j['name'] for j in jobs)


def test_auditor_template_changes_only_its_stop_and_status_names():
    template = '# invariant tensor checks\n' + ('UTILITY_STRENGTH_FINAL_WEIGHT_AUDITOR_STOP\n' * 2)
    template += 'utility-strength-final-weight-auditor-status.json\nutility-strength-final-weight-auditor-status.tmp\n'
    changed = auditor_source(template)
    reverted = changed.replace('UTILITY_STRENGTH_SEED202_FINAL_WEIGHT_AUDITOR_STOP',
                               'UTILITY_STRENGTH_FINAL_WEIGHT_AUDITOR_STOP')
    reverted = reverted.replace('utility-strength-seed202-final-weight-auditor-status',
                                'utility-strength-final-weight-auditor-status')
    assert reverted == template
    with pytest.raises(ValueError):
        auditor_source('unknown implementation')


@pytest.mark.parametrize('devices', [['0', '0'], ['6', '7'], ['0']])
def test_replica_rejects_duplicate_unassigned_or_incomplete_gpu_groups(tmp_path, devices):
    with pytest.raises(ValueError, match='assigned physical GPUs'):
        confirmation_jobs(tmp_path / NAMESPACE, tmp_path, tmp_path / 'old', devices,
            allowed_gpus=['0', '1', '2', '3'], work_root=tmp_path / 'work',
            audit_driver=tmp_path / 'driver.py', audit_implementation=tmp_path / 'audit.py', audit_sha='a' * 64)


def test_negative_branch_preserves_queue_and_does_not_require_or_create_auditors(tmp_path):
    campaign = tmp_path / 'campaign'
    root, work = campaign / NAMESPACE, tmp_path / 'work'
    root.mkdir(parents=True)
    work.mkdir()
    (root / 'fixed-final-decision-v1.json').write_text(json.dumps(development(False)))
    close_decision(root, campaign)
    (work / 'queue-jobs.json').write_text('[]\n')
    queue(SimpleNamespace(root=root, campaign_root=campaign, work_root=work))
    assert (work / 'queue-jobs.json').read_text() == '[]\n'
    assert not (root / 'seed-confirmation-v1').exists()
    assert json.loads((root / 'seed202-queue-decision-v1.json').read_text())['status'] == 'NO_STRENGTH_CONFIRMATION_QUEUED'


def test_positive_branch_appends_only_bound_replica_jobs_and_keeps_controls(tmp_path):
    campaign, work = tmp_path / 'campaign', tmp_path / 'work'
    root = campaign / NAMESPACE
    (root / 'configs').mkdir(parents=True)
    work.mkdir()
    (root / 'fixed-final-decision-v1.json').write_text(json.dumps(development()))
    close_decision(root, campaign)
    cfg = source(root)
    (root / 'configs' / (NAME + '.json')).write_text(json.dumps(cfg))
    replica = confirmation_recipe(cfg, root)
    for arm in common.CONTROLS:
        path = campaign / 'seed-confirmation-v1/configs' / (arm + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(control(replica, arm)))
    template = work / 'template.py'
    template.write_text('# ' + ('UTILITY_STRENGTH_FINAL_WEIGHT_AUDITOR_STOP\n# ' * 2)
        + 'utility-strength-final-weight-auditor-status.json\n# utility-strength-final-weight-auditor-status.tmp\n')
    driver = work / 'driver.py'
    driver.write_text('# synthetic lifecycle driver; not executed')
    names = [QUEUE_JOB] + [f'confirm-seed202-final-{arm}-noise{n}' for arm in common.CONTROLS for n in range(3)]
    names += ['functional-QA-s202-' + arm for arm in common.CONTROLS]
    existing = [{'name': name} for name in names]
    (work / 'queue-jobs.json').write_text(json.dumps(existing))
    queue(SimpleNamespace(root=root, campaign_root=campaign, work_root=work, previous_work=tmp_path / 'old',
        gpus='0,1', allowed_gpus='0,1,2,3', auditor_template=template,
        auditor_template_sha=hashlib.sha256(template.read_bytes()).hexdigest(), audit_driver=driver,
        audit_driver_sha=hashlib.sha256(driver.read_bytes()).hexdigest()))
    actual = json.loads((work / 'queue-jobs.json').read_text())
    assert actual[:len(existing)] == existing and len(actual) == len(existing) + 7
    assert (work / STOP_NAME).read_text() == 'DEFER_FINAL_AUDIT_FOR_RAM_V1\n'
    assert json.loads((root / 'seed-confirmation-v1/configs' / (NAME + '.json')).read_text())['training_seed'] == 202
    result = json.loads((root / 'seed202-queue-decision-v1.json').read_text())
    assert result['status'] == 'STRENGTH_CONFIRMATION_QUEUED'
    assert not result['model_adopted'] and not result['full_goal_complete']


@pytest.mark.parametrize('fixed_v1_tp,expected', [(39, True), (80, False)])
def test_summary_routes_the_strength_candidate_and_gates_reused_v1_too(tmp_path, monkeypatch, fixed_v1_tp, expected):
    """Synthetic counts prove policy/routing, not actual model performance."""
    campaign = tmp_path / 'campaign'
    root = campaign / NAMESPACE
    folder, controls = root / 'seed-confirmation-v1', campaign / 'seed-confirmation-v1'
    root.mkdir(parents=True)
    value = development()
    value['training_audit'] = {'initial_iou': .4}
    (root / 'fixed-final-decision-v1.json').write_text(json.dumps(value))
    close_decision(root, campaign)
    cfg = confirmation_recipe(source(root), root)
    records = [(NAME, folder, cfg, 60)]
    records += [(arm, controls, control(cfg, arm), 45) for arm in common.CONTROLS]

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def evaluation(tp, noise):
        return {'per_asset': [{'sample_id': str(i), 'source_cluster_id': 's' + str(i),
            'seed': noise + 10, 'target_voxels_sha256': 'b' * 64, 'target_voxel_shape': [64] * 3,
            'tp': tp, 'fp': 10, 'fn': 100 - tp} for i in range(4)]}

    for arm, location, recipe_cfg, tp in records:
        write(location / 'configs' / (arm + '.json'), recipe_cfg)
        qa_root = root if arm == NAME else campaign
        write(qa_root / 'functional-QA-v1' / ('s202-' + arm) / 'summary.json', {
            'saved_frozen_head_exact_to_selected_checkpoint': True,
            'frozen_head_state_keys': list(range(24))})
        for noise in range(3):
            write(location / 'evaluations' / arm / f'noise-{noise}/evaluation.json', evaluation(tp, noise))
            if arm == NAME:
                write(root / 'evaluations' / NAME / f'noise-{noise}/evaluation.json', evaluation(tp, noise))
        audit_root = root if arm == NAME else campaign
        job = 'confirm-seed202-train-' + arm
        folders = [audit_root / 'final-weight-audits-v1' / job]
        folders += [audit_root / 'epoch-audits-v1' / job / f'epoch{n:02d}' for n in range(1, 6)]
        for directory in folders:
            write(directory / 'audit.json', {})
            write(directory / 'file4-closure.json', {})
    v1 = tmp_path / 'fixed-v1'
    for noise in range(3):
        write(v1 / 'evaluations/v1' / f'noise-{noise}/evaluation.json', evaluation(fixed_v1_tp, noise))
    monkeypatch.setattr(common, 'audit_training', lambda *a: {})
    monkeypatch.setattr(common, 'audit_final_replay', lambda *a: True)
    monkeypatch.setattr(common, 'validate_evaluation', lambda *a: None)
    monkeypatch.setattr(common, 'validate_saved_qa', lambda *a: None)
    monkeypatch.setattr(common, 'validate_saved_audits', lambda *a, **k: None)
    monkeypatch.setattr(common, 'task_gradient_audit', lambda *a: {})

    def audit(path, ignored):
        if path.parent.name == 'final-weight-audits-v1':
            return {'actual_optimizer_step': 480, 'frozen_base_tensors_exact': 701,
                    'LoRA_tensors_updated': 360, 'matching_tensors_changed': {'decoder': [], 'score': []}}
        return {'training_seed': 202, 'step': 96 * int(path.name.removeprefix('epoch')),
                'global_batch': 96, 'microbatch_per_rank': 48, 'accumulation': 1, 'world_size': 2,
                'actual_remote_four_numeric_fields_verified': True, 'actual_remote_PNG_byte_exact': True}

    monkeypatch.setattr(common, 'verify_closed_audit', audit)
    common.summarize(root, campaign, candidate_name=NAME, profile_validator=validate_profile,
                     fixed_reference_sources={'v1': (v1, 'v1')})
    result = json.loads((folder / 'final-confirmation-report.json').read_text())
    assert result['candidate'] == NAME
    assert result['positive_CI_and_all3noise_direction_at_both_training_seeds'] is expected
    assert set(result['per_training_seed_contrasts']['202']) == {'v1', *common.CONTROLS}
    assert result['fixed_reference_labels_reused_without_new_training'] == ['v1']
    assert result['models'][NAME]['micro_voxel_iou'] == pytest.approx(60 / 110)
    assert not result['model_adopted'] and not result['reserved121_evaluated']
