"""Synthetic reserved-panel gating and complete16-condition/3-noise replay."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting import frozen_utility_additional_validation as extra


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def contrast():
    return {'gain': .002, 'paired_cluster_bootstrap_ci95': [.001, .003],
            'per_noise': [{'gain': .002}] * 3}


def reports(positive=True):
    primary = {'candidate': extra.NAME, 'model_adopted': False, 'full_goal_complete': False,
               'first_round_negative_preserved': True, 'inputs': [],
               'status': 'PROMISING_FROZEN_UTILITY_REQUIRES_CONFIRMATION',
               'eligible_for_training_seed_confirmation': True,
               'contrasts': {name: contrast() for name in ['v1', *extra.CONTROLS]}}
    repeated = {'status': 'PASS_ACTUAL_FROZEN_UTILITY_SEED202_COMPARISON',
                'models': {name: {} for name in [extra.NAME, *extra.CONTROLS]},
                'model_adopted': False, 'full_goal_complete': False, 'inputs': [],
                'positive_CI_and_all3noise_direction_at_both_training_seeds': positive,
                'per_training_seed_contrasts': {seed: {name: contrast() for name in extra.CONTROLS}
                                                for seed in ['101', '202']}}
    if not positive:
        repeated['per_training_seed_contrasts']['202']['static-top1-c1']['paired_cluster_bootstrap_ci95'][0] = -.001
    return primary, repeated


def closed(path, receipt, canonical):
    save(receipt, {'status': 'PASS', 'local_equals_actual_file4_SHA256': True,
                   'files': [{'relative_path': str(path.relative_to(canonical)),
                              'sha256': extra.sha(path), 'bytes': path.stat().st_size}]})


@pytest.mark.parametrize('positive', [False, True])
def test_replication_gate_requires_both_controls_at_both_training_seeds(positive):
    _, repeated = reports(positive)
    assert extra.replication_supported(repeated) is positive
    wrong = copy.deepcopy(repeated)
    wrong['positive_CI_and_all3noise_direction_at_both_training_seeds'] = not positive
    with pytest.raises(ValueError, match='flag disagrees'):
        extra.replication_supported(wrong)


def test_missing_same_retargeter_seed_control_cannot_open_the_reserved_panel():
    _, repeated = reports()
    repeated['per_training_seed_contrasts']['202'].pop('static-top1-c1')
    with pytest.raises(ValueError, match='both Top1 controls'):
        extra.replication_supported(repeated)


def test_failed_replication_never_reads_reserved_rows_or_appends_gpu_jobs(tmp_path, monkeypatch):
    campaign, work = tmp_path / 'campaign', tmp_path / 'work'
    root = campaign / 'F'
    work.mkdir()
    primary, repeated = reports(False)
    save(root / 'fixed-final-decision-v1.json', primary)
    closed(root / 'fixed-final-decision-v1.json', root / 'fixed-final-report-file4.json', tmp_path)
    save(root / 'seed-confirmation-v1/final-confirmation-report.json', repeated)
    before = b'[]\n'
    (work / 'queue-jobs.json').write_bytes(before)
    monkeypatch.setattr(extra, 'close_confirmation', lambda args: None)
    def forbidden(*args):
        raise AssertionError('reserved IDs/outcomes must remain unopened after failed replication')
    monkeypatch.setattr(extra, 'reservation_inputs', forbidden)
    extra.queue(SimpleNamespace(root=root, campaign_root=campaign, work_root=work, previous_work=tmp_path / 'previous'))
    decision = json.loads((root / 'additional-validation-v1/queue-decision.json').read_text())
    assert decision['status'] == 'NO_RESERVED_EVALUATION_REPLICATION_FAILED'
    assert not decision['model_adopted'] and not decision['full_goal_complete']
    assert (work / 'queue-jobs.json').read_bytes() == before
    assert not (root / 'additional-validation-v1/evaluation-contract.json').exists()


def population(tmp_path):
    campaign, root, work = tmp_path / 'campaign', tmp_path / 'campaign/F', tmp_path / 'work'
    work.mkdir()
    ids = [f'extra-{i}' for i in range(121)]
    rows = [{'sample_id': sid, 'source_cluster_id': f'extra-source-{i % 39}'} for i, sid in enumerate(ids)]
    development = [{'sample_id': f'dev-{i}', 'source_cluster_id': f'dev-source-{i % 93}'} for i in range(128)]
    rows += development + [{'sample_id': f'unused-{i}', 'source_cluster_id': f'dev-source-{i % 93}'} for i in range(259)]
    val, train = tmp_path / 'validation.jsonl', tmp_path / 'train.jsonl'
    val.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    train.write_text(json.dumps({'sample_id': 'train', 'source_cluster_id': 'train-source'}) + '\n')
    save(campaign / 'configs/top1-reference.json', {'data_dir': str(train), 'validation': {'roots_file': str(val)}})
    save(campaign / 'configs/zero-shot-reference.json', {'trainer': {'args': {'continuation_ckpt': {'denoiser': 'SS4210'}}}})
    save(campaign / 'evaluations/top1-reference/noise-0/evaluation.json', {'per_asset': development})
    folder = campaign / 'additional-validation-v1'
    panel = folder / 'panel-source-disjoint-121.json'
    save(panel, {'sample_ids': ids})
    reservation = {'status': 'RESERVED_NOT_EVALUATED', 'additional_assets': 121, 'additional_source_clusters': 39,
                   'panel_SHA256': extra.sha(panel), 'manifest_SHA256': {'train': extra.sha(train), 'validation': extra.sha(val)},
                   'seed_maps': {}}
    for noise in range(3):
        path = tmp_path / f'noise-{noise}.json'
        save(path, {sid: i + 1000 * noise for i, sid in enumerate(ids)})
        reservation['seed_maps'][str(noise)] = {'path': str(path), 'sha256': extra.sha(path)}
    save(folder / 'panel-contract.json', reservation)
    primary, repeated = reports()
    save(root / 'fixed-final-decision-v1.json', primary)
    closed(root / 'fixed-final-decision-v1.json', root / 'fixed-final-report-file4.json', tmp_path)
    save(root / 'seed-confirmation-v1/final-confirmation-report.json', repeated)
    closed(root / 'seed-confirmation-v1/final-confirmation-report.json',
           root / 'seed-confirmation-v1/final-confirmation-report-file4.json', tmp_path)
    return root, campaign, work


def test_all_sixteen_saved_conditions_wait_for_replication_without_new_training(tmp_path):
    root, campaign, work = population(tmp_path)
    reservation, panel, *_ = extra.reservation_inputs(campaign)
    jobs, contract = extra.jobs_and_contract(root, campaign, reservation, panel)
    assert len(contract['models']) == 16 and len(jobs) == 49
    assert {'s101-frozen-utility-c1', 's202-frozen-utility-c1', 's202-static-top1-c1',
            's101-task-no-decoder', 's101-task-no-prior', 's101-task-native-spatial',
            'previous-v1', 'previous-c1', 'zero-source-only', 'zero-edited-rgb'} <= contract['models'].keys()
    assert all(job['resource'] == 'evaluation' and job['depends_on'] == [extra.QUEUE_JOB]
               and 'gpu' not in job and 'gpus' not in job for job in jobs[:-1])
    assert all('--generation-batch' in job['command'] for job in jobs[:-1])
    assert contract['models']['s101-frozen-utility-c1']['config'] == str(root / 'configs/frozen-utility-c1.json')
    assert contract['models']['previous-c1']['config'].endswith('ss-seed-confirmation-v1/configs/c1.json')
    assert extra.factory_job(root, campaign, work, tmp_path)['depends_on'] == [extra.SUMMARY_JOB]


@pytest.mark.parametrize('corrupt_target', [False, True])
def test_complete_synthetic16_condition_summary_and_changed_GT_rejection(tmp_path, monkeypatch, corrupt_target):
    root, campaign, work = population(tmp_path)
    monkeypatch.setattr(extra, 'close_confirmation', lambda args: None)
    save(work / 'queue-jobs.json', [{'name': extra.QUEUE_JOB}])
    extra.queue(SimpleNamespace(root=root, campaign_root=campaign, work_root=work, previous_work=tmp_path / 'previous'))
    scheduled = json.loads((work / 'queue-jobs.json').read_text())
    assert len(scheduled) == 51  # Existing factory plus48evaluation,summary,actualfile4closure.
    closure = scheduled[-1]
    assert closure['name'] == 'close-frozen-utility-additional121-file4'
    assert closure['resource'] == 'cpu' and len(closure['requires']) == 100
    contract = json.loads((root / 'additional-validation-v1/evaluation-contract.json').read_text())
    reservation, panel, ids, rows, maps, _ = extra.reservation_inputs(campaign)
    for label, spec in contract['models'].items():
        tp = 60 if label in ['s101-' + extra.NAME, 's202-' + extra.NAME] else 50
        for noise in range(3):
            result = {'status': 'PASS', 'stage': 'ss64', 'step': spec['step'], 'weights': 'raw',
                      'sampling_steps': 12, 'guidance': 3., 'generation_batch_size': 8,
                      'optimizer_updates_during_evaluation': 0, 'conditioning_mode': spec['conditioning_mode'],
                      'target_modality': 'rgb', 'diagnostic_only': False, 'checkpoint': spec['checkpoint'],
                      'manifest_sha256': reservation['manifest_SHA256']['validation'],
                      'panel_file': str(panel), 'seed_map_file': reservation['seed_maps'][str(noise)]['path'],
                      'conditioner_checkpoint': spec['conditioner_checkpoint'], 'arm_config': spec['config'],
                      'input_camera_ids': [8, 9, 10, 11], 'input_states': ['M0'] * 4 if spec['conditioning_mode'] == 'source-only'
                      else ['M1', 'M0', 'M0', 'M0'], 'quick_panel_ids': ids,
                      'per_asset': [{'sample_id': sid, 'source_cluster_id': rows[sid]['source_cluster_id'],
                                     'seed': maps[noise][sid], 'tp': tp, 'fp': 10, 'fn': 100 - tp,
                                     'target_voxels_sha256': hashlib.sha256(sid.encode()).hexdigest(),
                                     'target_voxel_shape': [64, 64, 64]} for sid in ids]}
            if corrupt_target and label == 's202-' + extra.NAME and noise == 2:
                result['per_asset'][3]['target_voxels_sha256'] = 'incorrect-target'
            save(root / 'additional-validation-v1/evaluations' / label / f'noise-{noise}/evaluation.json', result)
    if corrupt_target:
        with pytest.raises(ValueError, match='identity'):
            extra.summarize(root, campaign)
        return
    extra.summarize(root, campaign)
    result = json.loads((root / 'additional-validation-v1/final-report.json').read_text())
    assert len(result['models']) == 16
    assert result['positive_CI_and_all_noises_at_both_seeds_vs_both_Top1_controls']
    assert result['positive_vs_fixed_V1_at_both_candidate_training_seeds']
    assert result['passes_locked_candidate_quality_controls']
    assert result['per_training_seed_both_Top1_controls']['202']['static-top1-c1']['source_clusters'] == 39
    assert result['per_training_seed_both_Top1_controls']['101']['top1-reference']['paired_target_hashes_verified'] == 363
    assert result['candidate_reselected'] is False and result['model_adopted'] is False
    assert result['full_goal_complete'] is False
