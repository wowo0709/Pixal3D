"""Synthetic routing/dependency checks; never train or report a real model."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting import orthogonal_utility_followups as module
from experiments.deformnext_retargeting import utility_softprior_followup as soft
from experiments.deformnext_retargeting import voxel_query_followup as voxel
from experiments.deformnext_retargeting.frozen_utility_strength import recipe as strength_recipe


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def receipt(path, paths, campaign):
    save(path, {'status': 'PASS', 'local_equals_actual_file4_SHA256': True,
        'files': [{'relative_path': str(p.relative_to(campaign.parent)),
                   'sha256': module.sha(p), 'bytes': p.stat().st_size} for p in paths]})


def setup(tmp_path, positive):
    campaign = tmp_path / 'synthetic-campaign'; strength = campaign / 'utility-strength-followup-v1'
    contrast = {'gain': .002, 'paired_cluster_bootstrap_ci95': [.001, .003],
                'per_noise': [{'gain': .002}] * 3}
    contrasts = {name: copy.deepcopy(contrast) for name in ['v1', 'static-top1-c1', 'top1-reference']}
    if not positive: contrasts['static-top1-c1']['paired_cluster_bootstrap_ci95'][0] = -.001
    decision = strength / 'fixed-final-decision-v1.json'
    save(decision, {'candidate': module.STRENGTH_NAME, 'model_adopted': False, 'full_goal_complete': False,
        'first_round_negative_preserved': True, 'contrasts': contrasts, 'inputs': [],
        'status': 'PROMISING_FROZEN_UTILITY_REQUIRES_CONFIRMATION' if positive else 'NO_FROZEN_UTILITY_IMPROVEMENT_CONFIRMED',
        'eligible_for_training_seed_confirmation': positive})
    receipt(strength / 'fixed-final-report-file4.json', [decision], campaign)
    save(strength / 'seed202-queue-decision-v1.json', {'candidate': module.STRENGTH_NAME,
        'status': 'STRENGTH_CONFIRMATION_QUEUED' if positive else 'NO_STRENGTH_CONFIRMATION_QUEUED',
        'eligible': positive, 'development_decision_SHA256': module.sha(decision)})
    work = tmp_path / 'work'; previous = tmp_path / 'previous'
    save(work / 'queue-jobs.json', [{'name': name} for name in [module.FACTORY, 'native-proof-voxel-query-c1']])
    save(previous / 'panels/panel-32.json', {'synthetic': True})
    save(previous / 'panels/seed-map.json', {'synthetic': True})
    source = strength_recipe({'ckpt': 'none', 'load_dir': '', 'num_gpus': 2, 'training_seed': 101,
        'default_output_dir': str(strength / 'runs_ss64/frozen-utility-c1'),
        'experiment': {'arm': 'frozen-utility-c1', 'required_microbatch_per_gpu': 48},
        'trainer': {'name': 'ImageConditionedProjFlowMatchingCFGTrainer', 'args': {
            'max_steps': 480, 'batch_size_per_gpu': 48, 'batch_split': 1,
            'deform_conditioner': {'implementation': 'r9-native-v5', 'arm': 'c1', 'score_temperature': .005,
                'task_attention': {'decoder': True, 'score_prior': True, 'train_decoder': False,
                    'train_score': False, 'content_mode': 'residual-zero', 'chunk_size': 64}}}}}, strength)
    for profile in [voxel, soft]:
        root = campaign / profile.NAMESPACE
        save(root / 'configs' / (profile.NAME + '.json'), profile.recipe(source, root))
    folder = campaign / module.NAMESPACE
    folder.mkdir(parents=True)
    runtime = folder / 'synthetic-source.py'; runtime.write_text('# Synthetic nonexecuted fixture\n')
    archived = folder / 'archived-synthetic-source.py'; archived.write_bytes(runtime.read_bytes())
    contract = folder / 'source-contract.json'
    save(contract, {'runtime_components': [{'runtime_path': str(runtime), 'archived_path': str(archived),
                                          'sha256': module.sha(runtime)}]})
    source_receipt = folder / 'source-file4.json'
    receipt(source_receipt, [contract, archived], campaign)
    epoch = tmp_path / 'epoch-template.py'; epoch.write_text('def epoch(): pass\nwhile True: pass\n')
    final = tmp_path / 'final-template.py'; final.write_text('def audit(): pass\nwhile True: pass\n')
    return SimpleNamespace(campaign_root=campaign, work_root=work, previous_work=previous,
        source_contract=contract, source_receipt=source_receipt, epoch_template=epoch,
        final_template=final, epoch_sha=module.sha(epoch), final_sha=module.sha(final),
        allowed_gpus='0,1,2,3', voxel_gpus='0,1', soft_gpus='2,3')


def test_positive_strength_outcome_does_not_schedule_alternative_training(tmp_path):
    args = setup(tmp_path, True)
    before = (args.work_root / 'queue-jobs.json').read_bytes()
    module.queue(args)
    result = module.read(args.campaign_root / module.NAMESPACE / 'queue-decision-v1.json')
    assert result['status'] == 'SKIPPED_FOR_STRENGTH_CONFIRMATION'
    assert (args.work_root / 'queue-jobs.json').read_bytes() == before
    assert not result['reserved121_accessed']


def test_negative_strength_queues_two_complete_independent_protocols(tmp_path, monkeypatch):
    args = setup(tmp_path, False)
    # The CLI executes with __name__='__main__'; future commands must retain
    # the actual module entrypoint rather than python -m __main__.
    monkeypatch.setattr(module, '__name__', '__main__')
    module.queue(args)
    jobs = module.read(args.work_root / 'queue-jobs.json')
    assert len(jobs) == 36  # Two existing fixtures plus17 jobs per model.
    names = {job['name'] for job in jobs}
    assert all(dep in names for job in jobs for dep in job.get('depends_on', []))
    mains = [job for job in jobs if job.get('resource') == 'exclusive']
    assert [job['gpus'] for job in mains] == [['0', '1'], ['2', '3']]
    assert voxel.__name__ in mains[0]['command'] and soft.__name__ in mains[1]['command']
    assert 'native-proof-voxel-query-c1' in mains[0]['depends_on']
    assert mains[1]['depends_on'] == [module.FACTORY]
    for profile in [voxel, soft]:
        epochs = [job for job in jobs if job['name'].startswith('audit-epoch-' + profile.NAME)]
        assert len(epochs) == 5 and all('depends_on' not in job for job in epochs)
        assert all('run_saved_audit_once' in ' '.join(job['command']) for job in epochs)
        summary = next(job for job in jobs if job['name'] == 'summarize-fixed-final-' + profile.NAME)
        assert module.MODULE in summary['command'] and '__main__' not in summary['command']
        assert len(summary['depends_on']) == 10
        assert all(f'final-{profile.NAME}-noise{noise}' in summary['depends_on'] for noise in range(3))
        assert 'audit-final-' + profile.NAME in summary['depends_on']
    result = module.read(args.campaign_root / module.NAMESPACE / 'queue-decision-v1.json')
    assert result['queued_jobs'] == 34 and result['new_training_runs'] == 2
    assert not result['model_adopted'] and not result['reserved121_accessed']


@pytest.mark.parametrize('pairs', [('0,1', '1,2'), ('0,1', '6,7'), ('0', '2,3')])
def test_gpu_pairs_must_be_disjoint_and_assigned(tmp_path, pairs):
    args = setup(tmp_path, False); args.voxel_gpus, args.soft_gpus = pairs
    with pytest.raises(ValueError, match='physical GPU pairs'): module.queue(args)


def test_closed_source_contract_detects_changed_runtime_bytes(tmp_path):
    args = setup(tmp_path, False)
    module.require_source_contract(args.source_contract, args.source_receipt, args.campaign_root)
    entry = module.read(args.source_contract)['runtime_components'][0]
    from pathlib import Path
    Path(entry['runtime_path']).write_text('# Changed synthetic nonexecuted source\n')
    with pytest.raises(ValueError, match='executed'): module.require_source_contract(
        args.source_contract, args.source_receipt, args.campaign_root)
