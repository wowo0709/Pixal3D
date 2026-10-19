"""Replica routing preserves numerical recipes and rejects unsupported gains."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting import orthogonal_utility_confirmation as module
from experiments.deformnext_retargeting.frozen_utility_strength import recipe as strength_recipe


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup(tmp_path):
    campaign = tmp_path / 'synthetic-campaign'
    strength = campaign / 'utility-strength-followup-v1'
    source = strength_recipe({'ckpt': 'none', 'load_dir': '', 'num_gpus': 2, 'training_seed': 101,
        'default_output_dir': str(strength / 'runs_ss64/frozen-utility-c1'),
        'experiment': {'arm': 'frozen-utility-c1', 'required_microbatch_per_gpu': 48},
        'trainer': {'name': 'ImageConditionedProjFlowMatchingCFGTrainer', 'args': {
            'max_steps': 480, 'batch_size_per_gpu': 48, 'batch_split': 1,
            'deform_conditioner': {'implementation': 'r9-native-v5', 'arm': 'c1', 'score_temperature': .005,
                'task_attention': {'decoder': True, 'score_prior': True, 'train_decoder': False,
                    'train_score': False, 'content_mode': 'residual-zero', 'chunk_size': 64}}}}}, strength)
    for profile in module.study.PROFILES.values():
        root = campaign / profile.NAMESPACE
        save(root / 'configs' / (profile.NAME + '.json'), profile.recipe(source, root))
    folder = campaign / module.study.NAMESPACE
    runtime = folder / 'fixture.py'; archived = folder / 'archived.py'
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text('# Nonexecuted source-closure fixture\n'); archived.write_bytes(runtime.read_bytes())
    contract = folder / 'source-contract.json'; receipt = folder / 'source-file4.json'
    save(contract, {'runtime_components': [{'runtime_path': str(runtime), 'archived_path': str(archived),
                                          'sha256': module.sha(runtime)}]})
    save(receipt, {'status': 'PASS', 'local_equals_actual_file4_SHA256': True, 'files': [
        {'relative_path': str(path.relative_to(campaign.parent)), 'sha256': module.sha(path),
         'bytes': path.stat().st_size} for path in [contract, archived]]})
    epoch = tmp_path / 'epoch.py'; final = tmp_path / 'final.py'
    epoch.write_text('def epoch(): pass\nwhile True: pass\n')
    final.write_text('def audit(): pass\nwhile True: pass\n')
    work = tmp_path / 'work'
    save(work / 'queue-jobs.json', [{'name': name} for name in [module.FACTORY, module.DECISION,
        *['close-fixed-final-' + name for name in module.study.PROFILES],
        *[f'confirm-seed202-final-{arm}-noise{noise}' for arm in module.common.CONTROLS for noise in range(3)],
        *['functional-QA-s202-' + arm for arm in module.common.CONTROLS]]])
    return SimpleNamespace(campaign_root=campaign, work_root=work, previous_work=tmp_path / 'previous',
        source_contract=contract, source_receipt=receipt, epoch_template=epoch, final_template=final,
        epoch_sha=module.sha(epoch), final_sha=module.sha(final), allowed_gpus='0,1,2,3',
        voxel_gpus='0,1', soft_gpus='2,3')


@pytest.mark.parametrize('name', list(module.study.PROFILES))
def test_replica_preserves_actual_profile_and_rejects_numerical_changes(tmp_path, name):
    args = setup(tmp_path); profile = module.study.PROFILES[name]
    root = args.campaign_root / profile.NAMESPACE
    source = module.read(root / 'configs' / (name + '.json')); before = copy.deepcopy(source)
    cfg = module.replica_recipe(source, root, profile)
    module.validate_replica(cfg, root / 'seed-confirmation-v1', profile)
    assert source == before and cfg['trainer'] == source['trainer']
    assert cfg['training_seed'] == cfg['experiment']['training_seed'] == 202
    assert cfg['master_port'] == module.PORTS[name]
    launched = copy.deepcopy(cfg)
    launched['experiment'].update(release_status='native-and-memory-verified', microbatch_per_gpu=48)
    module.validate_replica(launched, root / 'seed-confirmation-v1', profile)
    changed = copy.deepcopy(cfg); changed['trainer']['args']['batch_size_per_gpu'] = 24
    with pytest.raises(ValueError, match='replica changed'): module.validate_replica(
        changed, root / 'seed-confirmation-v1', profile)


def test_source_guard_allows_only_launcher_status_changes_in_prepared_recipes(tmp_path):
    args = setup(tmp_path)
    bindings = []
    receipt = module.read(args.source_receipt)
    for profile in module.study.PROFILES.values():
        runtime = args.campaign_root / profile.NAMESPACE / 'configs' / (profile.NAME + '.json')
        archived = args.source_contract.parent / (profile.NAME + '-original.json')
        archived.write_bytes(runtime.read_bytes())
        bindings.append({'runtime_path': str(runtime), 'archived_path': str(archived)})
        receipt['files'].append({'relative_path': str(archived.relative_to(args.campaign_root.parent)),
            'sha256': module.sha(archived), 'bytes': archived.stat().st_size})
        cfg = module.read(runtime)
        cfg['experiment'].update(release_status='native-and-memory-verified', microbatch_per_gpu=48)
        save(runtime, cfg)
    contract = module.read(args.source_contract); contract['prepared_recipe_bindings'] = bindings
    save(args.source_contract, contract)
    for row in receipt['files']:
        if row['relative_path'] == str(args.source_contract.relative_to(args.campaign_root.parent)):
            row.update(sha256=module.sha(args.source_contract), bytes=args.source_contract.stat().st_size)
    save(args.source_receipt, receipt)
    module.require_sources(args)
    path = Path(bindings[0]['runtime_path']); cfg = module.read(path)
    cfg['trainer']['args']['max_steps'] = 960; save(path, cfg)
    with pytest.raises(ValueError, match='numerical recipe changed'): module.require_sources(args)


def test_replica_jobs_close_every_epoch_final_and_both_control_comparisons(tmp_path):
    args = setup(tmp_path)
    for profile, devices in zip(module.study.PROFILES.values(), [['0', '1'], ['2', '3']]):
        jobs = module.replica_jobs(args, profile, devices)
        assert len(jobs) == 13
        assert jobs[0]['gpus'] == devices and module.MODULE in jobs[0]['command']
        assert jobs[0]['command'][jobs[0]['command'].index('--candidate') + 1] == profile.NAME
        assert '--source-contract' in jobs[0]['command'] and '--source-receipt' in jobs[0]['command']
        assert jobs[0]['depends_on'] == [module.DECISION]
        names = {job['name'] for job in jobs + module.read(args.work_root / 'queue-jobs.json')}
        assert all(dep in names for job in jobs for dep in job.get('depends_on', []))
        epochs = [job for job in jobs if job['name'].startswith('audit-epoch-')]
        assert len(epochs) == 5 and all('depends_on' not in job for job in epochs)
        assert all('run_saved_audit_once' in ' '.join(job['command']) for job in epochs)
        assert len([job for job in jobs if job['resource'] == 'exclusive']) == 1
        assert jobs[-1]['outputs'][0].endswith('final-confirmation-report-file4.json')


def test_strength_positive_skips_alternative_confirmation_without_dangling_dependencies(tmp_path):
    args = setup(tmp_path)
    save(args.campaign_root / module.study.NAMESPACE / 'queue-decision-v1.json', {
        'status': 'SKIPPED_FOR_STRENGTH_CONFIRMATION'})
    before = (args.work_root / 'queue-jobs.json').read_bytes()
    module.stage(args)
    assert (args.work_root / 'queue-jobs.json').read_bytes() == before
    decision = module.read(args.campaign_root / module.study.NAMESPACE / 'confirmation-stage-v1.json')
    assert decision['queued_jobs'] == 0 and not decision['full_goal_complete']


def test_negative_candidates_do_not_create_replica_configs_or_jobs(tmp_path, monkeypatch):
    args = setup(tmp_path)
    for profile in module.study.PROFILES.values():
        root = args.campaign_root / profile.NAMESPACE
        save(root / 'fixed-final-decision-v1.json', {'synthetic': True})
    monkeypatch.setattr(module, 'verify_development', lambda *unused: ({}, False))
    before = (args.work_root / 'queue-jobs.json').read_bytes()
    module.queue(args)
    assert (args.work_root / 'queue-jobs.json').read_bytes() == before
    assert not any((args.campaign_root / p.NAMESPACE / 'seed-confirmation-v1').exists()
                   for p in module.study.PROFILES.values())
    decision = module.read(args.campaign_root / module.study.NAMESPACE / 'seed202-queue-decision-v1.json')
    assert decision['status'] == 'NO_ORTHOGONAL_CANDIDATE_SUPPORTED'


def test_stage_rejects_changed_runtime_before_queue_mutation(tmp_path):
    args = setup(tmp_path)
    Path(module.read(args.source_contract)['runtime_components'][0]['runtime_path']).write_text('# Changed\n')
    before = (args.work_root / 'queue-jobs.json').read_bytes()
    with pytest.raises(ValueError, match='executed'): module.stage(args)
    assert (args.work_root / 'queue-jobs.json').read_bytes() == before


def test_two_supported_profiles_queue_exactly_two_fresh_replicas(tmp_path, monkeypatch):
    args = setup(tmp_path)
    for profile in module.study.PROFILES.values():
        root = args.campaign_root / profile.NAMESPACE
        save(root / 'fixed-final-decision-v1.json', {'synthetic': True})
    for arm in module.common.CONTROLS:
        save(args.campaign_root / 'seed-confirmation-v1/configs' / (arm + '.json'), {'synthetic': True})
    # Only routing is synthetic here; real recipe/control compatibility is also
    # checked separately against the immutable actual resolved configurations.
    monkeypatch.setattr(module, 'verify_development', lambda *unused: ({}, True))
    monkeypatch.setattr(module, 'validate_control', lambda *unused: None)
    before = module.read(args.work_root / 'queue-jobs.json')
    module.queue(args)
    jobs = module.read(args.work_root / 'queue-jobs.json')[len(before):]
    assert len(jobs) == 26
    assert [job['gpus'] for job in jobs if job['resource'] == 'exclusive'] == [['0', '1'], ['2', '3']]
    for profile in module.study.PROFILES.values():
        root = args.campaign_root / profile.NAMESPACE
        cfg = module.read(root / 'seed-confirmation-v1/configs' / (profile.NAME + '.json'))
        module.validate_replica(cfg, root / 'seed-confirmation-v1', profile)
    decision = module.read(args.campaign_root / module.study.NAMESPACE / 'seed202-queue-decision-v1.json')
    assert decision['queued_jobs'] == 26 and not decision['model_adopted']


def test_actual_closed_development_ci_crossing_zero_is_ineligible(tmp_path):
    args = setup(tmp_path); profile = next(iter(module.study.PROFILES.values()))
    root = args.campaign_root / profile.NAMESPACE
    good = {'gain': .002, 'paired_cluster_bootstrap_ci95': [.001, .003], 'per_noise': [{'gain': .002}] * 3}
    contrasts = {arm: copy.deepcopy(good) for arm in ['v1', *module.common.CONTROLS]}
    contrasts['static-top1-c1']['paired_cluster_bootstrap_ci95'][0] = -.0001
    path = root / 'fixed-final-decision-v1.json'
    save(path, {'candidate': profile.NAME, 'contrasts': contrasts, 'model_adopted': False,
        'full_goal_complete': False, 'first_round_negative_preserved': True, 'inputs': [],
        'status': 'NO_FROZEN_UTILITY_IMPROVEMENT_CONFIRMED', 'eligible_for_training_seed_confirmation': False})
    save(root / 'fixed-final-report-file4.json', {'status': 'PASS', 'local_equals_actual_file4_SHA256': True,
        'files': [{'relative_path': str(path.relative_to(args.campaign_root.parent)), 'sha256': module.sha(path),
                   'bytes': path.stat().st_size}]})
    assert module.verify_development(root, args.campaign_root, profile)[1] is False


@pytest.mark.parametrize('devices', ['6,7', '1,2', '0'])
def test_unassigned_or_overlapping_replica_devices_rejected(tmp_path, devices):
    args = setup(tmp_path); args.soft_gpus = devices
    with pytest.raises(ValueError, match='physical GPU pairs'): module.queue(args)
