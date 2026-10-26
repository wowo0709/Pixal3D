from copy import deepcopy
from hashlib import sha256
import json
import os

import psutil
import pytest

from experiments.deformnext_retargeting.dev_cascade_cache_retirement import ARMS, own_contract
from experiments.deformnext_retargeting.operational_dev_geometry_retirement import (
    actual_queue_jobs, pending_reads, prior_closed_retirements, reader_snapshot, validate_geometry_candidate,
)


def completed(tmp_path, arm='A_FIXED_UNIFORM'):
    sid = 'actual-fixture-DEV'
    owned = tmp_path / 'owned'
    root = owned / arm / 'cases' / sha256(sid.encode()).hexdigest()[:16]
    path = root / 'geometry/prediction.npz'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'closed geometry raw')
    case = dict(SS_arm=arm, sample_id=sid, status=own_contract(arm).CASE_STATUS,
        new_training=False, TEST_forward=False, same_original_material_M1_reference=True,
        all_original_numerical_source_files_unmodified=True, reused_completed_native_generation=None,
        new_Shape_cascade_calls=1, new_PBR_replay_calls=1, geometry_root=str(root / 'geometry'),
        final_generation_root=str(root / 'PBR'), shared_root=str(root / 'shared'),
        SS_prediction_SHA256='a' * 64, actual_owned_terminals=[
            dict(phase=p, returncode=0, actual_session_drained=True)
            for p in ['geometry', 'PBR', 'render', 'shared']])
    geometry = dict(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION',
        sample_id=sid, SS_prediction_SHA256='a' * 64, GT_geometry_or_coords_used_in_generation=False,
        GT_loaded_only_after_generation=True, per_output_alignment=False)
    replay = dict(status='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE', sample_id=sid,
        PBR_arm='top1-only', checkpoint_kind='trained-native1900', engineering_only=False,
        source_weights_unchanged=True, geometry_array_byte_identity=dict(vertices=True, faces=True))
    evidence = dict(status='PASS_ACTUAL_COMPLETE_SS7_GEOMETRY12_FULL_FAMILY48_PAPER_EXPORT',
        assets=128, source_clusters=93, SS_rows=[dict(arm=a) for a in [*ARMS, 'E_FIXED_RAW', 'C_DECODER']],
        all_previous_six_means_exact=True, actual_new_NN_forward_or_quality_computation=0,
        all48_intervals=dict(family_comparisons=48))
    digest = sha256(path.read_bytes()).hexdigest()
    return path, dict(case_root=root, owned_root=owned, case=case, geometry=geometry,
        replay=replay, canonical_entry=dict(sha256=digest, bytes=path.stat().st_size),
        actual_local_SHA256=digest, readers=[], late_evidence=evidence)


@pytest.mark.parametrize('arm', ARMS)
def test_only_completed_late_geometry_is_eligible_without_unlink(tmp_path, arm):
    path, kwargs = completed(tmp_path, arm)
    assert validate_geometry_candidate(path, **kwargs)['canonical_file4_retained'] is True
    assert path.read_bytes() == b'closed geometry raw'


@pytest.mark.parametrize('problem', ['reader', 'late-pending', 'late-short', 'late-family',
    'case-failed', 'undrained', 'TEST', 'GT', 'changed-model-geometry', 'remote-sha', 'remote-size',
    'different-owner', 'symlink'])
def test_incomplete_or_unsafe_raw_stays_present(tmp_path, problem):
    path, kwargs = completed(tmp_path)
    kwargs = deepcopy(kwargs)
    if problem == 'reader': kwargs['readers'] = ['actual open process']
    elif problem == 'late-pending': kwargs['late_evidence']['status'] = 'pending'
    elif problem == 'late-short': kwargs['late_evidence']['assets'] = 127
    elif problem == 'late-family': kwargs['late_evidence']['all48_intervals']['family_comparisons'] = 8
    elif problem == 'case-failed': kwargs['case']['actual_owned_terminals'][2]['returncode'] = 1
    elif problem == 'undrained': kwargs['case']['actual_owned_terminals'][1]['actual_session_drained'] = False
    elif problem == 'TEST': kwargs['case']['TEST_forward'] = True
    elif problem == 'GT': kwargs['geometry']['GT_geometry_or_coords_used_in_generation'] = True
    elif problem == 'changed-model-geometry': kwargs['replay']['geometry_array_byte_identity']['vertices'] = False
    elif problem == 'remote-sha': kwargs['canonical_entry']['sha256'] = 'b' * 64
    elif problem == 'remote-size': kwargs['canonical_entry']['bytes'] += 1
    elif problem == 'different-owner': kwargs['owned_root'] = tmp_path / 'other'
    else:
        target = tmp_path / 'canonical-original.npz'
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(ValueError): validate_geometry_candidate(path, **kwargs)
    assert path.exists()


@pytest.mark.parametrize('relative', ['geometry/native-shape-guides.pt', 'PBR/prediction.npz',
    'shared/camera08.npz', 'render/camera10.npz', 'ckpts/model.pt'])
def test_other_retained_artifacts_cannot_be_selected(tmp_path, relative):
    _, kwargs = completed(tmp_path)
    path = kwargs['case_root'] / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'closed geometry raw')
    with pytest.raises(ValueError): validate_geometry_candidate(path, **kwargs)
    assert path.exists()


@pytest.mark.parametrize('status', ['pending', 'running', 'HELD', 'unknown-future'])
def test_unfinished_and_held_plan_reads_are_protected(tmp_path, status):
    folder = tmp_path / 'geometry'
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(dict(closed_inputs={str(folder / 'prediction.npz'): 'a' * 64})))
    job = dict(name='reader', command=['python', '--plan', str(plan)])
    assert pending_reads([job], {'reader': dict(status=status)}, [folder]) == ['reader']


def test_missing_pending_plan_is_not_absence_of_a_reader(tmp_path):
    job = dict(name='reader', command=['python', '--plan', str(tmp_path / 'missing.json')])
    with pytest.raises(ValueError): pending_reads([job], {}, [tmp_path / 'geometry'])
    assert pending_reads([job], {'reader': dict(status='completed')}, [tmp_path / 'geometry']) == []


def test_second_snapshot_sees_new_actual_reader_and_new_queue_plan(tmp_path, monkeypatch):
    folder = tmp_path / 'geometry'
    folder.mkdir()
    raw = folder / 'prediction.npz'
    raw.write_bytes(b'actual raw')
    jobs, states = tmp_path / 'jobs.json', tmp_path / 'states.json'
    jobs.write_text('[]')
    states.write_text('{}')
    policy = dict(reader_queues=[dict(jobs=str(jobs), states=str(states))])
    process = psutil.Process(os.getpid())
    monkeypatch.setattr(psutil, 'process_iter', lambda: [process])
    first, first_queue = reader_snapshot(policy, [folder])
    assert not first
    with raw.open('rb'):
        second, _ = reader_snapshot(policy, [folder])
        assert second == ['actual open process ' + str(os.getpid())]
    plan = tmp_path / 'new-plan.json'
    plan.write_text(json.dumps(dict(generation=str(folder))))
    jobs.write_text(json.dumps([dict(name='new-reader', command=['python', '--plan', str(plan)])]))
    second, second_queue = reader_snapshot(policy, [folder])
    assert second == ['new-reader'] and first_queue != second_queue
    assert raw.exists()


def queues(tmp_path, job, main_state, aux_job=None, aux_state=None):
    rows = []
    for label, jobs, states in [('main', [job], {job['name']: main_state}),
        ('aux', [aux_job] if aux_job else [], {job['name']: aux_state or dict(status='HELD')})]:
        jp, sp = tmp_path / (label + '-jobs.json'), tmp_path / (label + '-states.json')
        jp.write_text(json.dumps(jobs)); sp.write_text(json.dumps(states))
        rows.append(dict(jobs=str(jp), states=str(sp)))
    return dict(reader_queues=rows)


def test_migrated_held_history_cannot_override_actual_completed_owner(tmp_path):
    folder = tmp_path / 'geometry'
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(dict(generation=str(folder))))
    job = dict(name='migrated-reader', command=['python', '--plan', str(plan)])
    policy = queues(tmp_path, job, dict(status='completed', returncode=0))
    jobs, states, _ = actual_queue_jobs(policy)
    assert jobs == [job] and states == {job['name']: dict(status='completed', returncode=0)}
    assert pending_reads(jobs, states, [folder]) == []


def test_actual_held_owner_cannot_borrow_completed_history_from_other_queue(tmp_path):
    folder = tmp_path / 'geometry'
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(dict(generation=str(folder))))
    job = dict(name='actual-held-reader', command=['python', '--plan', str(plan)])
    policy = queues(tmp_path, job, dict(status='HELD'), aux_state=dict(status='completed', returncode=0))
    jobs, states, _ = actual_queue_jobs(policy)
    assert pending_reads(jobs, states, [folder]) == [job['name']]


def test_two_actual_descriptors_cannot_share_an_owner(tmp_path):
    job = dict(name='duplicated', command=[])
    policy = queues(tmp_path, job, dict(status='completed'), aux_job=job)
    with pytest.raises(ValueError, match='one actual dispatcher owner'):
        actual_queue_jobs(policy)


@pytest.mark.parametrize('problem', [None, 'failed', 'changed-bytes', 'outside', 'duplicate',
    'still-present', 'symlink', 'missing-proof', 'not-canonical', 'wrong-size'])
def test_only_explicit_closed_prior_batches_establish_retired_files(tmp_path, problem):
    raw = tmp_path / 'owned/case/geometry/prediction.npz'
    rows = [dict(case_root=str(raw.parent.parent))]
    audit = dict(status='PASS_COMPLETED_DEV_LATE_GEOMETRY_LOCAL_DUPLICATES_RETIRED',
        canonical_file4_retained=True, guides_pixels_metrics_and_checkpoints_retained=True,
        original505_numerical_sources_unchanged=True, bytes_retired=20,
        retired=[dict(path=str(raw), bytes=20, sha256='a' * 64,
            local_duplicate_only=True, canonical_file4_retained=True)])
    path = tmp_path / 'closed-batch.json'
    if problem == 'failed': audit['status'] = 'FAILED'
    if problem == 'outside': audit['retired'][0]['path'] = str(tmp_path / 'other/prediction.npz')
    if problem == 'not-canonical': audit['retired'][0]['canonical_file4_retained'] = False
    if problem == 'wrong-size': audit['bytes_retired'] = 19
    path.write_text(json.dumps(audit))
    spec = dict(path=str(path), SHA256=sha256(path.read_bytes()).hexdigest(), receipt=str(tmp_path / 'receipt.json'))
    if problem == 'changed-bytes': path.write_text('{}')
    if problem in ['still-present', 'symlink']:
        raw.parent.mkdir(parents=True)
        if problem == 'symlink': raw.symlink_to(tmp_path / 'absent-target')
        else: raw.write_bytes(b'actual bytes')
    def read(path, receipt):
        if problem == 'missing-proof': raise ValueError('actual closed receipt unavailable')
        assert receipt == spec['receipt']
        return json.loads(open(path).read())
    policy = dict(prior_retirement_batches=[spec, spec] if problem == 'duplicate' else [spec])
    if problem is None:
        assert prior_closed_retirements(policy, rows, read) == {str(raw): audit['retired'][0]}
    else:
        with pytest.raises(ValueError): prior_closed_retirements(policy, rows, read)
