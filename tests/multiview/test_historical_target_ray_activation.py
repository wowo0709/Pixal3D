import hashlib
import json
from pathlib import Path
import sys

import pytest

from experiments.deformnext_retargeting import activate_historical_target_ray_reproduction as activation
from experiments.deformnext_retargeting import historical_target_ray_summary as summary
from experiments.deformnext_retargeting.historical_target_ray_reproduction import ARMS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fake_closer(module, monkeypatch):
    class Closed:
        def __init__(self, _): self.inputs = {}
        def check(self, *_): pass
    monkeypatch.setattr(module, 'ClosedInputs', Closed)
    monkeypatch.setattr(module.os, 'sched_setaffinity', lambda *_: None)
    monkeypatch.setattr(module.os, 'nice', lambda *_: None)
    calls = []
    monkeypatch.setattr(module.subprocess, 'run', lambda args, **kwargs: calls.append(args))
    return calls


@pytest.mark.parametrize('supported', [False, True])
def test_cpu_activation_actual_queue_prefix_and_positive_branch_no_write(tmp_path, monkeypatch, supported):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('MATCHING_V2_ALLOWED_GPUS', '0,1,2,3')
    calls = fake_closer(activation, monkeypatch)
    queue = tmp_path/'queue.json'; write(queue, [dict(name='existing', resource='exclusive')]); before = queue.read_bytes()
    selection = tmp_path/'selection.json'; write(selection, dict(
        selected_candidate='member-qkv' if supported else None,
        status='PROMISING_NATIVE_SOURCE_COMBINATION_FOR_SEED_CONFIRMATION' if supported else 'NO_PROMISING_NATIVE_SOURCE_COMBINATION_YET',
        heldout_test_predictions_read=False, model_adopted=False))
    index = tmp_path/'index.json'; write(index, dict(models={arm: dict(GPU_pair=[0, 1], phases={
        phase: dict(plan='/plan/'+arm+'/'+phase, plan_SHA256='pinned', receipt='/receipt', output_receipt='/output')
        for phase in ['capacity', 'science', 'noise0', 'noise1', 'noise2']}) for arm in ARMS}))
    out = tmp_path/'activation'
    plan = dict(repo='/source', python='python', canonical_local_root=str(tmp_path), canonical_closer='closer',
        CPU_affinity=[0], assigned_GPUs=[0, 1, 2, 3], source_SHA256={},
        selection=str(selection), selection_receipt='/selection-receipt', templates_index=str(index),
        templates_receipt='/templates-receipt', queue_path=str(queue), output=str(out),
        activation_job='activate', activation_receipt=str(out/'file4-closure.json'),
        summary_receipt='/summary-receipt', summary_plan='/summary-plan', summary_plan_SHA256='summary')
    path = tmp_path/'activation-plan.json'; write(path, plan)
    monkeypatch.setattr(sys, 'argv', ['activate', '--plan', str(path), '--plan-sha256', hashlib.sha256(path.read_bytes()).hexdigest()])
    activation.main()
    assert len(calls) == 1
    current = json.loads(queue.read_text())
    if supported:
        assert queue.read_bytes() == before
    else:
        assert current[:1] == json.loads(before) and len(current) == 17
    assert json.loads((out/'activation.json').read_text())['appended_jobs'] == (0 if supported else 16)


def test_summary_pairs_all_three_sampling_noises_and_historical_models(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    calls = fake_closer(summary, monkeypatch)
    models = {}
    for arm in ARMS:
        item = dict(evaluations=[], historical=[])
        for group in item:
            for noise in range(3):
                path = tmp_path/arm/group/f'noise{noise}.json'
                data = dict(status='PASS', step=480, optimizer_updates_during_evaluation=0,
                    sampling_steps=12, guidance=3., per_asset=[dict(sample_id=str(i),
                        source_cluster_id=str(i//4), seed=noise*1000+i, tp=10, fp=2, fn=3,
                        target_voxels_sha256='same', target_voxel_shape=[64, 64, 64]) for i in range(128)])
                write(path, data); item[group].append(dict(path=str(path), receipt='/unit-only-receipt'))
        models[arm] = item
    out = tmp_path/'summary'
    plan = dict(canonical_local_root=str(tmp_path), CPU_affinity=[0], source_SHA256={}, models=models,
        output=str(out), python='python', canonical_closer='closer')
    path = tmp_path/'summary-plan.json'; write(path, plan)
    monkeypatch.setattr(sys, 'argv', ['summary', '--plan', str(path), '--plan-sha256', hashlib.sha256(path.read_bytes()).hexdigest()])
    summary.main()
    result = json.loads((out/'summary.json').read_text())
    assert len(result['contrasts']) == 6 and len(calls) == 1
    assert all(x['gain'] == 0 and x['sampling_noise_seeds'] == 3
               and x['paired_target_hashes_verified'] == 384 for x in result['contrasts'].values())
    assert not result['independent_training_seed_confirmation'] and not result['full_goal_complete']
