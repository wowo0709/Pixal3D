import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting import historical_target_ray_stage as stage
from test_historical_target_ray_reproduction import reports


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def harness(tmp_path, monkeypatch, phase):
    output, native = tmp_path/'result', tmp_path/'result/native'
    golden = reports('direct-dq')
    paths = {}
    for rank in [0, 1]:
        p = tmp_path/f'golden{rank}.json'; write(p, golden[rank]); paths[str(rank)] = str(p)
    checkpoint = tmp_path/'frozen.pt'; checkpoint.write_bytes(b'frozen checkpoint')
    proof = tmp_path/'own-capacity.json'
    write(proof, dict(status='PASS_ACTUAL_TWO_RANK_R96_TARGET_PATCH_CAPACITY_REPRODUCTION'))
    plan = dict(phase=phase, arm='direct-dq', canonical_local_root=str(tmp_path), repo=str(tmp_path),
        python='python', canonical_closer='/explicit/CPU-closer.py', assigned_GPUs=[0, 1, 2, 3],
        CPU_affinity_per_GPU={str(g): [84+2*g, 85+2*g] for g in range(4)},
        output=str(output), native_output=str(native), command=['explicit-native-entry'],
        source_SHA256={}, input_SHA256={}, closed_inputs={}, checkpoint_inputs=[str(checkpoint)],
        capacity_audit=str(proof), golden_profiles=paths, noise=0)
    path = tmp_path/'plan.json'; write(path, plan)
    monkeypatch.setenv('MATCHING_V2_ALLOWED_GPUS', '0,1,2,3')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2' if phase == 'evaluation' else '0,1')
    monkeypatch.setattr(stage.os, 'sched_setaffinity', lambda *_: None)

    class Closed:
        def __init__(self, _): self.inputs = {}
        def check(self, *_): pass

    calls, closures = [], []

    class Process:
        pid = 99999999

        def wait(self):
            calls.append('native')
            native.mkdir()
            if phase == 'capacity':
                for rank in [0, 1]: write(native/f'joint-profile.rank{rank}.json', golden[rank])
            elif phase == 'science':
                write(native/'progress.json', dict(stage='completed', current=480))
                (native/'validation.jsonl').write_text('\n'.join(json.dumps(dict(
                    step=step, quick_per_asset=[{}]*32, quick_metrics={'voxel_iou': .4}))
                    for step in [96, 192, 288, 384, 480]))
                for rank in [0, 1]: write(native/f'anchor-stage-initialization.rank{rank}.json',
                    dict(restored_common_Adam_states=366, reference_step=96))
                for key in ['denoiser', 'deform_conditioner', 'misc']:
                    p = native/f'ckpts/{key}_step0000480.pt'; p.parent.mkdir(exist_ok=True); p.write_bytes(b'unit-only')
            else:
                write(native/'evaluation.json', dict(status='PASS', stage='ss64', step=480,
                    per_asset=[{}]*128, optimizer_updates_during_evaluation=0,
                    deterministic_evaluation=dict(actual_deterministic_algorithms=True), quick_metrics={}))
            return 0

    monkeypatch.setattr(stage, 'ClosedInputs', Closed)
    monkeypatch.setattr(stage.psutil, 'Process', lambda _: SimpleNamespace(create_time=lambda: 1.))
    monkeypatch.setattr(stage.subprocess, 'Popen', lambda *args, **kwargs: Process())
    monkeypatch.setattr(stage.subprocess, 'run', lambda args, **kwargs: closures.append(args))
    return path, plan, calls, closures, golden


@pytest.mark.parametrize('phase', ['capacity', 'science', 'evaluation'])
def test_native_phase_contract_and_actual_schema_keys_with_mocked_producer(tmp_path, monkeypatch, phase):
    path, plan, calls, closures, golden = harness(tmp_path, monkeypatch, phase)
    result = stage.execute(path, stage.sha(path))
    assert calls == ['native'] and len(closures) == 1
    assert not result['full_goal_complete']
    assert Path(plan['output'], 'terminal.json').is_file()
    assert closures[0][closures[0].index('--receipt')+1] == 'result/file4-closure.json'


def test_capacity_same_rank_divergence_closes_failure_and_does_not_release_science(tmp_path, monkeypatch):
    path, plan, calls, closures, golden = harness(tmp_path, monkeypatch, 'capacity')
    golden[1]['updates'][0]['loss']['mse'] += .001
    with pytest.raises(ValueError, match='same-rank'):
        stage.execute(path, stage.sha(path))
    assert calls == ['native'] and len(closures) == 1
    assert closures[0][closures[0].index('--receipt')+1] == 'result/failure-file4-closure.json'
    assert not Path(plan['output'], 'audit.json').exists()


def test_missing_own_capacity_proof_prevents_scientific_native_launch(tmp_path, monkeypatch):
    path, plan, calls, closures, _ = harness(tmp_path, monkeypatch, 'science')
    write(Path(plan['capacity_audit']), dict(status='OLD_OR_OTHER_ARM_CAPACITY'))
    with pytest.raises(ValueError, match='own two-rank'):
        stage.execute(path, stage.sha(path))
    assert not calls and len(closures) == 1


def test_wrong_GPU_count_cannot_create_output_or_start_native(tmp_path, monkeypatch):
    path, plan, calls, closures, _ = harness(tmp_path, monkeypatch, 'science')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='DDP pair'):
        stage.execute(path, stage.sha(path))
    assert not calls and not closures and not Path(plan['output']).exists()
