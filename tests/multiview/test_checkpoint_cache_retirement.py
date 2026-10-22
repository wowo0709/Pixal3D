from pathlib import Path
import json

import pytest

from experiments.deformnext_retargeting.checkpoint_cache_retirement import (
    pending_checkpoint_reads, validate_retirement_candidate,
    remaining_checkpoint_budget, disk_admission,
)


def arguments(tmp_path):
    root = tmp_path/'owned'; (root/'ckpts').mkdir(parents=True)
    path = root/'ckpts/denoiser_step0000192.pt'; path.write_bytes(b'unit fixture, not actual model weights')
    return path, dict(run_root=root, final_step=480, allowed_steps={96, 192, 288, 384},
        terminal=dict(returncode=0), progress=dict(stage='completed', current=480),
        canonical_entry=dict(sha256='verified', bytes=path.stat().st_size), actual_local_sha='verified', pending_reads={})


def test_retirement_decision_never_deletes_and_keeps_final_and_other_namespace(tmp_path):
    path, args = arguments(tmp_path)
    assert validate_retirement_candidate(path, **args)['step'] == 192 and path.is_file()
    for update in [dict(final_step=192), dict(allowed_steps={96}), dict(run_root=tmp_path/'other'),
                   dict(terminal={'returncode': 1}), dict(progress={'stage': 'running', 'current': 192}),
                   dict(actual_local_sha='changed'), dict(pending_reads={str(path): ['reader']})]:
        with pytest.raises(ValueError): validate_retirement_candidate(path, **dict(args, **update))
    assert path.is_file()


def test_own_checkpoint_symlinks_cannot_cross_into_original_data(tmp_path):
    path, args = arguments(tmp_path)
    original = tmp_path/'original.pt'; original.write_bytes(path.read_bytes())
    path.unlink(); path.symlink_to(original)
    with pytest.raises(ValueError): validate_retirement_candidate(path, **args)
    assert original.exists()


def test_pending_actual_bundle_or_input_hash_protects_checkpoint_but_receipt_index_does_not(tmp_path):
    path, _ = arguments(tmp_path)
    bundle = tmp_path/'bundle.json'; bundle.write_text(json.dumps(dict(checkpoint=str(path))))
    plan = tmp_path/'plan.json'
    job = dict(name='read', command=['worker', '--plan', str(plan)])
    states = {'read': dict(status='pending')}
    for value in [dict(bundle=str(bundle)), dict(input_SHA256={str(path): 'hash'}),
                  dict(closed_inputs={str(path): 'receipt.json'}), dict(checkpoint=str(path))]:
        plan.write_text(json.dumps(value))
        assert str(path) in pending_checkpoint_reads([job], states, [path])
        assert str(path) in pending_checkpoint_reads([job], {}, [path])
    plan.write_text(json.dumps(dict(files=[str(path)], verified_inputs={str(path): 'hash'})))
    assert not pending_checkpoint_reads([job], states, [path])
    assert not pending_checkpoint_reads([job], {'read': dict(status='completed')}, [path])


def test_missing_future_bundle_conservatively_protects_its_step(tmp_path):
    path, _ = arguments(tmp_path)
    plan = tmp_path/'plan.json'; plan.write_text(json.dumps(dict(bundle=str(tmp_path/'bundles-step192/arm/bundle.json'))))
    job = dict(name='future-eval', command=['worker', '--plan', str(plan)])
    assert str(path) in pending_checkpoint_reads([job], {'future-eval': dict(status='pending')}, [path])


def test_future_growth_reservation_does_not_double_count_already_written_checkpoints():
    # Units can be GiB in this arithmetic fixture; runtime passes actual bytes.
    remaining = remaining_checkpoint_budget(16, [3, 3, 3, 3])
    assert remaining == 4
    assert disk_admission(actual_free=30, new_budget=16, outstanding=[remaining], workspace=8)['admitted']
    assert not disk_admission(actual_free=39, new_budget=16, outstanding=[16], workspace=8)['admitted']
    assert disk_admission(actual_free=40, new_budget=16, outstanding=[16], workspace=8)['admitted']
    assert remaining_checkpoint_budget(16, [20]) == 0


def test_checkpoint_directory_symlink_cannot_turn_cache_retirement_into_original_deletion(tmp_path):
    original = tmp_path/'original'; original.mkdir()
    path = original/'denoiser_step0000192.pt'; path.write_bytes(b'original')
    root = tmp_path/'owned'; root.mkdir(); (root/'ckpts').symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError):
        validate_retirement_candidate(root/'ckpts'/path.name, run_root=root, final_step=480,
            allowed_steps={192}, terminal={'returncode':0}, progress={'stage':'completed','current':480},
            canonical_entry={'sha256':'verified','bytes':8}, actual_local_sha='verified', pending_reads={})
    assert path.exists()
