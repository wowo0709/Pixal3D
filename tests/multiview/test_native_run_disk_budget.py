import json

import pytest

from experiments.deformnext_retargeting.native_run_disk_budget import DiskBudget


def test_atomic_admission_prevents_two_runs_overcommitting_same_free_space(tmp_path):
    store = DiskBudget(tmp_path/'registry.json', workspace_bytes=8, free_bytes=lambda: 39)
    store.reserve('one', checkpoint_budget_bytes=16, native_output=tmp_path/'one',
        wrapper_output=tmp_path/'one-wrapper', owner={'PID': 123, 'birth': 10.})
    with pytest.raises(ValueError, match='actual free disk'):
        store.reserve('two', checkpoint_budget_bytes=16, native_output=tmp_path/'two',
            wrapper_output=tmp_path/'two-wrapper', owner={'PID': 124, 'birth': 11.})
    assert list(json.loads((tmp_path/'registry.json').read_text())['runs']) == ['one']


def test_written_bytes_reduce_future_growth_but_missing_PID_does_not_release(tmp_path):
    root = tmp_path/'one'; (root/'ckpts').mkdir(parents=True)
    store = DiskBudget(tmp_path/'registry.json', workspace_bytes=8, free_bytes=lambda: 100)
    store.reserve('one', checkpoint_budget_bytes=16, native_output=root,
        wrapper_output=tmp_path/'wrapper', owner={'PID': -1, 'birth': 0.})
    (root/'ckpts/denoiser_step0000380.pt').write_bytes(b'123456789012')
    assert store.inspect(16)['outstanding_future_bytes'] == 4
    with pytest.raises(ValueError, match='cannot be reused'):
        store.reserve('one', checkpoint_budget_bytes=16, native_output=root,
            wrapper_output=tmp_path/'wrapper', owner={})


def test_release_requires_own_actual_canonical_wrapper_receipt(tmp_path):
    wrapper = tmp_path/'wrapper'; wrapper.mkdir()
    store = DiskBudget(tmp_path/'registry.json', workspace_bytes=8, free_bytes=lambda: 100)
    store.reserve('one', checkpoint_budget_bytes=16, native_output=tmp_path/'one', wrapper_output=wrapper, owner={})
    receipt = wrapper/'file4-closure.json'
    receipt.write_text(json.dumps(dict(status='FAIL', local_equals_actual_file4_SHA256=False)))
    with pytest.raises(ValueError): store.release('one', actual_wrapper_receipt=receipt)
    assert store.inspect(0)['outstanding_future_bytes'] == 16
    # Synthetic receipt checks policy flow only; it is not real file4 evidence.
    receipt.write_text(json.dumps(dict(status='PASS', local_equals_actual_file4_SHA256=True, synthetic_unit=True)))
    store.release('one', actual_wrapper_receipt=receipt)
    assert store.inspect(0)['outstanding_future_bytes'] == 0
