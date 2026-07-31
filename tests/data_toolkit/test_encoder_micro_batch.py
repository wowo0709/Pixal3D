import time
import threading

import pytest
import torch

from data_toolkit.pipeline import sparse_batching
from data_toolkit.pipeline.sparse_batching import (
    batch_sparse_tensors,
    run_encoder_tasks,
    split_sparse_tensor,
    validate_record_prefix,
)
from pixal3d.modules.sparse.basic import SparseTensor


def test_eight_tasks_at_four_use_two_model_calls():
    call_sizes = []

    def process_batch(payloads):
        call_sizes.append(len(payloads))
        return [payload * 10 for payload in payloads]

    records = run_encoder_tasks(
        tasks=list(range(8)),
        micro_batch_size=4,
        load=lambda task, cancel: (task, None),
        process_batch=process_batch,
        save=lambda task, payload, cancel: {
            "task": task,
            "value": payload,
        },
    )

    assert call_sizes == [4, 4]
    assert records == [
        {"task": task, "value": task * 10} for task in range(8)
    ]


def test_oom_halves_pending_batch_without_losing_tasks(monkeypatch):
    # Keep this synthetic halving test independent from CUDA state initialized
    # while collecting unrelated tests; adaptive low-peak regrowth is separate.
    monkeypatch.setattr(sparse_batching.torch.cuda, "is_initialized", lambda: False)
    call_sizes = []
    failed = False

    def process_batch(payloads):
        nonlocal failed
        call_sizes.append(len(payloads))
        if len(payloads) == 4 and not failed:
            failed = True
            raise torch.OutOfMemoryError("CUDA out of memory")
        return list(payloads)

    records = run_encoder_tasks(
        tasks=list(range(6)),
        micro_batch_size=4,
        load=lambda task, cancel: (task, None),
        process_batch=process_batch,
        save=lambda task, payload, cancel: task,
    )

    assert call_sizes == [4, 2, 2, 2]
    assert records == list(range(6))


def test_non_oom_model_error_is_not_retried():
    def fail(payloads):
        raise RuntimeError("model contract violation")

    with pytest.raises(RuntimeError, match="model contract violation"):
        run_encoder_tasks(
            tasks=[0, 1],
            micro_batch_size=2,
            load=lambda task, cancel: (task, None),
            process_batch=fail,
            save=lambda task, payload, cancel: task,
        )


def test_records_remain_in_task_order_with_out_of_order_io():
    def load(task, cancel):
        time.sleep(0.01 * (3 - task))
        return task, None

    def save(task, payload, cancel):
        time.sleep(0.01 * task)
        return task

    records = run_encoder_tasks(
        tasks=[0, 1, 2, 3],
        micro_batch_size=2,
        load=load,
        process_batch=lambda payloads: list(payloads),
        save=save,
        loader_workers=4,
        saver_workers=4,
    )

    assert records == [0, 1, 2, 3]


def test_encoder_workers_are_joined_before_return(monkeypatch):
    original_thread = threading.Thread
    created_threads = []

    class RecordingThread(original_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.was_joined = False
            created_threads.append(self)

        def join(self, *args, **kwargs):
            self.was_joined = True
            return super().join(*args, **kwargs)

    monkeypatch.setattr(sparse_batching.threading, "Thread", RecordingThread)

    assert run_encoder_tasks(
        tasks=[0, 1],
        micro_batch_size=2,
        load=lambda task, cancel: (task, None),
        process_batch=lambda payloads: list(payloads),
        save=lambda task, payload, cancel: task,
    ) == [0, 1]

    assert created_threads
    assert all(thread.was_joined for thread in created_threads)
    assert not any(thread.is_alive() for thread in created_threads)


@pytest.mark.parametrize("value", ["../chunk", "chunk/", "chunk\\", "bad\0"])
def test_record_prefix_rejects_path_separators(value):
    with pytest.raises(ValueError, match="record prefix"):
        validate_record_prefix(value)


def test_record_prefix_accepts_empty_and_chunk_identifiers():
    assert validate_record_prefix("") == ""
    assert validate_record_prefix("chunk007_") == "chunk007_"


def _sparse(feats, coords, device="cpu"):
    return SparseTensor(
        torch.tensor(feats, dtype=torch.float32, device=device),
        torch.tensor(coords, dtype=torch.int32, device=device),
    )


def _encode_sparse_batch(values):
    combined = batch_sparse_tensors(values)
    encoded = combined.replace(combined.feats * 2.0 + 1.0)
    return split_sparse_tensor(encoded)


def test_sparse_batch_four_matches_batch_one_on_backend():
    values = [
        _sparse([[float(index)]], [[0, index + 1, 2, 3]])
        for index in range(4)
    ]

    batch_one = [_encode_sparse_batch([value])[0] for value in values]
    batch_four = _encode_sparse_batch(values)

    for expected, actual in zip(batch_one, batch_four):
        assert torch.equal(actual.coords, expected.coords)
        torch.testing.assert_close(
            actual.feats, expected.feats, rtol=1e-6, atol=1e-6
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparse_batch_four_matches_batch_one_on_cuda_backend():
    values = [
        _sparse(
            [[float(index)]],
            [[0, index + 1, 2, 3]],
            device="cuda",
        )
        for index in range(4)
    ]

    batch_one = [_encode_sparse_batch([value])[0] for value in values]
    batch_four = _encode_sparse_batch(values)

    for expected, actual in zip(batch_one, batch_four):
        assert torch.equal(actual.coords, expected.coords)
        torch.testing.assert_close(
            actual.feats, expected.feats, rtol=1e-6, atol=1e-6
        )
