from __future__ import annotations

from collections.abc import Sequence
import math
from queue import Empty, Full, Queue
import threading
import time

import torch


INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def validate_record_prefix(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("record prefix must be a string")
    if any(separator in value for separator in ("/", "\\", "\0")):
        raise ValueError("record prefix must not contain path separators")
    return value


def micro_batches(tasks, size: int) -> list[list]:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("micro-batch size must be a positive integer")
    values = list(tasks)
    return [
        values[index : index + size]
        for index in range(0, len(values), size)
    ]


def _validate_layout(tensor) -> tuple[slice, ...]:
    feats = getattr(tensor, "feats", None)
    coords = getattr(tensor, "coords", None)
    if not isinstance(feats, torch.Tensor) or not isinstance(
        coords, torch.Tensor
    ):
        raise TypeError("sparse tensors must expose tensor feats and coords")
    if feats.ndim < 1 or coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("invalid sparse tensor dimensions")
    if feats.shape[0] != coords.shape[0] or feats.shape[0] == 0:
        raise ValueError("sparse features and coordinates must be nonempty")
    if not bool(torch.isfinite(feats).all()):
        raise ValueError("sparse features must be finite")
    if coords.dtype not in INTEGER_DTYPES:
        raise ValueError("sparse coordinates must use an integral dtype")

    layout = tuple(tensor.layout)
    if not layout:
        raise ValueError("sparse layout cannot be empty")
    offset = 0
    for batch_index, item in enumerate(layout):
        if (
            not isinstance(item, slice)
            or item.step not in (None, 1)
            or item.start != offset
            or item.stop is None
            or item.stop <= item.start
        ):
            raise ValueError("sparse layout must be contiguous and nonempty")
        if not bool(torch.all(coords[item, 0] == batch_index)):
            raise ValueError("sparse layout batch coordinates are not contiguous")
        offset = item.stop
    if offset != feats.shape[0]:
        raise ValueError("sparse layout must cover every coordinate")
    return layout


def batch_sparse_tensors(tensors: Sequence) -> object:
    values = list(tensors)
    if not values:
        raise ValueError("cannot batch zero sparse tensors")
    tensor_type = type(values[0])
    if any(type(value) is not tensor_type for value in values):
        raise TypeError("all values must use the same sparse tensor type")

    reference_feats = values[0].feats
    reference_coords = values[0].coords
    for value in values:
        layout = _validate_layout(value)
        if len(layout) != 1:
            raise ValueError("batch inputs must each contain one sparse sample")
        if (
            value.feats.dtype != reference_feats.dtype
            or value.feats.device != reference_feats.device
            or value.feats.shape[1:] != reference_feats.shape[1:]
        ):
            raise ValueError("sparse feature contracts must match")
        if (
            value.coords.dtype != reference_coords.dtype
            or value.coords.device != reference_coords.device
            or value.coords.shape[1:] != reference_coords.shape[1:]
        ):
            raise ValueError("sparse coordinate contracts must match")

    combined = tensor_type.from_tensor_list(
        [value.feats for value in values],
        [value.coords for value in values],
    )
    if type(combined) is not tensor_type or len(_validate_layout(combined)) != len(
        values
    ):
        raise ValueError("sparse batch output count or type is invalid")
    return combined


def split_sparse_tensor(tensor) -> list:
    layout = _validate_layout(tensor)
    outputs = []
    for item in layout:
        coords = tensor.coords[item].clone()
        coords[:, 0] = 0
        output = type(tensor)(tensor.feats[item], coords)
        if len(_validate_layout(output)) != 1:
            raise ValueError("split sparse output must contain one sample")
        outputs.append(output)
    if len(outputs) != len(layout):
        raise ValueError("split sparse output count is invalid")
    return outputs


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
        and "cuda" in str(error).lower()
    )


def run_encoder_tasks(
    *,
    tasks,
    micro_batch_size: int,
    load,
    process_batch,
    save,
    loader_workers: int = 2,
    saver_workers: int = 1,
    timeout_seconds: float = 300,
    gpu_memory_target_percent: float = 80.0,
):
    """Run ordered encoder tasks with bounded I/O and adaptive OOM retry."""
    if (
        type(micro_batch_size) is not int
        or micro_batch_size <= 0
    ):
        raise ValueError("micro-batch size must be a positive integer")
    for name, value in (
        ("loader workers", loader_workers),
        ("saver workers", saver_workers),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout seconds must be finite and positive")
    if (
        not isinstance(gpu_memory_target_percent, (int, float))
        or isinstance(gpu_memory_target_percent, bool)
        or not math.isfinite(gpu_memory_target_percent)
        or not 0 < gpu_memory_target_percent < 100
    ):
        raise ValueError("GPU memory target percent must be in (0, 100)")
    if not callable(load) or not callable(process_batch) or not callable(save):
        raise TypeError("load, process_batch, and save must be callable")

    indexed_tasks = list(enumerate(tasks))
    if not indexed_tasks:
        return []

    cancel_event = threading.Event()
    loader_tasks = Queue(maxsize=max(2, loader_workers * 2))
    loader_results = Queue(maxsize=max(2, loader_workers * 2))
    saver_tasks = Queue(maxsize=max(2, saver_workers * 2))
    saver_results = Queue(maxsize=max(2, saver_workers * 2))
    errors = Queue(maxsize=max(2, loader_workers + saver_workers))
    sentinel = object()

    def put_cancellable(queue, value) -> bool:
        while not cancel_event.is_set():
            try:
                queue.put(value, timeout=0.05)
                return True
            except Full:
                continue
        return False

    def report_error(stage, index, task, error) -> None:
        try:
            errors.put_nowait((stage, index, task, error))
        except Full:
            pass
        cancel_event.set()

    def raise_worker_error() -> None:
        try:
            stage, _, task, error = errors.get_nowait()
        except Empty:
            return
        raise RuntimeError(f"{stage} failed for {task}: {error}") from error

    def wait_result(queue, stage):
        deadline = time.monotonic() + timeout_seconds
        while True:
            raise_worker_error()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{stage} inactivity timed out after "
                    f"{timeout_seconds} seconds"
                )
            try:
                return queue.get(timeout=min(0.05, remaining))
            except Empty:
                continue

    def produce() -> None:
        for item in indexed_tasks:
            if not put_cancellable(loader_tasks, item):
                return
        for _ in range(loader_workers):
            if not put_cancellable(loader_tasks, sentinel):
                return

    def loader_worker() -> None:
        while not cancel_event.is_set():
            try:
                item = loader_tasks.get(timeout=0.05)
            except Empty:
                continue
            if item is sentinel:
                return
            index, task = item
            try:
                loaded = load(task, cancel_event)
            except BaseException as error:
                report_error("loader", index, task, error)
                return
            if not put_cancellable(loader_results, (index, task, loaded)):
                return

    def saver_worker() -> None:
        while not cancel_event.is_set():
            try:
                item = saver_tasks.get(timeout=0.05)
            except Empty:
                continue
            if item is sentinel:
                return
            index, task, payload = item
            try:
                record = save(task, payload, cancel_event)
            except BaseException as error:
                report_error("saver", index, task, error)
                return
            if not put_cancellable(
                saver_results, (index, task, record)
            ):
                return

    threads = [threading.Thread(target=produce, daemon=True)]
    threads.extend(
        threading.Thread(target=loader_worker, daemon=True)
        for _ in range(loader_workers)
    )
    threads.extend(
        threading.Thread(target=saver_worker, daemon=True)
        for _ in range(saver_workers)
    )
    for thread in threads:
        thread.start()

    loaded_by_index = {}
    records_by_index = {}
    pending_batch = []
    pending_saves = set()
    next_loaded_index = 0
    received_loads = 0
    current_batch_size = micro_batch_size

    def drain_savers() -> None:
        while True:
            raise_worker_error()
            try:
                index, _, record = saver_results.get_nowait()
            except Empty:
                return
            pending_saves.remove(index)
            if record is not None:
                records_by_index[index] = record

    def enqueue_save(index, task, payload) -> None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            drain_savers()
            try:
                saver_tasks.put_nowait((index, task, payload))
                pending_saves.add(index)
                return
            except Full:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "saver queue timed out while awaiting capacity"
                    )
                time.sleep(0.01)

    def process_ready(*, flush: bool) -> None:
        nonlocal current_batch_size
        while len(pending_batch) >= current_batch_size or (
            flush and pending_batch
        ):
            count = min(current_batch_size, len(pending_batch))
            selected = pending_batch[:count]
            cuda_was_initialized = torch.cuda.is_initialized()
            if cuda_was_initialized:
                torch.cuda.reset_peak_memory_stats()
            try:
                outputs = list(
                    process_batch([item[2] for item in selected])
                )
            except BaseException as error:
                if not _is_cuda_oom(error) or current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                continue
            if len(outputs) != len(selected):
                raise ValueError(
                    "encoder batch output count must match its input count"
                )
            if torch.cuda.is_initialized():
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(
                    device
                ).total_memory
                peak_percent = (
                    torch.cuda.max_memory_reserved(device)
                    * 100.0
                    / total_memory
                )
                if peak_percent > gpu_memory_target_percent:
                    current_batch_size = max(1, current_batch_size // 2)
                elif peak_percent < 70.0:
                    current_batch_size = min(
                        micro_batch_size,
                        current_batch_size * 2,
                    )
            del pending_batch[:count]
            for (index, task, _), output in zip(selected, outputs):
                if output is not None:
                    enqueue_save(index, task, output)

    completed = False
    try:
        while received_loads < len(indexed_tasks):
            drain_savers()
            index, task, loaded = wait_result(loader_results, "loader")
            received_loads += 1
            if (
                not isinstance(loaded, tuple)
                or len(loaded) != 2
            ):
                raise TypeError(
                    "loader must return (payload, immediate_record)"
                )
            loaded_by_index[index] = (task, loaded[0], loaded[1])
            while next_loaded_index in loaded_by_index:
                next_task, payload, immediate_record = loaded_by_index.pop(
                    next_loaded_index
                )
                if immediate_record is not None:
                    records_by_index[next_loaded_index] = immediate_record
                if payload is not None:
                    pending_batch.append(
                        (next_loaded_index, next_task, payload)
                    )
                next_loaded_index += 1
                process_ready(flush=False)

        process_ready(flush=True)
        while pending_saves:
            index, _, record = wait_result(saver_results, "saver")
            pending_saves.remove(index)
            if record is not None:
                records_by_index[index] = record
        records = [
            records_by_index[index]
            for index in sorted(records_by_index)
        ]
        completed = True
        return records
    finally:
        cancel_event.set()
        for thread in threads:
            thread.join(timeout=None if completed else 0.25)
