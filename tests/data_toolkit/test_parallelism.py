import pytest

from data_toolkit.pipeline.parallelism import (
    GpuMemoryState,
    NodeResourceBroker,
    select_micro_batch,
)


def test_gpu_memory_state_reports_percent():
    state = GpuMemoryState(index=2, used_mib=48_943.5, total_mib=97_887.0)

    assert state.percent == pytest.approx(50.0)


def test_gpu_memory_state_rejects_invalid_values():
    with pytest.raises(ValueError, match="total"):
        GpuMemoryState(index=0, used_mib=1.0, total_mib=0.0)
    with pytest.raises(ValueError, match="used"):
        GpuMemoryState(index=0, used_mib=-1.0, total_mib=10.0)


def test_micro_batch_steps_down_above_target(config):
    assert select_micro_batch(
        resolution=1024,
        configured=4,
        peak_percent=84.0,
        oom=False,
        config=config.parallelism,
    ) == 2


def test_micro_batch_halves_after_oom(config):
    assert select_micro_batch(
        resolution=512,
        configured=8,
        peak_percent=50.0,
        oom=True,
        config=config.parallelism,
    ) == 4


def test_micro_batch_steps_up_only_below_seventy_percent(config):
    assert select_micro_batch(
        resolution=256,
        configured=4,
        peak_percent=69.9,
        oom=False,
        config=config.parallelism,
    ) == 8
    assert select_micro_batch(
        resolution=256,
        configured=8,
        peak_percent=70.0,
        oom=False,
        config=config.parallelism,
    ) == 8


def test_broker_never_oversubscribes_cpu_and_release_is_idempotent():
    broker = NodeResourceBroker(cpu_limit=44, gpu_count=7)
    first = broker.try_acquire(cpu_cores=24, gpu_indices=())
    second = broker.try_acquire(cpu_cores=20, gpu_indices=())

    assert first is not None and second is not None
    assert broker.try_acquire(cpu_cores=1, gpu_indices=()) is None
    second.release()
    second.release()
    assert broker.try_acquire(cpu_cores=1, gpu_indices=()) is not None


def test_broker_rejects_gpu_admission_at_hard_limit():
    broker = NodeResourceBroker(cpu_limit=44, gpu_count=7, gpu_hard_percent=90.0)

    assert broker.try_acquire(
        cpu_cores=0, gpu_indices=(0,), gpu_memory_percent=89.9
    ) is not None
    assert broker.try_acquire(
        cpu_cores=0, gpu_indices=(1,), gpu_memory_percent=90.0
    ) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cpu_cores": True, "gpu_indices": ()},
        {"cpu_cores": -1, "gpu_indices": ()},
        {"cpu_cores": 0, "gpu_indices": (7,)},
        {"cpu_cores": 0, "gpu_indices": (0, 0)},
        {"cpu_cores": 0, "gpu_indices": (0,), "gpu_memory_percent": float("nan")},
    ],
)
def test_broker_rejects_invalid_requests(kwargs):
    broker = NodeResourceBroker(cpu_limit=44, gpu_count=7)

    with pytest.raises(ValueError):
        broker.try_acquire(**kwargs)
