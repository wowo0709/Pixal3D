import pytest

from data_toolkit.pipeline.parallelism import (
    GeometryProfile,
    GpuMemoryState,
    NodeResourceBroker,
    configure_geometry_threads,
    geometry_affinity_sets,
    geometry_profile,
    select_micro_batch,
)


def test_geometry_thread_cap_matches_affinity_width():
    calls = []

    class FakeTorch:
        @staticmethod
        def set_num_threads(value):
            calls.append(("threads", value))

        @staticmethod
        def set_num_interop_threads(value):
            calls.append(("interop", value))

    configure_geometry_threads(4, torch_module=FakeTorch())

    assert calls == [("threads", 4), ("interop", 1)]


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


def test_geometry_profile_uses_44_physical_cores(config):
    profile = geometry_profile(config.parallelism)

    assert profile.processes == 44
    assert profile.native_threads == 1
    assert profile.processes * profile.native_threads == 44


@pytest.mark.parametrize("profile", [GeometryProfile(22, 2), GeometryProfile(44, 1)])
def test_geometry_affinity_sets_are_disjoint_and_cover_all_physical_cores(profile):
    affinity_sets = geometry_affinity_sets(profile)

    assert len(affinity_sets) == profile.processes
    assert all(len(value) == profile.native_threads for value in affinity_sets)
    assert all(
        set(left).isdisjoint(right)
        for index, left in enumerate(affinity_sets)
        for right in affinity_sets[index + 1 :]
    )
    assigned = set().union(*(set(value) for value in affinity_sets))
    assert len(assigned) == 44
    assert assigned == set(range(20)) | set(range(24, 48))
    assert assigned.isdisjoint(range(20, 24))
