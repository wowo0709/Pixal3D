"""Wait for inspectable process metadata; never infer absence from AccessDenied."""
from contextlib import contextmanager
import time

import psutil


def wait_for_access(process, reader, *, method, emit=lambda event: None):
    """Return a real read, or propagate verified process exit/PID replacement.

    A protected live process remains an unknown reader. There is no timeout,
    empty substitute, permission change or signal to another process.
    """
    birth = process.create_time()
    began = None

    def finish(resolution):
        if began is not None:
            emit(dict(event='process-access-resolved', PID=process.pid, birth=birth,
                      method=method, resolution=resolution,
                      waited_seconds=time.monotonic() - began))

    while True:
        try:
            current = psutil.Process(process.pid)
            if current.create_time() != birth:
                raise psutil.NoSuchProcess(process.pid, msg='original process identity exited')
            if current.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                raise psutil.ZombieProcess(process.pid)
            result = reader(process)
        except psutil.AccessDenied:
            if began is None:
                began = time.monotonic()
                emit(dict(event='waiting-for-process-access', PID=process.pid,
                          birth=birth, method=method))
            time.sleep(.05)
            continue
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            finish('original-process-exited')
            raise
        finish('metadata-readable')
        return result


@contextmanager
def process_access_waits(*, emit=lambda event: None):
    """Scope the wait to operational file/environment inspection only."""
    originals = {name: getattr(psutil.Process, name) for name in ('open_files', 'environ')}
    def wrap(name, original):
        def read(process):
            return wait_for_access(process, original, method=name, emit=emit)
        return read
    try:
        for name, original in originals.items():
            setattr(psutil.Process, name, wrap(name, original))
        yield
    finally:
        for name, original in originals.items():
            setattr(psutil.Process, name, original)
