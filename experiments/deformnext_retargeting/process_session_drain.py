"""Keep a GPU reservation until an owned process session has no live children."""
import os
import time

import psutil


def owned_session_processes(session_id, started_at):
    if type(session_id)is not int or session_id<=0 or started_at<=0:
        raise ValueError('actual positive owned session and start time required')
    result=[]
    for proc in psutil.process_iter(['pid','create_time','status']):
        try:
            row=proc.info
            if (row['create_time']>=started_at-.01 and row['status']not in {psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD}
                    and os.getsid(row['pid'])==session_id):
                result.append(dict(PID=row['pid'],birth=row['create_time'],status=row['status']))
        except (psutil.NoSuchProcess,psutil.AccessDenied,ProcessLookupError,PermissionError):
            continue
    return result


def drain_owned_session(session_id, started_at, *, progress=None, interval=1.):
    if not 0<interval<=5:raise ValueError('short bounded polling interval required')
    observations=[]
    while True:
        rows=owned_session_processes(session_id,started_at)
        if not rows:return observations
        observations.append(rows)
        if progress is not None:
            try:progress(rows)
            except Exception:pass  # Failed archive/disk IO cannot release live CUDA children.
        time.sleep(interval)
