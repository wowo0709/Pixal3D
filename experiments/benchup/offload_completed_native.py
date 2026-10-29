"""Retire only locally duplicated, processed native predictions with verified archives."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from .run_v1_cascade import sha


def live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--post-root', type=Path, required=True)
    p.add_argument('--method', choices=['trellis2', 'recon', 'official-rgb1', 'mvadapter-recon', 'corradapter-recon'], required=True)
    p.add_argument('--post-pids', required=True)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    pids = [int(x) for x in a.post_pids.split(',')]
    if not pids or any(x <= 0 for x in pids):
        raise ValueError('explicit postprocessor handles required')
    complete = json.loads((a.run_root / 'archive-receipts/complete.json').read_text())
    if complete != {'status': 'PASS_ALL109_FILE4_ARCHIVED', 'samples': 109}:
        raise ValueError('raw archive must be complete before retirement')
    done = set()
    while True:
        markers = sorted(a.post_root.glob(f'{a.method}-postprocess-shard-*/cases/*/archive.json'))
        for marker in markers:
            sid = marker.parent.name
            if sid in done:
                continue
            case = a.run_root / 'cases' / sid
            source, receipt, archive = case/'prediction.npz', case/'receipt.json', a.run_root/'archive-receipts'/f'{sid}.json'
            if not source.exists() and (case/'offload.json').exists():
                done.add(sid); continue
            if not source.exists() or not receipt.exists() or not archive.exists():
                continue
            raw, archived, processed = (json.loads(x.read_text()) for x in (receipt, archive, marker))
            if archived['status'] != 'PASS_FILE4_SHA256' or processed['status'] != 'PASS_FILE4_SHA256' or archived['sha256'] != raw['prediction_sha256'] or sha(source) != archived['sha256']:
                raise ValueError(f'archive or local hash mismatch: {sid}')
            payload = {'status':'LOCAL_DUPLICATE_RETIRED_VERIFIED_FILE4_RETAINED', 'sample_id':sid, 'sha256':archived['sha256'], 'remote':archived['remote']+'/prediction.npz'}
            if a.apply:
                temp = case/'offload.tmp'; temp.write_text(json.dumps(payload, indent=2)+'\n'); os.replace(temp, case/'offload.json'); source.unlink()
            done.add(sid)
            print(json.dumps({'status':'RAW_OFFLOADED' if a.apply else 'DRY_RUN', 'method':a.method, 'sample_id':sid, 'count':len(done)}), flush=True)
        if not a.apply or not any(live(pid) for pid in pids):
            break
        time.sleep(20)


if __name__ == '__main__':
    main()
