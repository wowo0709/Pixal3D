"""Cache B1 FRAME RAW Top-1 without a teacher or learned matching head."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from pixal3d.modules.r9_matching import RAW_MATCH_SCHEMA, RAW_MATCH_CONTRACT, frame_raw, frame_topk
from .precompute_matches import remap


def valid_cache(path, sample_id):
    if not path.is_file(): return False
    try:
        with np.load(path) as cache:
            return (str(cache['schema']) == RAW_MATCH_SCHEMA
                    and json.loads(str(cache['raw_matching_contract'])) == RAW_MATCH_CONTRACT
                    and str(cache['sample_id']) == sample_id
                    and cache['semantic_indices'].shape == (1024, 1)
                    and (cache['semantic_indices'] >= 0).all()
                    and (cache['semantic_indices'] < 3072).all()
                    and len(str(cache['descriptor_input_sha256'])) == 64)
    except (KeyError, ValueError, OSError):
        return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--path-map',action='append',default=[])
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world-size',type=int,default=1)
    args=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(0)
    maps=[tuple(x.split('=',1)) for x in args.path_map]
    rows=[json.loads(line) for line in args.manifest.read_text().splitlines() if line][args.rank::args.world_size]
    args.output_root.mkdir(parents=True,exist_ok=True)
    completed=skipped=0
    with torch.no_grad():
        for row in rows:
            output=args.output_root/f"{row['sample_id']}.npz"
            if valid_cache(output,row['sample_id']):skipped+=1;continue
            with np.load(remap(row['transport_cache'],maps)) as cache:
                if int(cache['omega_completed_updates']) != RAW_MATCH_CONTRACT['omega_completed_updates']:
                    raise ValueError('RAW cache requires the approved epoch-7 Omega descriptor')
                target=cache['target_keys'];source=cache['source_keys'];valid=cache['source_valid']
            digest=hashlib.sha256()
            for array in (target,source,valid):digest.update(array.tobytes())
            target=frame_raw(torch.from_numpy(target).cuda().float())
            source=frame_raw(torch.from_numpy(source).cuda().float())
            _,indices=frame_topk(target[None],source[None],torch.from_numpy(valid).cuda().bool()[None],1)
            tmp=output.with_name(f'.{output.name}.{os.getpid()}.tmp.npz')
            np.savez(tmp,schema=np.asarray(RAW_MATCH_SCHEMA),
                raw_matching_contract=np.asarray(json.dumps(RAW_MATCH_CONTRACT,sort_keys=True)),
                sample_id=np.asarray(row['sample_id']),split=np.asarray(row['split']),
                descriptor_input_sha256=np.asarray(digest.hexdigest()),
                semantic_indices=indices[0].cpu().numpy().astype(np.uint16))
            if not valid_cache(tmp,row['sample_id']):raise ValueError('invalid RAW Top-1 cache')
            os.replace(tmp,output);completed+=1
            if (completed+skipped)%50==0:
                print(json.dumps(dict(rank=args.rank,completed=completed,skipped=skipped,assigned=len(rows))),flush=True)
    print(json.dumps(dict(rank=args.rank,completed=completed,skipped=skipped,assigned=len(rows),status='PASS')),flush=True)


if __name__=='__main__':main()
