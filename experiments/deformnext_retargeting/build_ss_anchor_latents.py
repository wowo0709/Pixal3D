"""Encode SS64 from canonical M1 Shape1024 support, preserving old labels."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from pixal3d import models

SCHEMA = 'deformnext-ss64-m1-anchor-v3'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--manifest-root', type=Path, required=True)
    parser.add_argument('--published-root', required=True)
    parser.add_argument('--path-map', action='append', default=[])
    parser.add_argument('--batch-size', type=int, default=32)
    args=parser.parse_args()
    torch.set_num_threads(4)
    mappings=[tuple(v.split('=',1)) for v in args.path_map]
    def remap(value):
        for source,dest in mappings:
            if value == source or value.startswith(source.rstrip('/')+'/'):
                return Path(dest+value[len(source):])
        return Path(value)
    encoder=models.from_pretrained('microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16').cuda().eval()
    args.output_root.mkdir(parents=True,exist_ok=True)
    args.manifest_root.mkdir(parents=True,exist_ok=True)
    summary={'schema':SCHEMA,'splits':{}}
    for manifest in args.manifest:
        rows=[json.loads(line) for line in manifest.read_text().splitlines() if line]
        pending=[]
        def flush():
            if not pending:return
            occupancy=torch.zeros(len(pending),1,64,64,64,device='cuda')
            for index,(row,coords,digest) in enumerate(pending):
                coords=torch.from_numpy(coords).cuda().long()
                occupancy[index,0,coords[:,0],coords[:,1],coords[:,2]]=1
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                latent=encoder(occupancy,sample_posterior=False).float().cpu().numpy()
            if latent.shape[1:] != (8,16,16,16) or not np.isfinite(latent).all():
                raise ValueError('invalid SS encoder output')
            for value,(row,coords,digest) in zip(latent,pending):
                out=args.output_root/(row['sample_id']+'.npz');tmp=out.with_name(f'.{out.name}.{os.getpid()}.npz')
                np.savez(tmp,z=value,schema=np.asarray(SCHEMA),source_coords_sha256=np.asarray(digest))
                os.replace(tmp,out)
            pending.clear()
        processed=0
        for row in rows:
            with np.load(remap(row['shape_latent'])) as archive:
                coords=np.asarray(archive['coords']).copy()
            if coords.ndim!=2 or coords.shape[1]!=3 or np.any((coords<0)|(coords>=64)):
                raise ValueError(f'invalid canonical M1 support: {row["sample_id"]}')
            digest=hashlib.sha256(coords.tobytes()).hexdigest()
            out=args.output_root/(row['sample_id']+'.npz')
            valid=False
            if out.exists():
                with np.load(out) as z:
                    valid=('schema' in z and str(z['schema'])==SCHEMA and
                           str(z['source_coords_sha256'])==digest and z['z'].shape==(8,16,16,16))
            if not valid:pending.append((row,coords,digest))
            if len(pending)>=args.batch_size:flush()
            row['ss_latent_previous']=row['ss_latent']
            row['ss_latent']=str(Path(args.published_root)/(row['sample_id']+'.npz'))
            row['ss_latent_schema']=SCHEMA
            processed+=1
            if processed%256==0:print(json.dumps({'split':manifest.stem,'processed':processed,'total':len(rows)}),flush=True)
        flush()
        dest=args.manifest_root/manifest.name;tmp=dest.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(row)+'\n' for row in rows));os.replace(tmp,dest)
        summary['splits'][manifest.stem]=len(rows)
    (args.manifest_root/'ready.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
