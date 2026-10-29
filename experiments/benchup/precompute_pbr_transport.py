"""Build PBR DINO/NAF descriptors from images, without target 3D support."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
os.environ.setdefault('ATTN_BACKEND','sdpa')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import torch
from experiments.deformnext_retargeting.precompute_transport import load_rgba,pbr_pixal_patches,atomic_npz
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--dino',type=Path,required=True)
    p.add_argument('--gpu',choices=['0','1','2','3'],required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu or not 0<=a.rank<a.world_size:raise ValueError('GPU/shard mismatch')
    torch.set_num_threads(2);torch.cuda.set_device(0)
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    if len(rows)!=109:raise ValueError('complete input population required')
    extractor=DinoV3ProjFeatureExtractor(model_name=str(a.dino),image_size=1024,grid_resolution=64,use_naf_upsample=True,naf_target_size=1024).cuda().eval()
    selected=[r for i,r in enumerate(rows) if i%a.world_size==a.rank]
    if a.limit:selected=selected[:a.limit]
    for row in selected:
        output=Path(row['transport_cache_pbr']);shared=Path(row['transport_cache'])
        if output.exists():
            with np.load(output) as z:
                if str(z['conditioning_input_sha256'])!=row['conditioning_input_sha256'] or 'patch_ids' in z:raise ValueError('incompatible cache')
            continue
        with np.load(shared) as z:
            if str(z['conditioning_input_sha256'])!=row['conditioning_input_sha256']:raise ValueError('SS descriptor input mismatch')
        with torch.inference_mode():
            images=torch.stack([load_rgba(Path(path),1024)[0] for path in [row['target_anchor'],*row['source_images']]]).cuda()
            values,query=pbr_pixal_patches(extractor,images)
            atomic_npz(output,schema=np.asarray('DEFORMNEXT_PIXAL_VALUES_PBR1024_V1'),conditioning_input_sha256=np.asarray(row['conditioning_input_sha256']),source_values=values[1:].cpu().half().numpy(),target_query=query[0].cpu().half().numpy())
        print(json.dumps({'status':'PBR_DESCRIPTOR_NO_TARGET_3D_GT','sample_id':row['sample_id']}),flush=True)

if __name__=='__main__':main()
