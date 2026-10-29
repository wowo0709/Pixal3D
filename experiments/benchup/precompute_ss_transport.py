"""Build Omega and Pixal SS inference descriptors without target depth or geometry."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
os.environ.setdefault('ATTN_BACKEND','sdpa')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import torch
from experiments.deformnext_retargeting.omega_loader import load_epoch7,layer12_patch_tokens
from experiments.deformnext_retargeting.precompute_transport import load_rgba,pixal_patches,patch_visibility,ss_patch_grid,atomic_npz
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest','vggt-root','base-checkpoint','epoch7-checkpoint','dino'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--gpu',required=True,choices=['0','1','2','3'])
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu or not 0<=a.rank<a.world_size:raise ValueError('assigned GPU/shard required')
    torch.set_num_threads(2);torch.cuda.set_device(0)
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    if len(rows)!=109 or any('mesh_scale' not in r for r in rows):raise ValueError('full image-estimated input manifest required')
    omega,checkpoint=load_epoch7(a.vggt_root,a.base_checkpoint,a.epoch7_checkpoint)
    dino=DinoV3ProjFeatureExtractor(model_name=str(a.dino),image_size=512,grid_resolution=16).cuda().eval()
    selected=[row for i,row in enumerate(rows) if i%a.world_size==a.rank]
    if a.limit:selected=selected[:a.limit]
    for row in selected:
        output=Path(row['transport_cache'])
        if output.exists():
            with np.load(output) as z:
                if str(z['conditioning_input_sha256'])!=row['conditioning_input_sha256'] or 'spatial_indices' in z:raise ValueError('existing cache differs')
            continue
        paths=[row['target_anchor'],*row['source_images']]
        with torch.inference_mode():
            images=[load_rgba(Path(x),512) for x in paths]
            rgb=torch.stack([x[0] for x in images]).cuda();alpha=torch.stack([x[1] for x in images]).cuda()
            omega_tokens=layer12_patch_tokens(omega,rgb[None])[0]; dino_tokens=pixal_patches(dino,rgb)
            atomic_npz(output,schema=np.asarray('DEFORMNEXT_OMEGA_L12_PIXAL_VALUES_SS64_V1'),
                conditioning_input_sha256=np.asarray(row['conditioning_input_sha256']),omega_completed_updates=np.int64(checkpoint['completed_updates']),
                target_keys=omega_tokens[0].cpu().half().numpy(),source_keys=omega_tokens[1:].cpu().half().numpy(),
                source_values=dino_tokens[1:].cpu().half().numpy(),source_valid=patch_visibility(alpha[1:]).cpu().numpy(),
                target_query=dino_tokens[0].cpu().half().numpy(),patch_grid=ss_patch_grid(row).numpy())
        print(json.dumps({'status':'SS_TRANSPORT_NO_GT_DEPTH','sample_id':row['sample_id'],'output':str(output)}),flush=True)

if __name__=='__main__':main()
