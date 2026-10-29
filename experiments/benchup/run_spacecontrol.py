"""SpaceControl with the raw target-only TRELLIS.2 prediction as spatial control."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','manifest','proxy-root','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu',required=True,choices=['0','1','2','3']);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu or not 0<=a.rank<a.world_size:raise ValueError('explicit allowed GPU and shard required')
    os.environ.update(ATTN_BACKEND='xformers',SPCONV_ALGO='native',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0,str(a.repo));os.chdir(a.repo)
    import numpy as np,torch,open3d as o3d
    from PIL import Image
    import xformers.ops as xops
    attention=xops.memory_efficient_attention
    def compatible_attention(*args,**kwargs):
        kwargs['op']=(xops.fmha.cutlass.FwOp,None);return attention(*args,**kwargs)
    xops.memory_efficient_attention=compatible_attention
    from trellis.pipelines import TrellisTextTo3DPipeline
    from trellis.utils import postprocessing_utils
    torch.set_num_threads(2);torch.cuda.set_device(0)
    # The pinned CLIP snapshot was downloaded by revision and has no mutable
    # `main` ref in the offline cache. Resolve that same official checkpoint
    # locally instead of letting transformers query a moving branch.
    from unittest.mock import patch
    from transformers import CLIPTextModel,AutoTokenizer
    clip='/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
    text_loader=CLIPTextModel.from_pretrained;tokenizer_loader=AutoTokenizer.from_pretrained
    def local_text(name,*args,**kwargs):return text_loader(clip if name=='openai/clip-vit-large-patch14' else name,*args,**kwargs)
    def local_tokenizer(name,*args,**kwargs):return tokenizer_loader(clip if name=='openai/clip-vit-large-patch14' else name,*args,**kwargs)
    with patch.object(CLIPTextModel,'from_pretrained',side_effect=local_text),patch.object(AutoTokenizer,'from_pretrained',side_effect=local_tokenizer):
        pipeline=TrellisTextTo3DPipeline.from_pretrained(str(a.repo/'gui'));pipeline.cuda()
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    if len(rows)!=109 or len({r['sample_id'] for r in rows})!=109:raise ValueError('frozen population required')
    rows=rows[a.rank::a.world_size]
    if a.limit:rows=rows[:a.limit]
    for row in rows:
        sid=row['sample_id'];proxy=a.proxy_root/'cases'/sid;prediction=proxy/'prediction.npz';prec=json.loads((proxy/'receipt.json').read_text());source=Path(row['source_anchor_rgb']);out=a.output/'cases'/sid
        if prec['status']!='PASS_TRELLIS2_GENERATED_RAW' or prec['prediction_sha256']!=sha(prediction) or prec['target_sha256']!=sha(row['target_anchor']):raise ValueError('proxy input mismatch')
        if out.exists():
            receipt=out/'receipt.json';glb_path=out/'textured.glb'
            if not receipt.is_file() or not glb_path.is_file():raise FileExistsError(f'partial case needs inspection: {out}')
            old=json.loads(receipt.read_text())
            if old.get('status')!='SPACECONTROL_TARGET_PROXY_GENERATED' or old.get('proxy_prediction_sha256')!=sha(prediction) or old.get('source_anchor_sha256')!=sha(source) or old.get('glb_sha256')!=sha(glb_path):raise ValueError(f'existing output differs: {out}')
            print(json.dumps({'status':'reused','sample_id':sid}),flush=True);continue
        start=time.monotonic()
        with np.load(prediction) as z:vertices=z['vertices'].astype(np.float64);faces=z['faces'].astype(np.int32)
        bounds=np.stack([vertices.min(0),vertices.max(0)]);center=bounds.mean(0);scale=1./np.max(bounds[1]-bounds[0]);vertices=(vertices-center)*scale
        if not np.isfinite(vertices).all() or len(faces)==0:raise ValueError('invalid proxy')
        out.mkdir(parents=True);mesh=o3d.geometry.TriangleMesh();mesh.vertices=o3d.utility.Vector3dVector(vertices);mesh.triangles=o3d.utility.Vector3iVector(faces)
        proxy_path=out/'target-proxy-raw-normalized.ply'
        if not o3d.io.write_triangle_mesh(str(proxy_path),mesh,write_ascii=False):raise RuntimeError('proxy export failed')
        del vertices,faces,mesh
        category=row['source_uid'].replace('_',' ');image=Image.open(source).convert('RGBA')
        # The official sampler is decorated with no_grad internally. Its GLB
        # baker subsequently optimizes the texture and requires autograd.
        result=pipeline.run('a '+category,image,seed=0,formats=['mesh','gaussian'],sparse_structure_sampler_params={'steps':12,'cfg_strength':7.5,'t0_idx_value':6,'spatial_control_mesh_path':str(proxy_path)})
        glb=postprocessing_utils.to_glb(result['gaussian'][0],result['mesh'][0],simplify=.95,texture_size=1024)
        glb.apply_scale(1./scale);glb.apply_translation(center);glb.export(out/'textured.glb')
        rec={'status':'SPACECONTROL_TARGET_PROXY_GENERATED','sample_id':sid,'proxy_prediction_sha256':sha(prediction),'source_anchor_sha256':sha(source),'glb_sha256':sha(out/'textured.glb'),'proxy_normalization_center':center.tolist(),'proxy_normalization_scale':scale,'control_source':'raw target-only TRELLIS.2 mesh, no target GT','appearance_source':'original source anchor RGB','prompt':'a '+category,'control_step_index':6,'seed':0,'code_commit':'7ae1d921504536cf9197ad3841f7c477c4db97c7','seconds':round(time.monotonic()-start,2)}
        (out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n');proxy_path.unlink();print(json.dumps(rec),flush=True)

if __name__=='__main__':main()
