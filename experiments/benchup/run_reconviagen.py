"""Run pinned official ReconViaGen v0.5 with target RGB plus three source RGBs."""
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
    for flag in ('manifest','repo','ss-model','trellis2-model','output'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--gpu',choices=['0','1','2','3'],required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--limit',type=int,default=0);p.add_argument('--generated-views-root',type=Path);p.add_argument('--views-producer-pid',type=int)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu or not 0<=a.rank<a.world_size:raise ValueError('explicit allowed GPU and shard required')
    # Implicit GEMM in spconv/cumm can SIGFPE on SM120. Match the pinned
    # native convolution backend used by the direct ReconViaGen baseline.
    os.environ.update(ATTN_BACKEND='sdpa',SPARSE_ATTN_BACKEND='xformers',SPCONV_ALGO='native',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.path[:0]=[str(a.repo),str(a.repo/'wheels/vggt'),str(a.repo/'wheels/TRELLIS.2')]
    import torch,numpy as np
    from PIL import Image
    torch.set_num_threads(2);torch.cuda.set_device(0)
    # xFormers defaults to a Hopper flash kernel which fails on this SM120
    # GPU. The same attention math through CUTLASS is supported here.
    import xformers.ops as xops
    xformers_attention=xops.memory_efficient_attention
    def compatible_attention(*args,**kwargs):
        kwargs['op']=(xops.fmha.cutlass.FwOp,None)
        return xformers_attention(*args,**kwargs)
    xops.memory_efficient_attention=compatible_attention
    from trellis.pipelines.trellis_image_to_3d import TrellisVGGTTo3DPipeline
    from trellis.pipelines.trellis_hybrid_pipeline import TrellisHybridPipeline
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from unittest.mock import patch
    from transformers import AutoModelForImageSegmentation,DINOv3ViTModel
    biref=AutoModelForImageSegmentation.from_pretrained;dino=DINOv3ViTModel.from_pretrained
    local_biref='/root/.cache/huggingface/hub/models--ZhengPeng7--BiRefNet/snapshots/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4'
    local_dino='/root/.cache/huggingface/hub/models--camenduru--dinov3-vitl16-pretrain-lvd1689m/snapshots/3c276edd87d6f6e569ff0c4400e086807d0f3881'
    def biref_local(name,*args,**kwargs):return biref(local_biref if name=='ZhengPeng7/BiRefNet' else name,*args,**kwargs)
    def dino_local(name,*args,**kwargs):return dino(local_dino if name=='facebook/dinov3-vitl16-pretrain-lvd1689m' else name,*args,**kwargs)
    with patch.object(AutoModelForImageSegmentation,'from_pretrained',side_effect=biref_local),patch.object(DINOv3ViTModel,'from_pretrained',side_effect=dino_local):
        vp=TrellisVGGTTo3DPipeline.from_pretrained(str(a.ss_model));vp.cuda();vp.VGGT_model.cuda();vp.birefnet_model.cuda()
        del vp.models['slat_decoder_gs']
        tp=Trellis2ImageTo3DPipeline.from_pretrained(str(a.trellis2_model));tp.cuda()
    original_sample_slat=vp.sample_slat
    def checked_sample_slat(cond,coords,*args,**kwargs):
        print(json.dumps({'status':'RECON_SPARSE_STRUCTURE_DIAGNOSTIC','coordinates':int(coords.shape[0])}),flush=True)
        if coords.shape[0]==0:raise ValueError('ReconViaGen sparse structure is empty; record as failed generation')
        return original_sample_slat(cond,coords,*args,**kwargs)
    vp.sample_slat=checked_sample_slat
    pipeline=TrellisHybridPipeline(vp,tp,low_vram=False)
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    if len(rows)!=109:raise ValueError('complete BenchUp population required')
    selected=[r for i,r in enumerate(rows) if i%a.world_size==a.rank]
    if a.limit:selected=selected[:a.limit]
    a.output.mkdir(parents=True,exist_ok=True)
    ss={'steps':12,'cfg_strength':7.5,'cfg_interval':[.6,1.],'guidance_rescale':.7,'rescale_t':5.}
    slat={'steps':12,'cfg_strength':7.5,'cfg_interval':[.6,1.],'guidance_rescale':.5,'rescale_t':3.}
    shape={'steps':12,'guidance_strength':7.5,'guidance_rescale':.5,'rescale_t':3.}
    tex={'steps':12,'guidance_strength':1.,'guidance_rescale':0.,'rescale_t':3.}
    if a.views_producer_pid and not a.generated_views_root:raise ValueError('views producer requires generated views')
    def ready_rows():
        pending=list(selected)
        while pending:
            ready=[r for r in pending if not a.generated_views_root or (a.generated_views_root/'cases'/r['sample_id']/'receipt.json').is_file()]
            for r in ready:
                yield r;pending.remove(r)
            if pending and not ready:
                if not a.views_producer_pid:raise FileNotFoundError('generated views missing for '+pending[0]['sample_id'])
                try:os.kill(a.views_producer_pid,0)
                except ProcessLookupError:raise RuntimeError(f'generated views producer stopped with {len(pending)} missing')
                time.sleep(10)
    for row in ready_rows():
        sid=row['sample_id'];out=a.output/'cases'/sid
        if a.generated_views_root:
            view_dir=a.generated_views_root/'cases'/sid;view_receipt=json.loads((view_dir/'receipt.json').read_text())
            if view_receipt.get('status') not in {'MVADAPTER_TARGET_ONLY_SIX_GENERATED_VIEWS','CORRADAPTER_TARGET_ONLY_SIX_GENERATED_VIEWS'} or view_receipt.get('target_sha256')!=sha(row['target_anchor']) or len(view_receipt['views'])!=6:raise ValueError('generated-view contract mismatch')
            paths=[view_dir/x['name'] for x in view_receipt['views']]
            if any(sha(path)!=entry['sha256'] for path,entry in zip(paths,view_receipt['views'])):raise ValueError('generated view hash mismatch')
            input_kind=view_receipt['status']
        else:
            paths=[Path(row['target_anchor']),*[Path(x) for x in row['source_images']]];input_kind='edited target RGB + three original source RGB'
        if out.exists():
            receipt=out/'receipt.json';prediction=out/'prediction.npz'
            if not receipt.is_file() or not prediction.is_file():raise FileExistsError(f'partial case needs inspection: {out}')
            old=json.loads(receipt.read_text())
            if old.get('status')!='RECONVIAGEN_V05_GENERATED_RAW' or old.get('prediction_sha256')!=sha(prediction) or old.get('input')!=input_kind or old.get('input_sha256')!=[sha(x) for x in paths]:raise ValueError(f'existing output mismatch: {out}')
            print(json.dumps({'status':'reused','sample_id':sid}),flush=True);continue
        images=[Image.open(path).convert('RGBA') for path in paths];start=time.monotonic()
        with torch.inference_mode():
            meshes,(_,_,resolution)=pipeline.run_multi_image(images,strategy='adaptive_guidance_weight',seed=0,ss_sampler_params=ss,slat_sampler_params=slat,shape_slat_sampler_params=shape,tex_slat_sampler_params=tex,pipeline_type='1024_cascade',preprocess_image=True,return_latent=True,ss_source='mesh',max_num_tokens=32768)
        if len(meshes)!=1 or resolution!=1024:raise ValueError('unexpected output')
        m=meshes[0];out.mkdir(parents=True)
        np.savez_compressed(out/'prediction.npz',vertices=m.vertices.cpu().numpy(),faces=m.faces.cpu().numpy(),coords=m.coords.cpu().numpy(),attrs=m.attrs.cpu().numpy())
        rec={'status':'RECONVIAGEN_V05_GENERATED_RAW','sample_id':sid,'prediction_sha256':sha(out/'prediction.npz'),'input_sha256':[sha(x) for x in paths],'code_commit':'19a902f','ss_model_revision':'647659a5ad5fbf67e22793e7b5e2cee4b30c5d13','input':input_kind,'camera_used':False,'target_mesh_used':False,'resolution':1024,'seed':0,'ss_source':'mesh','fusion':'adaptive_guidance_weight','xformers_forward_kernel':'cutlass','spconv_algo':'native','seconds':round(time.monotonic()-start,2)}
        (out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)

if __name__=='__main__':main()
