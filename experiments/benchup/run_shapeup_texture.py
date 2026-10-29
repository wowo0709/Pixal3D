"""Texture ShapeUP geometry using its official source-view conditioned pipeline."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, types
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','manifest','geometry-root','source-root','base','vae','adapter','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu',required=True,choices=['0','1','2','3']);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--limit',type=int,default=0);p.add_argument('--geometry-producer-pid',type=int)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu or not 0<=a.rank<a.world_size:raise ValueError('explicit allowed GPU and shard required')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SHAPEUP_TEXTURE_CHECKPOINT=str(a.adapter))
    sys.path.insert(0,str(a.repo))
    for key in ('shapeup_geometry.data','shapeup_geometry.systems'):
        module=types.ModuleType(key);module.__path__=[str(a.repo.joinpath(*key.split('.')))];sys.modules[key]=module
    import torch,trimesh
    from shapeup_texture.pipelines.step1x_3d_texture_synthesis_shapeup_pipeline import Step1X3DTextureConfig,Step1X3DTexturePipeline
    torch.set_num_threads(2);torch.cuda.set_device(0)
    config=Step1X3DTextureConfig(cond_type='cross_attention');config.base_model=str(a.base);config.vae_model=str(a.vae)
    pipe=Step1X3DTexturePipeline(config)
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    if len(rows)!=109 or len({x['sample_id'] for x in rows})!=109:raise ValueError('frozen population required')
    rows=rows[a.rank::a.world_size]
    if a.limit:rows=rows[:a.limit]
    def ready_rows():
        pending=list(rows)
        while pending:
            ready=[r for r in pending if (a.geometry_root/'cases'/r['sample_id']/'receipt.json').is_file()]
            for r in ready:
                yield r;pending.remove(r)
            if pending and not ready:
                if not a.geometry_producer_pid:raise FileNotFoundError('geometry missing for '+pending[0]['sample_id'])
                try:os.kill(a.geometry_producer_pid,0)
                except ProcessLookupError:raise RuntimeError(f'geometry producer stopped with {len(pending)} missing')
                time.sleep(10)
    for row in ready_rows():
        sid=row['sample_id'];geo=a.geometry_root/'cases'/sid;grec=json.loads((geo/'receipt.json').read_text());mesh=geo/'geometry.obj';target=Path(row['target_anchor']);source=a.source_root/row['source_uid']/'source';out=a.output/'cases'/sid
        if grec['status']!='SHAPEUP_GEOMETRY_ONLY_NO_TEXTURE' or grec['geometry_sha256']!=sha(mesh) or grec['target_sha256']!=sha(target):raise ValueError('geometry input mismatch')
        views=sorted((source/'mv').glob('render_*.png'))[:6]
        if len(views)!=6:raise ValueError('six official source views required')
        if out.exists():
            receipt=out/'receipt.json';glb=out/'textured.glb'
            if not receipt.is_file() or not glb.is_file():raise FileExistsError(f'partial case: {out}')
            old=json.loads(receipt.read_text())
            if old.get('status')!='SHAPEUP_TEXTURED_GENERATED' or old['glb_sha256']!=sha(glb) or old['geometry_sha256']!=sha(mesh):raise ValueError('output mismatch')
            print(json.dumps({'status':'reused','sample_id':sid}),flush=True);continue
        # Published inputs already carry alpha, so segmentation is unnecessary.
        start=time.monotonic();m=trimesh.load(mesh)
        with torch.inference_mode():textured,*_=pipe(str(target),m,str(source),guidance_scale=6.5,remove_bg=False,seed=0,cfg_type='separate',guidance_scale_img=2.5,guidance_scale_src_mv=3.5)
        out.mkdir(parents=True);textured.export(out/'textured.glb')
        rec={'status':'SHAPEUP_TEXTURED_GENERATED','sample_id':sid,'geometry_sha256':sha(mesh),'target_sha256':sha(target),'source_view_sha256':[sha(x) for x in views],'glb_sha256':sha(out/'textured.glb'),'code_commit':'68aa30068c464803aa03fb1d93e9fd9960c14f5e','adapter_revision':'67fe1e2a477ffe87dad4ec3d067c9bb1c9e99b37','target_3D_GT_used':False,'seed':0,'seconds':round(time.monotonic()-start,2)}
        (out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)

if __name__=='__main__':main()
