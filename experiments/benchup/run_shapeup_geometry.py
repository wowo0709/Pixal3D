"""Run official ShapeUP geometry on the published source point cloud and edit."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,sys,time,types
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('repo','manifest','source-root','base','adapter','dino','clip','output'):p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--gpu',required=True,choices=['0','1','2','3']);p.add_argument('--limit',type=int,default=0);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu or not 0<=a.rank<a.world_size:raise ValueError('valid physical GPU/shard required')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0,str(a.repo))
    # Avoid loading unrelated training dataset/system modules; inference model
    # registration and all model/sampler code remain the official implementation.
    for key in ('shapeup_geometry.data','shapeup_geometry.systems'):
        module=types.ModuleType(key);module.__path__=[str(a.repo.joinpath(*key.split('.')))];sys.modules[key]=module
    import numpy as np,torch
    from PIL import Image
    from shapeup_geometry.geometry_spec import GEOMETRY_SPEC
    from shapeup_geometry.models.pipelines.pipeline import ShapeUpGeometryPipeline
    torch.set_num_threads(2);torch.cuda.set_device(0)
    spec=copy.deepcopy(GEOMETRY_SPEC)
    for key in ('shape_model','denoiser_model'):spec[key]['pretrained_model_name_or_path']=str(a.base)
    spec['visual_condition']['pretrained_dino_name_or_path']=str(a.dino);spec['visual_condition']['pretrained_clip_name_or_path']=str(a.clip)
    pipe=ShapeUpGeometryPipeline.from_spec(spec,adapter_path=str(a.adapter),device='cuda',dtype=torch.float32)
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    if len(rows)!=109:raise ValueError('complete frozen BenchUp population required')
    rows=[r for i,r in enumerate(rows) if i%a.world_size==a.rank]
    if a.limit:rows=rows[:a.limit]
    a.output.mkdir(parents=True,exist_ok=True)
    for row in rows:
        sid=row['sample_id'];source=a.source_root/row['source_uid']/'source/pc.npz';target=Path(row['target_anchor']);out=a.output/'cases'/sid
        if out.exists():
            receipt=out/'receipt.json';geometry=out/'geometry.obj'
            if not receipt.is_file() or not geometry.is_file():raise FileExistsError(f'partial case needs inspection: {out}')
            previous=json.loads(receipt.read_text())
            if previous.get('status')!='SHAPEUP_GEOMETRY_ONLY_NO_TEXTURE' or previous.get('source_pointcloud_sha256')!=sha(source) or previous.get('target_sha256')!=sha(target) or previous.get('geometry_sha256')!=sha(geometry):raise ValueError(f'existing output differs: {out}')
            print(json.dumps({'status':'reused','sample_id':sid}),flush=True);continue
        with np.load(source) as pc:
            batch={'surface':torch.from_numpy(pc['fps_coarse_surface'][:,0]).float().unsqueeze(0),'sharp_surface':torch.from_numpy(pc['fps_sharp_surface'][:,0]).float().unsqueeze(0),'image':[Image.open(target).convert('RGBA')]}
        start=time.monotonic()
        with torch.inference_mode():
            result=pipe(batch,seed=0,num_inference_steps=30,octree_resolution=256,cfg_type='separate_visual',guidance_scale_visual=3.5,guidance_scale_shape=2.5,match_reference=False,do_remove_floater=False,do_reduce_face=False,do_shade_smooth=False,output_type='trimesh')
        mesh=result.mesh[0]
        if mesh is None or not len(mesh.faces):raise ValueError('empty geometry')
        out.mkdir(parents=True);mesh.export(out/'geometry.obj')
        receipt={'status':'SHAPEUP_GEOMETRY_ONLY_NO_TEXTURE','sample_id':sid,'source_pointcloud_sha256':sha(source),'target_sha256':sha(target),'geometry_sha256':sha(out/'geometry.obj'),'code_commit':'68aa30068c464803aa03fb1d93e9fd9960c14f5e','adapter_revision':'67fe1e2a477ffe87dad4ec3d067c9bb1c9e99b37','target_3D_GT_used':False,'seed':0,'seconds':round(time.monotonic()-start,2),'texture_stage_pending':True}
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
