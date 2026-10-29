"""Official single-image Pixal3D with estimated camera and unit default scale."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('manifest','model','output'):p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--gpu',choices=['0','1','2','3'],required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--resolution',type=int,choices=[1024,1536],default=1024);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu or not 0<=a.rank<a.world_size:raise ValueError('assigned GPU required')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',ATTN_BACKEND='sdpa')
    import torch,numpy as np
    from PIL import Image
    import inference
    from pixal3d.pipelines import pixal3d_image_to_3d as module
    from experiments.deformnext_retargeting.cascade_artifact_inputs import prepared_views
    torch.set_num_threads(2);torch.cuda.set_device(0)
    pipeline=inference.init_pipeline(str(a.model),low_vram=True);module.pil_views_to_tensor=prepared_views
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    if len(rows)!=109:raise ValueError('frozen 109 inputs required')
    selected=[r for i,r in enumerate(rows) if i%a.world_size==a.rank]
    if a.limit:selected=selected[:a.limit]
    for row in selected:
        sid=row['sample_id'];out=a.output/'cases'/sid
        if out.exists():
            receipt=out/'receipt.json'; prediction=out/'prediction.npz'
            if not receipt.is_file() or not prediction.is_file():raise FileExistsError(f'partial case needs inspection: {out}')
            previous=json.loads(receipt.read_text())
            if previous.get('status')!=f'OFFICIAL_PIXAL3D_{a.resolution}_ESTIMATED_CAMERA_UNCROPPED' or previous.get('prediction_sha256')!=sha(prediction) or previous.get('target_sha256')!=sha(row['target_anchor']):raise ValueError(f'existing output differs: {out}')
            print(json.dumps({'status':'reused','sample_id':sid}),flush=True);continue
        image=Image.open(row['target_anchor']).convert('RGBA');camera={'camera_angle_x':row['camera_angle_x'][0],'distance':float(np.linalg.norm(np.asarray(row['transform_matrix'][0])[:3,3]))}
        start=time.monotonic()
        with torch.inference_mode():
            meshes,(_,_,resolution)=pipeline.run(image,camera_params=camera,seed=0,sparse_structure_sampler_params={'steps':12,'guidance_strength':7.5,'guidance_rescale':.7,'rescale_t':5.},shape_slat_sampler_params={'steps':12,'guidance_strength':7.5,'guidance_rescale':.5,'rescale_t':3.},tex_slat_sampler_params={'steps':12,'guidance_strength':1.,'guidance_rescale':0.,'rescale_t':3.},preprocess_image=False,return_latent=True,pipeline_type=f'{a.resolution}_cascade',max_num_tokens=32768)
        if len(meshes)!=1 or resolution not in range(1024,a.resolution+1,128):raise ValueError('unexpected cascade output resolution')
        m=meshes[0];out.mkdir(parents=True);np.savez_compressed(out/'prediction.npz',vertices=m.vertices.cpu().numpy(),faces=m.faces.cpu().numpy(),attrs=m.attrs.cpu().numpy(),coords=m.coords.cpu().numpy())
        rec={'status':f'OFFICIAL_PIXAL3D_{a.resolution}_ESTIMATED_CAMERA_UNCROPPED','sample_id':sid,'prediction_sha256':sha(out/'prediction.npz'),'target_sha256':sha(row['target_anchor']),'model_revision':'0b31f9160aa400719af409098bff7936a932f726','requested_resolution':a.resolution,'resolution':resolution,'camera':'source_mesh_source_image_estimated','camera_params':camera,'mesh_scale_effective':1.,'target_GT_used':False,'preprocessing':'uncropped_RGBA_alpha_composite_for_camera_consistency','seed':0,'seconds':round(time.monotonic()-start,2)}
        (out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)

if __name__=='__main__':main()
