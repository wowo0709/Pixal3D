"""Run a diagnostic single-view 1024 Pixal3D cascade with a source-estimated camera."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=Path,required=True);p.add_argument('--camera-job',type=Path,required=True);p.add_argument('--camera-job-index',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--physical-gpu',type=int,choices=(0,1,2,3),required=True);p.add_argument('--pixal3d-repo',type=Path,required=True);p.add_argument('--native-setup',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    jobs=[json.loads(x) for x in a.camera_job.read_text().splitlines()];j=jobs[a.camera_job_index]
    camera=j['camera'];mat=camera['camera_to_world'];distance=sum(float(mat[i][3])**2 for i in range(3))**.5
    params={'camera_angle_x':float(camera['camera_angle_x']),'distance':distance}
    args=SimpleNamespace(output_root=a.output,rank=0,physical_gpu=a.physical_gpu,attention_backend='sdpa',native_setup=a.native_setup,pixal3d_repo=a.pixal3d_repo)
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.physical_gpu)
    from experiments.deformnext_retargeting.run_official_pixal3d_scale_ungiven1024 import load_pipeline
    import numpy as np
    from PIL import Image
    torch,pipeline=load_pipeline(args);image=pipeline.preprocess_image(Image.open(a.target));torch.manual_seed(0)
    with torch.inference_mode():
        meshes,(_,_,resolution)=pipeline.run(image,camera_params=params,seed=0,
          sparse_structure_sampler_params={'steps':12,'guidance_strength':7.5,'guidance_rescale':.7,'rescale_t':5.},
          shape_slat_sampler_params={'steps':12,'guidance_strength':7.5,'guidance_rescale':.5,'rescale_t':3.},
          tex_slat_sampler_params={'steps':12,'guidance_strength':1.,'guidance_rescale':0.,'rescale_t':3.},
          preprocess_image=False,return_latent=True,pipeline_type='1024_cascade',max_num_tokens=32768)
    if len(meshes)!=1 or int(resolution)!=1024:raise ValueError('unexpected cascade output')
    a.output.mkdir(parents=True,exist_ok=True);m=meshes[0];pred=a.output/'prediction.npz';np.savez_compressed(pred,vertices=m.vertices.detach().cpu().numpy(),faces=m.faces.detach().cpu().numpy(),coords=m.coords.detach().cpu().numpy(),attrs=m.attrs.detach().cpu().numpy())
    rec={'status':'PASS_DIAGNOSTIC_PIXAL3D','target_sha256':sha(a.target),'camera_job_sha256':sha(a.camera_job),'camera_mode':'source-mesh-estimated-unvalidated','camera_params':params,'mesh_scale_supplied':False,'attention_backend':'sdpa','cascade':1024,'seed':0,'prediction_sha256':sha(pred),'physical_gpu':a.physical_gpu,'evaluation_metrics_computed':False}
    (a.output/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)


if __name__=='__main__':main()
