"""Run pinned TRELLIS.2 on BenchUp edited RGB without camera or target geometry."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--physical-gpu',type=int,choices=(0,1,2,3),required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if not 0<=a.rank<a.world_size:raise ValueError('invalid worker rank')
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines()];ids={r['sample_id'] for r in rows}
    if len(rows)!=109 or len(ids)!=109:raise ValueError('frozen BenchUp population required')
    if not (a.model/'pipeline.json').is_file():raise FileNotFoundError('TRELLIS.2 checkpoint not complete')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.physical_gpu),ATTN_BACKEND='sdpa',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='1')
    import torch,numpy as np
    from PIL import Image
    from pixal3d.pipelines import Trellis2ImageTo3DPipeline
    torch.set_num_threads(3);torch.cuda.set_device(0)
    pipeline=Trellis2ImageTo3DPipeline.from_pretrained(str(a.model));pipeline.cuda()
    selected=rows[a.rank::a.world_size]
    if a.limit is not None:selected=selected[:a.limit]
    a.output.mkdir(parents=True,exist_ok=True)
    for n,row in enumerate(selected,1):
        sid=row['sample_id'];target=a.input_root/row['target_image'];case=a.output/'cases'/sid;receipt=case/'receipt.json'
        if receipt.is_file():
            d=json.loads(receipt.read_text())
            if d.get('target_sha256')!=sha(target) or d.get('prediction_sha256')!=sha(case/'prediction.npz'):raise ValueError('existing case differs')
            print(json.dumps({'status':'reused','sample_id':sid}),flush=True);continue
        if case.exists():raise FileExistsError('partial case needs inspection: '+str(case))
        started=time.monotonic();im=Image.open(target).convert('RGBA')
        with torch.inference_mode():
            meshes,(_,_,resolution)=pipeline.run(im,seed=0,return_latent=True,pipeline_type='1024_cascade',max_num_tokens=32768)
        if len(meshes)!=1 or resolution!=1024:raise ValueError('unexpected TRELLIS.2 output')
        m=meshes[0];vertices=m.vertices.detach().cpu().numpy();faces=m.faces.detach().cpu().numpy();attrs=m.attrs.detach().cpu().numpy();coords=m.coords.detach().cpu().numpy()
        if not len(vertices) or not len(faces) or not np.isfinite(vertices).all() or not np.isfinite(attrs).all():raise ValueError('invalid output mesh')
        case.mkdir(parents=True);pred=case/'prediction.npz';np.savez_compressed(pred,vertices=vertices,faces=faces,attrs=attrs,coords=coords)
        rec={'status':'PASS_TRELLIS2_GENERATED_RAW','sample_id':sid,'target_sha256':sha(target),'prediction_sha256':sha(pred),'pipeline_config_sha256':sha(a.model/'pipeline.json'),'model_revision':'af44b45f2e35a493886929c6d786e563ec68364d','target_image_only':True,'source_mesh_used':False,'camera_used':False,'seed':0,'cascade':1024,'attention_backend':'sdpa','physical_gpu':a.physical_gpu,'seconds':round(time.monotonic()-started,2)}
        receipt.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({'position':n,'total':len(selected),**rec}),flush=True)


if __name__=='__main__':main()
