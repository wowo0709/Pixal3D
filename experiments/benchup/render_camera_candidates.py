"""Render completed source-only camera hypotheses on the authorized GPU set."""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image
from .fit_source_orthographic_pilot import camera_basis


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--benchmark-root',type=Path,required=True);p.add_argument('--blender',type=Path,required=True);p.add_argument('--producer-pid',type=int,required=True);a=p.parse_args()
    plan=json.loads((a.campaign/'plan.json').read_text());total_views=plan['views'];assert 0<total_views<=180 and plan['seeds']==[0,1,2,3]
    sys_path=str(a.benchmark_root)
    import sys;sys.path.insert(0,sys_path)
    from deformnext_benchmark.io import canonical_hash
    protocol=a.benchmark_root/'protocol.json';ph=canonical_hash(json.loads(protocol.read_text()));conv=np.array([[1.,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
    a.output.mkdir(parents=True,exist_ok=True);index_path=a.output/'results.jsonl';seen=set()
    if index_path.exists():
        seen={(x['object_id'],x['view_name']) for x in map(json.loads,index_path.read_text().splitlines())}
    while len(seen)<total_views:
        groups={}
        for f in sorted((a.campaign/'fits').glob('*/*-seed*.json')):
            stem=f.stem;view,seed=stem.rsplit('-seed',1);groups.setdefault((f.parent.name,view),{})[int(seed)]=f
        ready=sorted(k for k,v in groups.items() if set(v)=={0,1,2,3} and k not in seen)
        if not ready:
            try:os.kill(a.producer_pid,0)
            except ProcessLookupError:
                if not (a.campaign/'complete.json').is_file():raise RuntimeError('camera producer stopped before completion')
                raise RuntimeError('fitted view population does not close')
            time.sleep(5);continue
        group=ready[:4];chunk=a.output/f'chunk-{len(seen):03d}';stage=chunk/'stage';stage.mkdir(parents=True,exist_ok=False);jobs=[]
        for obj,view in group:
            mesh=a.input_root/obj/'source/src.glb';dest=stage/(obj+'.glb')
            if not dest.exists():os.link(mesh,dest)
            for seed in (0,1,2,3):
                fit=json.loads(groups[obj,view][seed].read_text());cfg=fit['parameters'];r,u,n=camera_basis(cfg['azimuth'],cfg['elevation'],cfg['roll']);scale=cfg['scale_px'];extent=fit['mesh_extent'];distance=10.;pos=n*distance-r*cfg['shift_x']/scale+u*cfg['shift_y']/scale;c=np.eye(4);c[:3,:3]=np.stack([r,u,n],axis=1);c[:3,3]=pos;c=conv@c;fov=2*math.atan((128/scale)/(2*distance));norm=np.eye(4);norm[:3,:3]/=extent;norm[:3,3]=-np.asarray(fit['mesh_center'])/extent;norm=conv@norm@conv.T
                jobs.append({'schema':'deformnext-render-job-v1','camera':{'camera_to_world':c.tolist(),'camera_angle_x':fov,'convention':'Blender_OpenGL_camera_to_world'},'camera_id':int(view.split('_')[-1]),'frame_id':'benchup-source-normalized','mesh':obj+'.glb','metric_mesh':obj+'.glb','method':'camera_fit_source_only','output_rgba':f'{obj}/{view}-seed{seed}.png','protocol_sha256':ph,'sample_id':f'{obj}__{view}','to_world':norm.tolist(),'trusted_material_dump':None})
        (chunk/'jobs.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs));(chunk/'plan.json').write_text(json.dumps({'jobs':len(jobs)}));(chunk/'input-files.txt').write_text(''.join(obj+'.glb\n' for obj in sorted({obj for obj,_ in group})))
        command=[sys.executable,'-u','-m','experiments.deformnext_retargeting.render_source_sim3_chunk','--chunk',str(chunk),'--stage-root',str(stage),'--blender',str(a.blender),'--batch-script',str(a.repo/'experiments/deformnext_retargeting/render_source_sim3_blender_batch.py'),'--benchmark-root',str(a.benchmark_root),'--protocol',str(protocol),'--gpus','0,1,2,3','--threads-per-gpu','2']
        subprocess.run(command,cwd=a.repo,check=True)
        results=[]
        for obj,view in group:
            with Image.open(a.input_root/obj/'source/mv'/f'{view}.webp') as img: ref=np.asarray(img.convert('RGBA').resize((512,512)))
            mask=ref[:,:,3]>127;candidates=[]
            for seed in (0,1,2,3):
                image=stage/obj/f'{view}-seed{seed}.png'
                with Image.open(image) as img: val=np.asarray(img.convert('RGBA'))
                pred=val[:,:,3]>127;overlap=mask&pred;union=mask|pred
                iou=float(overlap.sum()/max(union.sum(),1));mae=float(np.mean(np.abs(ref[:,:,:3].astype(float)-val[:,:,:3].astype(float))[overlap])/255) if overlap.any() else 1.
                candidates.append({'seed':seed,'mask_iou':iou,'rgb_mae_overlap':mae,'diagnostic_score':0.7*iou+0.3*(1-mae),'job':str(chunk/'jobs.jsonl'),'job_index':group.index((obj,view))*4+seed,'render':str(image),'fit':str(groups[obj,view][seed])})
            best=max(candidates,key=lambda x:x['diagnostic_score']);results.append({'object_id':obj,'view_name':view,'status':'source_render_QA_only_not_final_camera','best':best,'candidates':candidates})
        with index_path.open('a') as stream:
            for r in results:stream.write(json.dumps(r,sort_keys=True)+'\n')
        seen.update(group);print(json.dumps({'status':'source_camera_QA_in_progress','views':len(seen),'total':total_views,'recent':[(x['object_id'],x['view_name'],round(x['best']['mask_iou'],3)) for x in results]}),flush=True)
    (a.output/'complete.json').write_text(json.dumps({'status':'PASS_SOURCE_CAMERA_RENDER_QA_NOT_PAPER_VALIDATION','views':total_views,'source_only':True},indent=2)+'\n')


if __name__=='__main__':main()
