"""Diagnostic source-only similarity ICP; output is not an evaluation camera."""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree


def similarity(a, b):
    am, bm = a.mean(0), b.mean(0)
    ac, bc = a-am, b-bm
    u, d, vt = np.linalg.svd(bc.T @ ac / len(a))
    sign = np.eye(3); sign[-1, -1] = np.linalg.det(u @ vt)
    rot = u @ sign @ vt
    scale = np.sum(d*np.diag(sign)) / np.mean(np.sum(ac*ac, axis=1))
    return scale, rot, bm-scale*(rot @ am)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--images', nargs=4, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    z = np.load(a.prediction); extr=z['extrinsics'][0]; intr=z['intrinsics'][0]; depth=z['depth'][0, ..., 0]; conf=z['depth_conf'][0]
    rng=np.random.default_rng(0); per=[]
    for i,path in enumerate(a.images):
        with Image.open(path) as im: mask=np.asarray(im.convert('RGBA').getchannel('A').resize((512,512))) > 127
        valid=mask & np.isfinite(depth[i]) & (depth[i] > 0) & (conf[i] > np.quantile(conf[i][mask],0.4))
        y,x=np.nonzero(valid)
        if len(y)<500:raise ValueError('too few valid foreground pixels')
        idx=rng.choice(len(y),min(2500,len(y)),replace=False); x=x[idx];y=y[idx];d=depth[i,y,x]
        cam=np.stack([(x-intr[i,0,2])*d/intr[i,0,0],(y-intr[i,1,2])*d/intr[i,1,1],d],axis=1)
        per.append((cam-extr[i,:3,3]) @ extr[i,:3,:3])
    source=np.concatenate(per); scene=trimesh.load(a.mesh,force='scene');mesh=scene.to_geometry(); target,_=trimesh.sample.sample_surface(mesh,20000,seed=0)
    tree=cKDTree(target); s0=np.linalg.norm(np.percentile(target,95,axis=0)-np.percentile(target,5,axis=0))/np.linalg.norm(np.percentile(source,95,axis=0)-np.percentile(source,5,axis=0))
    candidates=[]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            rot=np.eye(3)[list(perm)]*np.array(signs)[:,None]
            if np.linalg.det(rot)<0.5:continue
            s=s0; trans=target.mean(0)-s*rot@source.mean(0)
            for _ in range(30):
                mapped=s*(source@rot.T)+trans;dist,ind=tree.query(mapped,workers=2); keep=dist <= np.quantile(dist,0.8)
                ns,nr,nt=similarity(source[keep],target[ind[keep]])
                if abs(ns-s)<1e-7 and np.linalg.norm(nt-trans)<1e-7:break
                s,rot,trans=ns,nr,nt
            mapped=s*(source@rot.T)+trans;dist,_=tree.query(mapped,workers=2)
            candidates.append({'rmse_trim80':float(np.sqrt(np.mean(np.sort(dist)[:int(.8*len(dist))]**2))), 'median':float(np.median(dist)), 'p90':float(np.quantile(dist,.9)), 'scale':float(s), 'rotation':rot.tolist(), 'translation':trans.tolist()})
    candidates.sort(key=lambda c:c['rmse_trim80'])
    output={'status':'diagnostic_unvalidated','source_points':len(source),'target_points':len(target),'mesh_bounds':scene.bounds.tolist(),'candidate_count':len(candidates),'best':candidates[0],'next':candidates[1:5]}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output),flush=True)


if __name__=='__main__':main()
