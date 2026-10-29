"""Diagnostic: fit an orthographic source-mesh silhouette to a source view."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.optimize import differential_evolution


def camera_basis(azimuth,elevation,roll):
    outward=np.array([np.sin(azimuth)*np.cos(elevation),np.sin(elevation),np.cos(azimuth)*np.cos(elevation)])
    right=np.cross(np.array([0.,1.,0.]),outward);right/=np.linalg.norm(right)
    up=np.cross(outward,right);rr=right*np.cos(roll)+up*np.sin(roll);uu=up*np.cos(roll)-right*np.sin(roll)
    return rr,uu,outward


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--evaluated-surface',type=Path);a=p.parse_args()
    if a.evaluated_surface:
        with np.load(a.evaluated_surface) as surface:
            points=surface['points'].copy()[:22000];bounds=surface['bounds'].copy()
    else:
        scene=trimesh.load(a.mesh,force='scene'); mesh=scene.to_geometry(); points,_=trimesh.sample.sample_surface(mesh,22000,seed=0);bounds=np.asarray(scene.bounds)
    center=bounds.mean(axis=0); extent=float(np.max(bounds[1]-bounds[0]))
    if not np.isfinite(extent) or extent <= 0:raise ValueError('invalid source mesh extent')
    points=(points-center)/extent
    with Image.open(a.image) as image: mask=(np.asarray(image.convert('RGBA').getchannel('A').resize((128,128)))>127).astype(np.uint8)
    kernel=np.ones((3,3),np.uint8)
    def projection(x):
        az,el,roll,scale,tx,ty=x;right,up,_=camera_basis(az,el,roll)
        xx=np.rint(64+scale*(points@right)+tx).astype(int); yy=np.rint(64-scale*(points@up)+ty).astype(int)
        valid=(xx>=0)&(xx<128)&(yy>=0)&(yy<128); out=np.zeros((128,128),np.uint8);out[yy[valid],xx[valid]]=1
        return cv2.morphologyEx(cv2.dilate(out,kernel),cv2.MORPH_CLOSE,kernel)
    def objective(x):
        pred=projection(x); inter=np.count_nonzero(pred&mask); union=np.count_nonzero(pred|mask)
        return 1-inter/max(union,1)
    bounds=[(-np.pi,np.pi),(-1.35,1.35),(-np.pi,np.pi),(25,220),(-25,25),(-25,25)]
    result=differential_evolution(objective,bounds,seed=a.seed,popsize=9,maxiter=140,tol=0.002,polish=True,workers=1)
    output={'status':'diagnostic_unvalidated_orthographic','seed':a.seed,'silhouette_iou':float(1-result.fun),'parameters':dict(zip(('azimuth','elevation','roll','scale_px','shift_x','shift_y'),map(float,result.x))),'mesh_center':center.tolist(),'mesh_extent':extent,'mesh':str(a.mesh),'evaluated_surface':str(a.evaluated_surface) if a.evaluated_surface else None,'source_image':str(a.image),'evaluations':result.nfev}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2)+'\n');Image.fromarray(projection(result.x)*255).save(a.output.with_suffix('.png'));print(json.dumps(output),flush=True)

if __name__=='__main__':main()
