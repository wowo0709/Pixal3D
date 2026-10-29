"""Blender script: sample the evaluated source surface, including skinning at frame 1."""
import argparse,json,sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=30000);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(a.mesh));bpy.context.scene.frame_set(1);deps=bpy.context.evaluated_depsgraph_get();triangles=[];objects=[]
    conv=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
    for obj in bpy.context.scene.objects:
        if obj.type!='MESH' or obj.hide_render or all(c.hide_render for c in obj.users_collection):continue
        ev=obj.evaluated_get(deps);m=ev.to_mesh();m.calc_loop_triangles()
        verts=np.empty((len(m.vertices),3),dtype=np.float32);m.vertices.foreach_get('co',verts.ravel())
        transform=np.asarray(ev.matrix_world,dtype=np.float64);verts=verts@transform[:3,:3].T+transform[:3,3];verts=verts@conv.T
        faces=np.empty((len(m.loop_triangles),3),dtype=np.int32);m.loop_triangles.foreach_get('vertices',faces.ravel());triangles.append(verts[faces]);objects.append({'name':obj.name,'vertices':len(verts),'triangles':len(faces),'modifiers':[x.type for x in obj.modifiers]});ev.to_mesh_clear()
    t=np.concatenate(triangles);area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2
    if not np.isfinite(area).all() or not area.sum()>0:raise ValueError('invalid evaluated surface')
    rng=np.random.default_rng(0);idx=rng.choice(len(t),size=a.samples,p=area/area.sum());r=rng.random((a.samples,2));flip=r.sum(axis=1)>1;r[flip]=1-r[flip];sel=t[idx];samples=sel[:,0]+r[:,:1]*(sel[:,1]-sel[:,0])+r[:,1:]*(sel[:,2]-sel[:,0]);bounds=np.stack((t.min(axis=(0,1)),t.max(axis=(0,1))))
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,points=samples.astype(np.float32),bounds=bounds.astype(np.float32));print(json.dumps({'status':'PASS_EVALUATED_FRAME1_SURFACE','objects':objects,'bounds':bounds.tolist(),'samples':a.samples}))


if __name__=='__main__':main()
