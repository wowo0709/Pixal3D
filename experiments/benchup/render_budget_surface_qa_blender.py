import bpy,json,sys
import numpy as np
from pathlib import Path
from mathutils import Vector
args=sys.argv[sys.argv.index('--')+1:];source=Path(args[0]);out=Path(args[1]);budget=int(args[2]);bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(source))
objects=[o for o in bpy.context.scene.objects if o.type=='MESH'];assert len(objects)==1
obj=objects[0]
def arrays(mesh,matrix):
 mesh.calc_loop_triangles();v=np.empty(len(mesh.vertices)*3,dtype=np.float32);mesh.vertices.foreach_get('co',v);v=v.reshape(-1,3)
 t=np.array(matrix,dtype=np.float64);v=(v@t[:3,:3].T+t[:3,3]).astype(np.float32)
 f=np.empty(len(mesh.loop_triangles)*3,dtype=np.int32);mesh.loop_triangles.foreach_get('vertices',f);return v,f.reshape(-1,3)
v,f=arrays(obj.data,obj.matrix_world);face_count=len(f);assert face_count>budget
mod=obj.modifiers.new('common-render-only-budget','DECIMATE');mod.ratio=budget/face_count;mod.use_collapse_triangulate=True
obj_eval=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());eval_mesh=obj_eval.to_mesh();ev,ef=arrays(eval_mesh,obj_eval.matrix_world);obj_eval.to_mesh_clear()
out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'surfaces.npz',raw_vertices=v,raw_faces=f,decimated_vertices=ev,decimated_faces=ef)
(out/'blender.json').write_text(json.dumps({'original_faces':len(f),'decimated_faces':len(ef),'budget':budget,'ratio':budget/face_count,'blender_version':'.'.join(map(str,bpy.app.version))},indent=2)+'\n')
print(json.dumps({'original_faces':len(f),'decimated_faces':len(ef)}),flush=True)
