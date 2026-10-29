"""Register an already-textured native GLB to the input source mesh only."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from .run_v1_cascade import sha
from .register_source_m0 import output_to_world
from experiments.deformnext_retargeting.score_source_sim3_baselines import apply,register_m0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','glb','receipt','source-surface','benchmark-root','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--method',choices=['shapeup','spacecontrol'],required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    receipt=json.loads(a.receipt.read_text());sid=receipt['sample_id'];digest=sha(a.glb)
    expected={'shapeup':'SHAPEUP_TEXTURED_GENERATED','spacecontrol':'SPACECONTROL_TARGET_PROXY_GENERATED'}[a.method]
    if receipt.get('status')!=expected or receipt.get('glb_sha256')!=digest:raise ValueError('native artifact provenance mismatch')
    rows=[r for r in map(json.loads,a.manifest.read_text().splitlines()) if r['sample_id']==sid]
    if len(rows)!=1:raise ValueError('unique manifest row required')
    row=rows[0];sys.path.insert(0,str(a.benchmark_root))
    from deformnext_benchmark.io import read_mesh
    from deformnext_benchmark.metrics import sample_surface
    verts,faces=read_mesh(a.glb)
    if len(verts)==0 or len(faces)==0 or not np.isfinite(verts).all():raise ValueError('invalid native mesh')
    with np.load(a.source_surface) as z:points=z['points'].astype(np.float64);bounds=z['bounds'].astype(np.float64)
    center=bounds.mean(0);extent=np.max(bounds[1]-bounds[0]);blender=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    fixed=((points-center)/extent)@blender.T
    if a.method=='shapeup':
        # Official texture pipeline normalizes the predicted mesh to unit extent
        # in the original source object's coordinate frame before GLB export.
        to_world=np.eye(4);frame='shapeup_source_frame_gltf_import'
    else:
        # trimesh GLB export is Y-up; Blender converts this by B. SpaceControl
        # uses the same TRELLIS native frame as its target-proxy control.
        b=np.eye(4);b[:3,:3]=blender;q=np.eye(4);q[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
        to_world=output_to_world(row)@q@np.linalg.inv(b);frame='spacecontrol_trellis_to_blender_import'
    seed=int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)
    moving=apply(sample_surface(verts,faces,count=4096,seed=seed+103),to_world)
    rng=np.random.default_rng(seed+101);fixed=fixed[rng.choice(len(fixed),size=4096,replace=False)]
    correction,after,before=register_m0(moving,fixed)
    a.output.mkdir(parents=True)
    reg={'status':'SOURCE_M0_REGISTRATION_ONLY_NEEDS_RENDER_QA','sample_id':sid,'prediction_sha256':digest,'source_surface_sha256':sha(a.source_surface),'registration_reference':'input_source_mesh_only','target_GT_used':False,'native_frame':frame,'to_world':to_world.tolist(),'correction_sim3':correction.tolist(),'source_fit_before':before,'source_fit_after':after,'source_fit_is_not_target_accuracy':True,'source_camera_qa_iou':row['source_camera_qa_iou'],'target_camera_assumption':row['target_camera_assumption']}
    bake={'status':'NATIVE_TEXTURED_GLB_VALIDATED_NEEDS_RENDER_QA','sample_id':sid,'prediction_sha256':digest,'glb_sha256':digest,'source_receipt_sha256':sha(a.receipt),'native_glb_no_additional_bake':True,'vertices':len(verts),'faces':len(faces)}
    (a.output/'source-registration.json').write_text(json.dumps(reg,indent=2)+'\n');(a.output/'native-glb.json').write_text(json.dumps(bake,indent=2)+'\n');print(json.dumps(reg),flush=True)

if __name__=='__main__':main()
