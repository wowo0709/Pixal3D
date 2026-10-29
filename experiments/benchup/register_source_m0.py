"""Derive Pixal3D camera-frame conversion and source-only Sim(3) registration."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from experiments.deformnext_retargeting.historical_geometry import target_anchor_affine,world_to_target_anchor_points
from experiments.deformnext_retargeting.score_source_sim3_baselines import apply,register_m0
from .run_v1_cascade import sha

def output_to_world(row):
    scale=float(row['mesh_scale'])
    if not np.isfinite(scale) or scale<=0:raise ValueError('invalid predicted image scale')
    affine=target_anchor_affine({'transform_matrix':row['transform_matrix'][0]})
    s=np.eye(4);s[:3,:3]*=scale
    result=np.linalg.inv(s@affine)
    # Exercise the real training-side world-to-anchor mapping, including its
    # translation, instead of silently assuming a rotation-only camera.
    probe=np.array([[0,0,0],[.1,-.2,.3],[-.3,.2,.1]],np.float32)
    forward=world_to_target_anchor_points(probe,{'transform_matrix':row['transform_matrix'][0]},scale=scale)
    if not np.allclose(apply(forward,result),probe,atol=2e-6):raise ValueError('anchor affine round trip failed')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','prediction','receipt','source-surface','output','benchmark-root'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    receipt=json.loads(a.receipt.read_text());sid=receipt['sample_id']
    if receipt['prediction_sha256']!=sha(a.prediction):raise ValueError('prediction mismatch')
    rows=[r for r in map(json.loads,a.manifest.read_text().splitlines()) if r['sample_id']==sid]
    if len(rows)!=1:raise ValueError('sample manifest identity mismatch')
    row=rows[0];to_world=output_to_world(row)
    with np.load(a.source_surface) as z: points=z['points'].astype(np.float64);bounds=z['bounds'].astype(np.float64)
    center=bounds.mean(0);extent=np.max(bounds[1]-bounds[0])
    conv=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    fixed=((points-center)/extent)@conv.T
    sys.path.insert(0,str(a.benchmark_root));from deformnext_benchmark.metrics import sample_surface
    with np.load(a.prediction) as z:vertices=z['vertices'].copy();faces=z['faces'].copy()
    seed=int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)
    moving=apply(sample_surface(vertices,faces,count=4096,seed=seed+103),to_world)
    rng=np.random.default_rng(seed+101);fixed=fixed[rng.choice(len(fixed),size=4096,replace=False)]
    correction,after,before=register_m0(moving,fixed)
    result={'status':'SOURCE_M0_REGISTRATION_ONLY_NEEDS_RENDER_QA','sample_id':sid,'prediction_sha256':sha(a.prediction),'source_surface_sha256':sha(a.source_surface),'registration_reference':'input_source_mesh_evaluated_frame1_only','target_GT_used':False,'to_world':to_world.tolist(),'correction_sim3':correction.tolist(),'source_fit_before':before,'source_fit_after':after,'source_fit_is_not_target_accuracy':True,'source_camera_qa_iou':row['source_camera_qa_iou'],'target_camera_assumption':row['target_camera_assumption']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
