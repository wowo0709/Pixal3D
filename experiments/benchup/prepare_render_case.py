"""Prepare raw and source-aligned V1 renders with three calibrated views."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
from .run_v1_cascade import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest','registration','glb','bake-receipt','benchmark-root','output'):p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    reg=json.loads(a.registration.read_text());bake=json.loads(a.bake_receipt.read_text());sid=reg['sample_id']
    if bake['sample_id']!=sid or bake['glb_sha256']!=sha(a.glb) or bake['prediction_sha256']!=reg['prediction_sha256']:raise ValueError('prediction/bake/registration mismatch')
    rows=[x for x in map(json.loads,a.manifest.read_text().splitlines()) if x['sample_id']==sid]
    if len(rows)!=1:raise ValueError('unique input required')
    row=rows[0]
    sys.path.insert(0,str(a.benchmark_root));from deformnext_benchmark.io import canonical_hash
    ph=canonical_hash(json.loads((a.benchmark_root/'protocol.json').read_text()))
    stage=a.output/'stage';stage.mkdir(parents=True);os.link(a.glb,stage/'prediction.glb')
    # o_voxel writes GLB axes so Blender's glTF import reconstructs the raw
    # NPZ coordinates. The bake geometry check verifies that convention.
    raw=np.asarray(reg['to_world'])
    aligned=np.asarray(reg['correction_sim3'])@raw
    jobs=[]
    for mode,matrix in [('raw',raw),('source_m0',aligned)]:
        for index in range(3):
            jobs.append({'schema':'deformnext-render-job-v1','camera':{'camera_to_world':row['transform_matrix'][index],'camera_angle_x':row['camera_angle_x'][index],'convention':'Blender_OpenGL_camera_to_world'},'camera_id':index,'frame_id':'benchup-source-normalized','mesh':'prediction.glb','metric_mesh':'prediction.glb','method':'v1_'+mode,'output_rgba':f'{mode}/view{index}.png','protocol_sha256':ph,'sample_id':sid,'to_world':matrix.tolist(),'trusted_material_dump':None})
    (a.output/'jobs.jsonl').write_text(''.join(json.dumps(j,sort_keys=True)+'\n' for j in jobs));(a.output/'plan.json').write_text(json.dumps({'jobs':len(jobs),'sample_id':sid,'status':'PREPARED_RENDER_CONVENTION_QA'})+'\n');(a.output/'input-files.txt').write_text('prediction.glb\n')
    print(json.dumps({'sample_id':sid,'jobs':len(jobs),'output':str(a.output)}))

if __name__=='__main__':main()
