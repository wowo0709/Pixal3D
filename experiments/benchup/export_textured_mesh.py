"""Bake a BenchUp raw 1024 prediction to GLB, retaining the raw prediction."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import numpy as np

def surface_fidelity(vertices,faces,exported,export_faces,benchmark_root,samples=100000):
    """Upper-bound surface distance using deterministic area-uniform samples."""
    import sys
    from scipy.spatial import cKDTree
    sys.path.insert(0,str(benchmark_root));from deformnext_benchmark.metrics import sample_surface
    if samples<100000:raise ValueError('at least 100000 surface samples required')
    raw=sample_surface(vertices,faces,count=samples,seed=123)
    baked=sample_surface(exported,export_faces,count=samples,seed=456)
    forward=cKDTree(baked).query(raw,workers=2)[0]
    backward=cKDTree(raw).query(baked,workers=2)[0]
    return {'samples_per_surface':samples,'seed_raw':123,'seed_baked':456,'raw_to_baked_q99':float(np.quantile(forward,.99)),'baked_to_raw_q99':float(np.quantile(backward,.99)),'threshold':.01,'interpretation':'sampled-surface nearest-neighbor distance upper bounds nearest continuous-surface distance'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',required=True,choices=['0','1','2','3']);p.add_argument('--face-budget',type=int,default=1000000);p.add_argument('--texture-size',type=int,default=2048)
    p.add_argument('--benchmark-root',type=Path,default=Path('/root/dev/.cache/deformnext-benchmark-runtime'))
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu or a.output.exists():raise ValueError('assigned GPU and fresh output required')
    from .run_v1_cascade import sha
    receipt=json.loads(a.receipt.read_text())
    if receipt.get('status') not in {'PASS_TRELLIS2_GENERATED_RAW','RECONVIAGEN_V05_GENERATED_RAW','OFFICIAL_PIXAL3D_1024_ESTIMATED_CAMERA_UNCROPPED','OFFICIAL_PIXAL3D_1536_ESTIMATED_CAMERA_UNCROPPED','V1_CASCADE_GENERATED_NO_TARGET_3D_GT','CASCADE_GENERATED_NO_TARGET_3D_GT','SELECTED_FULL_PBR_GENERATED_NO_TARGET_3D_GT'} or receipt['prediction_sha256']!=sha(a.prediction):raise ValueError('source prediction receipt mismatch')
    grid_size=int(receipt.get('resolution',1024))
    if grid_size not in (1024,1152,1280,1408,1536):raise ValueError('invalid actual output grid')
    import torch,o_voxel,cumesh
    from fast_simplification import simplify
    torch.set_num_threads(2);torch.cuda.set_device(0);started=time.monotonic()
    with np.load(a.prediction,allow_pickle=False) as z:
        vertices=z['vertices'].copy();faces=z['faces'].copy();coords=z['coords'].copy();attrs=z['attrs'].copy()
    if len(vertices)==0 or len(faces)==0 or attrs.shape!=(len(coords),6) or not np.isfinite(vertices).all() or not np.isfinite(attrs).all():raise ValueError('invalid native prediction')
    if len(faces)>a.face_budget:
        v,f=simplify(vertices,faces,target_count=a.face_budget,agg=7)
        vertices_out=np.asarray(v,dtype=np.float32);faces_out=np.asarray(f,dtype=np.int32)
    else:vertices_out=vertices;faces_out=faces
    old=cumesh.CuMesh
    class PreserveTopologyMesh(old):
        def simplify(self,*args,**kwargs):return None
        def fill_holes(self,*args,**kwargs):return None
        def remove_duplicate_faces(self,*args,**kwargs):return None
        def repair_non_manifold_edges(self,*args,**kwargs):return None
        def remove_small_connected_components(self,*args,**kwargs):return None
        def unify_face_orientations(self,*args,**kwargs):return None
    cumesh.CuMesh=PreserveTopologyMesh
    try:
        glb=o_voxel.postprocess.to_glb(vertices=torch.from_numpy(vertices_out).cuda(),faces=torch.from_numpy(faces_out).cuda(),attr_volume=torch.from_numpy(attrs).cuda(),coords=torch.from_numpy(coords).cuda(),attr_layout={'base_color':slice(0,3),'metallic':slice(3,4),'roughness':slice(4,5),'alpha':slice(5,6)},aabb=[[-.5,-.5,-.5],[.5,.5,.5]],grid_size=grid_size,decimation_target=a.face_budget,texture_size=a.texture_size,remesh=False,use_tqdm=False)
    finally:cumesh.CuMesh=old
    a.output.parent.mkdir(parents=True,exist_ok=True);glb.export(str(a.output),extension_webp=True)
    import sys
    from scipy.spatial import cKDTree
    sys.path.insert(0,str(a.benchmark_root));from deformnext_benchmark.io import read_mesh
    exported,export_faces=read_mesh(a.output);rng=np.random.default_rng(42017)
    original_probe=vertices[rng.choice(len(vertices),min(4096,len(vertices)),replace=False)]
    export_probe=exported[rng.choice(len(exported),min(4096,len(exported)),replace=False)]
    q99=float(max(np.quantile(cKDTree(exported).query(original_probe,workers=1)[0],.99),np.quantile(cKDTree(vertices).query(export_probe,workers=1)[0],.99)))
    surface=None
    if not np.isfinite(q99) or q99>.01:surface=surface_fidelity(vertices,faces,exported,export_faces,a.benchmark_root)
    passed=np.isfinite(q99) and q99<=.01 or surface is not None and max(surface['raw_to_baked_q99'],surface['baked_to_raw_q99'])<=surface['threshold']
    result={'status':'TEXTURED_GLB_BAKED_NEEDS_RENDER_QA' if passed else 'EXPORT_GEOMETRY_MISMATCH','sample_id':receipt['sample_id'],'prediction_sha256':receipt['prediction_sha256'],'glb_sha256':sha(a.output),'original_faces':len(faces),'render_faces':len(faces_out),'grid_size':grid_size,'raw_frame_vertex_distance_q99':q99,'area_weighted_surface_validation':surface,'texture_size':a.texture_size,'source_native_frame_preserved':True,'source_mesh_or_target_gt_used':False,'seconds':round(time.monotonic()-started,2)}
    a.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status']!='TEXTURED_GLB_BAKED_NEEDS_RENDER_QA':raise ValueError('baked and original geometry differ; preserve for inspection')

if __name__=='__main__':main()
