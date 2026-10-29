"""Independently validate a bake rejected by vertex-only proximity."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from .export_textured_mesh import surface_fidelity
from .run_v1_cascade import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('prediction','receipt','bake-receipt','glb','benchmark-root','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    source=json.loads(a.receipt.read_text());bake=json.loads(a.bake_receipt.read_text())
    if bake.get('status')!='EXPORT_GEOMETRY_MISMATCH' or source['prediction_sha256']!=sha(a.prediction) or bake['prediction_sha256']!=source['prediction_sha256'] or bake['glb_sha256']!=sha(a.glb) or bake['sample_id']!=source['sample_id']:raise ValueError('input hashes or status differ')
    sys.path.insert(0,str(a.benchmark_root));from deformnext_benchmark.io import read_mesh
    with np.load(a.prediction) as z:vertices=z['vertices'].copy();faces=z['faces'].copy()
    exported,export_faces=read_mesh(a.glb);score=surface_fidelity(vertices,faces,exported,export_faces,a.benchmark_root)
    if max(score['raw_to_baked_q99'],score['baked_to_raw_q99'])>score['threshold']:raise ValueError('area-weighted surface check failed')
    bake.update(status='TEXTURED_GLB_BAKED_NEEDS_RENDER_QA',area_weighted_surface_validation=score,prior_bake_receipt_sha256=sha(a.bake_receipt),prior_status='EXPORT_GEOMETRY_MISMATCH')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(bake,indent=2)+'\n');print(json.dumps(bake),flush=True)

if __name__=='__main__':main()
