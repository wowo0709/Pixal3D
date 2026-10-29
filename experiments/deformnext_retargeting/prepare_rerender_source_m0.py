"""Fit source-only Sim(3) for regenerated geometry and retain paired surface scores."""
from __future__ import annotations
import argparse, concurrent.futures, json
from pathlib import Path
import numpy as np
from .score_source_sim3_baselines import apply, register_m0
from .surface_metrics import compare_surfaces

def compute(item):
    arm,sid,old,new,out=item
    with np.load(old,allow_pickle=False) as a, np.load(new,allow_pickle=False) as b:
        source=a['source_points'].copy();target=a['target_points'].copy()
        prediction=b['predicted_points'].copy();registration=b['registration_points'].copy()
    matrix,after,before=register_m0(registration,source)
    raw=compare_surfaces(prediction,target,threshold=.02)
    aligned=compare_surfaces(apply(prediction,matrix),target,threshold=.02)
    payload={'status':'PASS_REGENERATED_SOURCE_M0_SIM3','arm':arm,'sample_id':sid,
             'registration_reference':'input_source_M0_only','target_M1_used_for_registration':False,
             'correction_sim3':matrix.tolist(),'source_fit_before':before,'source_fit_after':after,
             'raw':raw,'aligned':aligned}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    return sid

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--original-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--workers',type=int,default=12)
    a=p.parse_args();campaign=json.loads(a.campaign.read_text());items=[]
    for row in campaign['plans']:
        arm,sid=row['arm'],row['sample_id'];out=a.output_root/arm/(sid+'.json')
        if out.is_file():continue
        old=a.original_root/'cases'/arm/sid/'shape'/'surface-evidence.npz'
        new=a.campaign.parent/'cases'/arm/sid/'shape'/'geometry-evaluation.npz'
        if not old.is_file() or not new.is_file():raise FileNotFoundError(str(old)+' / '+str(new))
        items.append((arm,sid,old,new,out))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,_ in enumerate(pool.map(compute,items),1):
            if i%50==0:print(json.dumps({'completed':i,'total':len(items)}),flush=True)
    print(json.dumps({'status':'PASS_REGENERATED_SOURCE_M0_SIM3_FITS','completed':len(items)}),flush=True)

if __name__=='__main__':main()
