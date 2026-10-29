"""Score a sharded mix of target and source-view geometry-ablation renders."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np

def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--registry',type=Path,required=True);p.add_argument('--benchmark-root',type=Path,required=True)
    p.add_argument('--physical-gpu',required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--world-size',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.physical_gpu not in '0,1,2,3,6,7'.split(',') or not 0<=a.rank<a.world_size:raise ValueError('authorized shard required')
    os.environ['CUDA_VISIBLE_DEVICES']=a.physical_gpu;os.environ['BENCHMARK_ASSIGNED_GPU']=a.physical_gpu;os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(2)
    import sys;sys.path.insert(0,str(a.benchmark_root));from deformnext_benchmark.io import read_rgba;from deformnext_benchmark.metrics import image_metrics
    from deformnext_benchmark.learned import OfflineMetrics
    tasks=rows(a.tasks)[a.rank::a.world_size];learned=OfflineMetrics(a.registry,device='cuda:0');result=[];pf=[];rf=[];feature_keys=[]
    for i,task in enumerate(tasks,1):
        pred,pa=read_rgba(task['prediction'],resolution=512,background=[1,1,1]);ref,ra=read_rgba(task['reference'],resolution=512,background=[1,1,1])
        if task['camera_id']==8:metrics=image_metrics(pred,ref,pa,ra)
        else:
            metrics=learned.image_pair(pred,ref);pf.append(learned.inception(pred));rf.append(learned.inception(ref));feature_keys.append([task['arm'],task['sample_id'],task['camera_id']])
        result.append({**{k:task[k] for k in ('arm','sample_id','camera_id')},'metrics':metrics})
        if i%25==0:print(json.dumps({'rank':a.rank,'position':i,'total':len(tasks)}),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in result))
    np.savez_compressed(a.output.with_suffix('.npz'),prediction=np.asarray(pf),reference=np.asarray(rf),keys=np.asarray([json.dumps(x) for x in feature_keys]))
if __name__=='__main__':main()
