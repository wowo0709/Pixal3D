"""Paired voxel-count comparison; repeated sampling seeds stay in source clusters."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(pairs, draws=4000):
    groups={};noise_results=[];population=None
    for name,baseline,candidate in pairs:
        left={r['sample_id']:r for r in baseline['per_asset']}
        right={r['sample_id']:r for r in candidate['per_asset']}
        if len(left)!=len(baseline['per_asset']) or len(right)!=len(candidate['per_asset']):
            raise ValueError('duplicate sample in paired evaluation')
        if left.keys()!=right.keys() or (population is not None and set(left)!=population):
            raise ValueError('paired sampling panels differ')
        population=set(left);counts=[]
        for sid in sorted(left):
            b,c=left[sid],right[sid]
            if b['seed']!=c['seed'] or b['source_cluster_id']!=c['source_cluster_id']:
                raise ValueError('paired noise seed or source cluster differs')
            cluster=b['source_cluster_id']
            if not cluster:raise ValueError('source cluster is required')
            x=np.array([c['tp'],c['fp'],c['fn'],b['tp'],b['fp'],b['fn']],dtype=np.float64)
            groups[cluster]=groups.get(cluster,np.zeros(6))+x;counts.append(x)
        total=np.stack(counts).sum(0)
        ciou=total[0]/total[:3].sum();biou=total[3]/total[3:].sum()
        noise_results.append({'noise':name,'baseline_iou':float(biou),'candidate_iou':float(ciou),'gain':float(ciou-biou)})
    if not groups:raise ValueError('empty paired evaluation')
    array=np.stack([groups[k] for k in sorted(groups)]);total=array.sum(0)
    ciou=total[0]/total[:3].sum();biou=total[3]/total[3:].sum()
    rng=np.random.default_rng(42017)
    sample=array[rng.integers(0,len(array),size=(draws,len(array)))].sum(1)
    gain=sample[:,0]/sample[:,:3].sum(1)-sample[:,3]/sample[:,3:].sum(1)
    return {'assets':len(population),'source_clusters':len(groups),'sampling_noise_seeds':len(pairs),
        'baseline_iou':float(biou),'candidate_iou':float(ciou),'gain':float(ciou-biou),
        'paired_cluster_bootstrap_ci95':np.quantile(gain,[.025,.975]).tolist(),
        'bootstrap_draws':draws,'bootstrap_seed':42017,'per_noise':noise_results,
        'aggregation':'sum TP/FP/FN across equal sampling repeats; bootstrap source clusters with all repeats together',
        'training_seed_replication':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-list',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();paths=json.loads(a.input_list.read_text());pairs=[]
    for x in paths:
        pairs.append((x['noise'],json.loads(Path(x['baseline']).read_text()),json.loads(Path(x['candidate']).read_text())))
    result=compare(pairs);result['inputs']=paths
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
