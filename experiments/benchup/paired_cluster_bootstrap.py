"""Paired object-cluster uncertainty for provisional BenchUp common cases."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

NAMES = ('PSNR', 'SSIM', 'M_IoU', 'source_LPIPS', 'source_DreamSim', 'source_CLIP')


def vector(runtime: Path, method: str, sid: str, mode: str) -> np.ndarray:
    basic = json.loads((runtime / (method+'-render-stream-v1') / 'cases' / sid / 'diagnostic-scores.json').read_text())
    learned = json.loads((runtime / (method+'-learned-metrics-v1') / sid / 'learned.json').read_text())
    if basic['sample_id'] != sid or learned['sample_id'] != sid:
        raise ValueError('sample identity mismatch')
    if basic['status'] != 'IMAGE_DIAGNOSTIC_PROVISIONAL_NOT_FINAL_BENCHMARK' or learned['status'] != 'PROVISIONAL_INPUT_IMAGE_METRICS':
        raise ValueError('invalid provisional metric status')
    target = basic['metrics'][mode][0]['metrics']
    source = [x['metrics'] for x in learned['metrics'][mode][1:]]
    if len(source) != 2:
        raise ValueError('expected exactly two source references')
    return np.array([target[k] for k in NAMES[:3]] + [np.mean([s[k] for s in source]) for k in ('LPIPS','DreamSim','CLIP')], dtype=np.float64)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--aggregate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replicates',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261030)
    a=p.parse_args()
    if a.replicates < 100: raise ValueError('at least 100 bootstrap replicates required')
    rows=[json.loads(line) for line in a.manifest.read_text().splitlines() if line]
    if len(rows)!=109 or len({x['sample_id'] for x in rows})!=109: raise ValueError('frozen manifest required')
    object_for={x['sample_id']:x['source_uid'] for x in rows}
    aggregate=json.loads(a.aggregate.read_text())
    if aggregate['status']!='PROVISIONAL_COMMON_SUCCESS_INPUT_IMAGE_METRICS': raise ValueError('common aggregate required')
    if aggregate['population_manifest_sha256']!=hashlib.sha256(a.manifest.read_bytes()).hexdigest(): raise ValueError('manifest differs from aggregate')
    sample_ids=aggregate['common_sample_ids']
    if len(set(sample_ids))!=aggregate['common_count'] or not set(sample_ids)<=object_for.keys(): raise ValueError('common population mismatch')
    mapping={'V1':'v1','Selected':'selected','MV-zero':'mv-zero','Official-RGB1':'official-rgb1','TRELLIS2':'trellis2','ReconViaGen':'recon','ShapeUP':'shapeup','SpaceControl':'spacecontrol','MVAdapter-Recon':'mvadapter-recon','CorrAdapter-Recon':'corradapter-recon'}
    clusters=sorted({object_for[s] for s in sample_ids})
    out={'status':'PROVISIONAL_PAIRED_CLUSTER_BOOTSTRAP','sample_count':len(sample_ids),'object_count':len(clusters),'replicates':a.replicates,'seed':a.seed,'comparison':'Selected minus comparator on identical cases; negative is favorable for LPIPS and DreamSim; positive is favorable for PSNR, SSIM, M-IoU and CLIP','warning':'Source references and estimated target camera are proxies; object clusters are resampled with replacement and all edits of a sampled object are retained. FID and KID are not included.','modes':{}}
    if len(clusters)<2:
        a.output.write_text(json.dumps(out,indent=2)+'\n');return
    index={c:np.array([i for i,s in enumerate(sample_ids) if object_for[s]==c]) for c in clusters}
    rng=np.random.default_rng(a.seed)
    draws=rng.integers(0,len(clusters),size=(a.replicates,len(clusters)))
    resamples=[np.concatenate([index[clusters[i]] for i in draw]) for draw in draws]
    for mode in ('raw','source_m0'):
        selected=np.stack([vector(a.runtime,'selected',sid,mode) for sid in sample_ids])
        mode_results={}
        for label,method in mapping.items():
            if label=='Selected' or label not in aggregate['methods']:continue
            diff=selected-np.stack([vector(a.runtime,method,sid,mode) for sid in sample_ids])
            draws_mean=np.stack([diff[ix].mean(axis=0) for ix in resamples])
            lo,hi=np.quantile(draws_mean,[0.025,0.975],axis=0)
            mean=diff.mean(axis=0)
            mode_results[label]={k:{'difference':float(mean[i]),'ci95_low':float(lo[i]),'ci95_high':float(hi[i])} for i,k in enumerate(NAMES)}
        out['modes'][mode]=mode_results
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(a.output)
    print(json.dumps({'status':out['status'],'cases':len(sample_ids),'objects':len(clusters),'output':str(a.output)}))

if __name__=='__main__':main()
