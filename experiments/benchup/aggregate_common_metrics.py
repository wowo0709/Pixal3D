"""Aggregate comparable BenchUp inputs on the exact common-success population."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

def fid_lowrank(pred,ref):
    """Exact empirical FID using the small sample Gram matrix."""
    x=np.asarray(pred,dtype=np.float64);y=np.asarray(ref,dtype=np.float64)
    if x.ndim!=2 or y.ndim!=2 or len(x)<2 or len(y)<2 or x.shape[1]!=y.shape[1] or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('valid feature matrices required')
    a=(x-x.mean(0))/np.sqrt(len(x)-1);b=(y-y.mean(0))/np.sqrt(len(y)-1)
    result=float(np.square(x.mean(0)-y.mean(0)).sum()+np.square(a).sum()+np.square(b).sum()-2*np.linalg.svd(a@b.T,compute_uv=False).sum())
    if result < -1e-6:raise ArithmeticError('invalid negative FID')
    return max(0.,result)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',action='append',nargs=3,metavar=('NAME','RENDER','LEARNED'),required=True)
    p.add_argument('--benchmark-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path);p.add_argument('--min-fid-count',type=int,default=2)
    a=p.parse_args();sys.path.insert(0,str(a.benchmark_root));from deformnext_benchmark.metrics import kernel_distance
    population=None
    if a.manifest:
        rows=[json.loads(line) for line in a.manifest.read_text().splitlines() if line]
        population={r['sample_id'] for r in rows}
        if len(rows)!=109 or len(population)!=109:raise ValueError('frozen 109-sample manifest required')
    data={}
    for name,render,learned in a.method:
        if name in data:raise ValueError('duplicate method')
        r=Path(render);l=Path(learned);ids={p.parent.name for p in r.glob('cases/*/diagnostic-scores.json')} & {p.parent.name for p in l.glob('*/learned.json')}
        if population is not None and not ids<=population:raise ValueError(f'unexpected sample in {name}')
        data[name]=(r,l,ids)
    if not data:raise ValueError('explicit methods required')
    common=sorted(set.intersection(*(v[2] for v in data.values())))
    result={'status':'PROVISIONAL_COMMON_SUCCESS_INPUT_IMAGE_METRICS','population_manifest_sha256':sha(a.manifest) if a.manifest else None,'common_count':len(common),'common_sample_ids':common,'available_counts':{k:len(v[2]) for k,v in data.items()},'target_reference':'edited input RGB, not held-out target GT','source_reference':'original source input RGB, can penalize intended edits','target_3d_metrics':'unavailable: no GT target mesh','lighting':'rendered predictions and original references may differ','distribution_note':'FID/KID compare input edited images with rendered predictions; n<100 and object-correlated samples are especially unreliable','methods':{}}
    expected_refs={};expected_identity=None
    for name,(render,learned,_) in data.items():
        outputs={};bundles=[]
        for sid in common:
            basic=json.loads((render/'cases'/sid/'diagnostic-scores.json').read_text());folder=learned/sid;score=json.loads((folder/'learned.json').read_text());npz=folder/'inception.npz'
            if basic['status']!='IMAGE_DIAGNOSTIC_PROVISIONAL_NOT_FINAL_BENCHMARK' or score['status']!='PROVISIONAL_INPUT_IMAGE_METRICS' or score['inception_sha256']!=sha(npz) or basic['sample_id']!=sid or score['sample_id']!=sid:raise ValueError(f'invalid evidence {name}:{sid}')
            identity=json.dumps(score['metric_identity'],sort_keys=True)
            if expected_identity is None:expected_identity=identity
            elif expected_identity!=identity:raise ValueError('metric identity differs')
            refs=[z['reference_sha256'] for z in score['metrics']['raw']]
            if sid in expected_refs and expected_refs[sid]!=refs:raise ValueError('reference image differs across methods')
            expected_refs[sid]=refs
            with np.load(npz) as features: arrays={k:features[k].copy() for k in ('reference_target','raw_target','source_m0_target')}
            bundles.append((basic,score,arrays))
        for mode in ('raw','source_m0'):
            if not bundles:outputs[mode]=None;continue
            metrics=[]
            for basic,score,_ in bundles:
                target=basic['metrics'][mode][0]['metrics'];source=[x['metrics'] for x in score['metrics'][mode][1:]]
                metrics.append([target['PSNR'],target['SSIM'],target['M_IoU'],*[np.mean([s[k] for s in source]) for k in ('LPIPS','DreamSim','CLIP')]])
            means=np.asarray(metrics,dtype=np.float64).mean(0);names=('PSNR','SSIM','M_IoU','source_LPIPS','source_DreamSim','source_CLIP')
            values={k:float(v) for k,v in zip(names,means)};values.update(FID=None,KID=None)
            if len(common)>=a.min_fid_count:
                pred=np.stack([x[2][mode+'_target'] for x in bundles]);ref=np.stack([x[2]['reference_target'] for x in bundles])
                values['FID']=fid_lowrank(pred,ref);values['KID']=kernel_distance(pred,ref,subset_size=min(100,len(common)),subsets=100,seed=20261020)
            outputs[mode]=values
        result['methods'][name]=outputs
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');tmp.replace(a.output)
    print(json.dumps({'status':result['status'],'common_count':len(common),'available_counts':result['available_counts'],'output':str(a.output)},ensure_ascii=False))

if __name__=='__main__':main()
