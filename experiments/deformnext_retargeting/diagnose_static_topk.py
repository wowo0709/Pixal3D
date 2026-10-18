"""Post-hoc descriptor/value diagnosis of completed static Top1/soft SS runs.

CPU only. Reads observed RGB alpha, cached proposals/native values and native
projection indices. It never opens depth, topology labels or SS latent arrays.
Generation outcomes come from the already completed fixed-final evaluations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from .semantic_topk_search import write_new


def evidence_statistics(values,indices,logits,valid,patch_grid,temperature,gate):
    values=np.asarray(values,dtype=np.float32);indices=np.asarray(indices)
    logits=np.asarray(logits,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
    grid=np.asarray(patch_grid).reshape(-1)
    if (values.ndim!=2 or indices.ndim!=2 or logits.shape!=indices.shape or valid.shape!=indices.shape
        or not np.issubdtype(indices.dtype,np.integer) or not np.issubdtype(grid.dtype,np.integer)
        or (indices<0).any() or (indices>=len(values)).any() or (grid<0).any() or (grid>=len(indices)).any()
        or not np.isfinite(values).all() or not np.isfinite(logits[valid]).all()
        or not np.isfinite(temperature) or not np.isfinite(gate) or temperature<=0):
        raise ValueError('invalid cached value/score/projection contract')
    active=valid.any(-1);multiplicity=np.bincount(grid,minlength=len(indices))[active]
    if not active.any():
        return {'queries':0,'projected_voxels':0,'native_voxels_evaluated':len(grid),
            'query_reaching_projection':0,'sums':{},'query_distribution':{}}
    ids=indices[active];mask=valid[active]
    scores=np.where(mask,logits[active],-np.inf)
    weights=np.exp((scores-scores.max(-1,keepdims=True))/temperature)
    weights/=weights.sum(-1,keepdims=True)
    entropy=-(weights*np.log(np.clip(weights,1e-300,1))).sum(-1)
    effective=np.exp(entropy);topmass=weights.max(-1);top=values[ids[np.arange(len(ids)),scores.argmax(-1)]]
    norm2=np.square(top,dtype=np.float64).sum(-1)
    if (norm2==0).any():raise ValueError('active native Top1 feature has zero norm')
    evidence=np.empty_like(top)
    for start in range(0,len(ids),64):
        selected=values[ids[start:start+64]]
        evidence[start:start+64]=np.einsum('qk,qkd->qd',weights[start:start+64],selected,optimize=True)
    delta2=np.square(evidence-top,dtype=np.float64).sum(-1)
    evidence2=np.square(evidence,dtype=np.float64).sum(-1)
    cosine=(evidence.astype(np.float64)*top).sum(-1)/np.sqrt(evidence2*norm2).clip(1e-30)
    vectors={'top1_norm2':norm2,'delta_norm2':delta2,'cosine':cosine,
        'max_weight':topmass,'effective_K':effective}
    sums={}
    for key,array in vectors.items():
        sums[key]=float(array.sum());sums['voxel_'+key]=float((array*multiplicity).sum())
    sums['fixed_gate_projection_delta2']=float((.75*gate)**2*(delta2*multiplicity).sum())
    sums['channels']=values.shape[-1]
    return {'queries':len(ids),'projected_voxels':int(multiplicity.sum()),'native_voxels_evaluated':len(grid),
        'query_reaching_projection':int((multiplicity>0).sum()),'sums':sums,
        'query_distribution':{'effective_K':effective,'max_weight':topmass,
            'relative_L2':np.sqrt(delta2/norm2),'evidence_cosine':cosine}}


def summarize_statistics(records):
    queries=sum(r['queries'] for r in records);voxels=sum(r['projected_voxels'] for r in records)
    total=sum(r['native_voxels_evaluated'] for r in records)
    if queries==0:return {'queries':0,'projected_voxels':voxels,'native_voxels_evaluated':total,
        'active_native_voxel_fraction':0.,'fixed_gate_projection_delta_RMS':0.}
    keys=next(r['sums'] for r in records if r['queries']).keys()
    sums={k:sum(r['sums'].get(k,0) for r in records) for k in keys if k!='channels'}
    channels={r['sums']['channels'] for r in records if r['queries']}
    if len(channels)!=1:raise ValueError('native feature channel count changed')
    result={'queries':queries,'projected_voxels':voxels,'native_voxels_evaluated':total,
        'queries_reaching_projection':sum(r['query_reaching_projection'] for r in records),
        'active_native_voxel_fraction':voxels/max(1,total),
        'query_relative_RMS':float(np.sqrt(sums['delta_norm2']/sums['top1_norm2'])),
        'query_mean_evidence_cosine':sums['cosine']/queries,
        'query_mean_effective_K':sums['effective_K']/queries,
        'query_mean_max_weight':sums['max_weight']/queries}
    if voxels:
        result.update(voxel_relative_RMS=float(np.sqrt(sums['voxel_delta_norm2']/sums['voxel_top1_norm2'])),
            voxel_mean_evidence_cosine=sums['voxel_cosine']/voxels,
            voxel_mean_effective_K=sums['voxel_effective_K']/voxels,
            voxel_mean_max_weight=sums['voxel_max_weight']/voxels,
            fixed_gate_projection_delta_RMS=float(np.sqrt(sums['fixed_gate_projection_delta2']/(total*next(iter(channels))))))
    return result


def rank_correlation(left,right):
    left,right=np.asarray(left),np.asarray(right)
    finite=np.isfinite(left)&np.isfinite(right);left,right=left[finite],right[finite]
    def ranks(x):
        _,inverse,count=np.unique(x,return_inverse=True,return_counts=True)
        end=count.cumsum();return ((2*end-count+1)/2)[inverse]
    if len(left)<3 or len(np.unique(left))<2 or len(np.unique(right))<2:return None
    return float(np.corrcoef(ranks(left),ranks(right))[0,1])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve the actual diagnosis')
    import torch
    torch.set_num_threads(2)
    cfg=json.loads((a.root/'configs/soft-topk.json').read_text());args=cfg['validation']['dataset_args']
    def resolve(value):
        for prefix,replacement in args['path_map'].items():
            if value==prefix or value.startswith(prefix+'/'):return Path(replacement+value[len(prefix):])
        return Path(value)
    evaluations={name:[json.loads((a.root/'evaluations'/name/f'noise-{i}/evaluation.json').read_text()) for i in range(3)] for name in ['top1-reference','soft-topk']}
    completed=json.loads((a.root/'firstwave-fixed-final-v1/report.json').read_text())
    if completed['status']!='PASS_FIXED_FINAL_FIRST_WAVE_ONLY':raise ValueError('completed fixed-final report required')
    for name in evaluations:
        for i in range(3):
            path=a.root/'evaluations'/name/f'noise-{i}/evaluation.json'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=completed['input_SHA256'][str(path)]:
                raise ValueError('actual completed evaluation changed')
    ids=[r['sample_id'] for r in evaluations['top1-reference'][0]['per_asset']]
    if len(ids)!=128 or len(set(ids))!=128 or any([r['sample_id'] for r in e['per_asset']]!=ids for es in evaluations.values() for e in es):
        raise ValueError('diagnosis requires the completed same128/3noise final panel')
    rows={r['sample_id']:r for r in [json.loads(s) for s in Path(cfg['validation']['roots_file']).read_text().splitlines()]}
    path=a.root/'runs_ss64/soft-topk/ckpts/deform_conditioner_step0000480.pt'
    gate=float(torch.load(path,map_location='cpu',weights_only=True)['fusion_gate'].detach().tanh())
    temperatures=[.005,float(cfg['trainer']['args']['deform_conditioner']['score_temperature'])]
    if len(set(temperatures))!=2:raise ValueError('expected original and broadened static temperatures')
    records={t:[] for t in temperatures};case_rows=[]
    for sid in ids:
        row=rows[sid]
        with np.load(resolve(row['transport_cache'])) as z:
            if str(z['conditioning_input_sha256'])!=row['conditioning_input_sha256']:raise ValueError('transport RGB identity differs')
            values=z['source_values'].reshape(-1,1024);source=z['source_valid'].reshape(-1).astype(bool)
        with np.load(Path(args['match_root'])/(sid+'.npz')) as z:
            if str(z['q2c_sha256'])!=args['expected_q2c_sha256'] or str(z['conditioning_input_sha256'])!=row['conditioning_input_sha256']:
                raise ValueError('matching head or RGB input differs')
            indices=z['semantic_indices'];logits=z['matching_logits'];valid=z['candidate_valid'].astype(bool)&source[indices]
        with Image.open(resolve(row['target_anchor'])) as im:
            alpha=np.asarray(im.convert('RGBA').resize((512,512),Image.Resampling.NEAREST))[8::16,8::16,3].reshape(-1)>127
        valid &= alpha[:,None] & (source.sum()>=64)
        with np.load(Path(args['projection_root'])/(sid+'.npz')) as z:
            if str(z['schema'])!='deformnext-anchor-projection-v2':raise ValueError('native projection schema differs')
            grid=z['patch_grid']
        case={'sample_id':sid,'source_cluster_id':row['source_cluster_id'],'category':row['category'],'temperatures':{}}
        for t in temperatures:
            one=evidence_statistics(values,indices,logits,valid,grid,t,gate);records[t].append(one)
            case['temperatures'][str(t)]=summarize_statistics([one])
        outcomes={}
        for name,es in evaluations.items():
            current=[next(r for r in e['per_asset'] if r['sample_id']==sid) for e in es]
            sums={k:sum(r[k] for r in current) for k in ['tp','fp','fn']}
            outcomes[name]={'pooled_three_noise_case_IoU':sums['tp']/sum(sums.values()),'counts':sums}
        case['outcomes']=outcomes
        case['soft_minus_top1_case_IoU']=outcomes['soft-topk']['pooled_three_noise_case_IoU']-outcomes['top1-reference']['pooled_three_noise_case_IoU']
        case_rows.append(case)
    t=temperatures[-1];analyzable=[r for r in case_rows if r['temperatures'][str(t)]['queries']]
    correlations={key:rank_correlation([r['temperatures'][str(t)].get(key,np.nan) for r in analyzable],
        [r['soft_minus_top1_case_IoU'] for r in analyzable]) for key in ['query_mean_effective_K','query_relative_RMS','voxel_relative_RMS','query_mean_max_weight']}
    observed=np.array([r['soft_minus_top1_case_IoU'] for r in case_rows])
    report={'status':'PASS_ACTUAL_FIXED128_STATIC_VALUE_DIAGNOSIS','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'assets':len(ids),'source_clusters':len({r['source_cluster_id'] for r in case_rows}),'actual_saved_soft_fusion_gate':gate,
        'temperature_summary':{str(t):summarize_statistics(rs) for t,rs in records.items()},
        'cases_soft_better':int((observed>0).sum()),'cases_soft_worse':int((observed<0).sum()),'cases_equal':int((observed==0).sum()),
        'case_gain_IoU_quantiles':{str(q):float(np.quantile(observed,q)) for q in [0,.1,.25,.5,.75,.9,1]},
        'posthoc_descriptive_Spearman_not_causal_or_significance':correlations,'per_case':case_rows,
        'scope':'Post-hoc fixed128 development only,zero training or generation updates,no GPU use. No depth,topology labels,SS latent or reserved additional121 opened. Values use stored native FP16 tokens with FP64 descriptor statistics,not bit-exact GPU inference.',
        'fixed_gate_projection_delta_scope':'Analytic0.75*g_soft*(E_soft-E_top1),weighted by actual native voxel-to-target patch multiplicity and RMS over the entire native16^3 domain (zero outside alpha/support). Holds gate/native/DINO values fixed;not the full condition difference between separately trained models,not a model quality effect.',
        'case_gain_scope':'Per-case TP/FP/FN pooled over3existing noises. Per-case scatter and rank correlations are descriptive;do not treat shared-source cases as independent statistical tests.',
        'model_selection_or_live_training_changed':False,'full_goal_complete':False,
        'config_SHA256':hashlib.sha256((a.root/'configs/soft-topk.json').read_bytes()).hexdigest(),
        'actual_evaluation_SHA256':{str(a.root/'evaluations'/name/f'noise-{i}/evaluation.json'):hashlib.sha256((a.root/'evaluations'/name/f'noise-{i}/evaluation.json').read_bytes()).hexdigest() for name in evaluations for i in range(3)}}
    write_new(a.output/'report.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for t in temperatures:
        effective=np.sort(np.concatenate([r['query_distribution']['effective_K'] for r in records[t] if r['queries']]))
        axes[0].plot(effective,np.arange(1,len(effective)+1)/len(effective),label=f'T={t:g}')
    axes[0].set(xlabel='Entropy effective candidates',ylabel='CDF over valid RGB queries',xscale='log');axes[0].legend()
    x=[100*r['temperatures'][str(temperatures[-1])].get('voxel_relative_RMS',np.nan) for r in case_rows]
    axes[1].scatter(x,100*observed,s=14,alpha=.65);axes[1].axhline(0,color='gray',lw=.8)
    axes[1].set(xlabel='Projected value relative RMS change (%)',ylabel='Soft - Top1 case IoU (pp)')
    axes[2].hist(100*observed,bins=21,color='#547a9f');axes[2].axvline(0,color='gray',lw=.8)
    axes[2].set(xlabel='Soft - Top1 case IoU (pp)',ylabel='Cases')
    figure.suptitle('Fixed128 post-hoc diagnosis; 3 sampling noises; descriptive, not causal')
    figure.savefig(a.output/'diagnosis.png',dpi=180);figure.savefig(a.output/'diagnosis.pdf');plt.close(figure)
    print(json.dumps({k:v for k,v in report.items() if k in ['status','assets','temperature_summary','cases_soft_better','cases_soft_worse','cases_equal','posthoc_descriptive_Spearman_not_causal_or_significance']}),flush=True)


if __name__=='__main__':main()
