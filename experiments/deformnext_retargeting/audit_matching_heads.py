"""Audit final heads: address accuracy, feature utility, attention and rejection."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .matching_v2_head import MatchingV2Head,masked_probabilities
from .matching_v2_finalize import calibrate
from .train_matching_v2 import load,read_rows


def confidence_summary(probability,labels,threshold):
    probability=np.asarray(probability);labels=np.asarray(labels,dtype=float)
    accepted=probability>=threshold;ece=0.
    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        mask=(probability>=lo)&((probability<hi) if hi<1 else (probability<=hi))
        if mask.any():ece+=mask.mean()*abs(probability[mask].mean()-labels[mask].mean())
    return {'brier':float(np.mean((probability-labels)**2)),'ece15':float(ece),
        'precision':float(labels[accepted].mean()) if accepted.any() else None,
        'coverage':float(accepted.mean()),'accepted':int(accepted.sum()),'evaluated_queries':len(labels),
        'threshold':threshold,'confidence_label':'selected exact correspondence correctness, not attention max'}


def audit(arm,a):
    directory=a.root/'heads'/arm;history=json.loads((directory/'history.json').read_text())
    if history[-1]['epoch']!=7:raise ValueError('audit requires fixed-final7 epochs')
    state=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
    model=MatchingV2Head().cuda().eval();model.load_state_dict(state['model'],strict=True)
    args=SimpleNamespace(proposals=a.root/'proposals',labels=a.labels,file4_mirror=a.file4_mirror,
        topk=state['provenance']['topk'])
    cal=json.loads((directory/'calibration.json').read_text())
    temp=float(min(cal['feature_error'],key=lambda t:cal['feature_error'][t]))
    reliability=None if state['provenance']['arm'] in {'h0','h1'} else calibrate(cal['confidence_logits'],cal['usable'])
    rows=read_rows(a.panels/'head_validation.jsonl')
    totals=dict(foreground=0,visible=0,proposal_hit=0,correct=0,near1=0,unobservable=0,uncertain=0,queries=0)
    sums=dict(feature_error=0.,uniform_feature_error=0.,centered_feature_error=0.,
        centered_uniform_error=0.,attention_max=0.,effective_candidates=0.,entropy=0.)
    topview=np.zeros(3);massview=np.zeros(3);p_use=[];labels=[];per_asset=[]
    with torch.inference_mode():
        for n,row in enumerate(rows):
            q,source,values,index,raw,valid,positive,visible,fg,pos,trusted=load(row,args,'cuda',True)
            with np.load(a.file4_mirror/row['transport_cache'].removeprefix('/file4/')) as z:
                source_valid=torch.from_numpy(z['source_valid'].copy()).cuda().flatten().bool()
            center=values[source_valid].mean(0) if source_valid.any() else values.mean(0)
            local=dict(foreground=int(fg.sum()),visible=int(visible.any(-1).sum()),proposal_hit=int(positive.any(-1).sum()),
                correct=0,near1=0,unobservable=int((fg&~visible.any(-1)).sum()),uncertain=int((fg&~trusted).sum()),queries=0)
            for chunk in torch.where(fg&valid.any(-1))[0].split(32):
                logits,confidence=model(q[chunk],source[index[chunk]],raw[chunk],valid[chunk])
                weights=masked_probabilities(logits,valid[chunk],temp);chosen=logits.argmax(-1)
                address=index[chunk].gather(1,chosen[:,None])[:,0]
                correct=positive[chunk].gather(1,chosen[:,None])[:,0]
                in_view=address[:,None]//1024==pos[chunk]//1024
                delta_x=(address[:,None]%32-pos[chunk]%32).abs()
                delta_y=((address[:,None]%1024)//32-(pos[chunk]%1024)//32).abs()
                near=(in_view&visible[chunk]&(delta_x<=1)&(delta_y<=1)).any(-1)
                local['correct']+=int(correct.sum());local['near1']+=int(near.sum());local['queries']+=len(chunk)
                uniform=valid[chunk].float()/valid[chunk].sum(-1,keepdim=True).clamp_min(1)
                candidate_values=values[index[chunk]]
                predicted=(candidate_values*weights[...,None]).sum(1)
                average=(candidate_values*uniform[...,None]).sum(1)
                target=(values[pos[chunk]]*visible[chunk,:,None]).sum(1)/visible[chunk].sum(-1).clamp_min(1)[:,None]
                hit=positive[chunk].any(-1)
                for key,x,y in [('feature_error',predicted,target),('uniform_feature_error',average,target),
                    ('centered_feature_error',predicted-center,target-center),('centered_uniform_error',average-center,target-center)]:
                    sums[key]+=float((1-F.cosine_similarity(x,y))[hit].sum())
                sums['attention_max']+=float(weights.max(-1).values.sum())
                sums['effective_candidates']+=float((1/weights.square().sum(-1).clamp_min(1e-12)).sum())
                sums['entropy']+=float(-(weights*weights.clamp_min(1e-12).log()).sum())
                view=(index[chunk]//1024).long()
                mass=torch.zeros(len(chunk),3,device='cuda');mass.scatter_add_(1,view,weights)
                massview+=mass.sum(0).cpu().numpy();topview+=np.bincount((address//1024).cpu().numpy(),minlength=3)
                if reliability is not None:
                    trustworthy=trusted[chunk]
                    probability=torch.sigmoid(reliability['slope']*confidence[:,0]+reliability['bias'])
                    p_use.extend(probability[trustworthy].cpu().tolist());labels.extend(correct[trustworthy].cpu().tolist())
            for k,v in local.items():totals[k]+=v
            per_asset.append({'sample_id':row['sample_id'],'source_cluster_id':row['source_cluster_id'],**local})
            if n%100==0:print(json.dumps({'arm':arm,'assets':n+1}),flush=True)
    result={'arm':arm,'assets':len(rows),'epoch':7,'topk':args.topk,'temperature':temp,'temperature_selection':'train-internal calibration only',**totals,
        'top1_accuracy':totals['correct']/totals['visible'],'near1_chebyshev_accuracy':totals['near1']/totals['visible'],
        'proposal_recall':totals['proposal_hit']/totals['visible'],
        **{k:v/max(1,totals['proposal_hit'] if 'error' in k else totals['queries']) for k,v in sums.items()},
        'top1_view_fraction':(topview/max(1,topview.sum())).tolist(),'attention_view_fraction':(massview/max(1,massview.sum())).tolist(),
        'confidence':confidence_summary(p_use,labels,reliability['threshold']) if reliability is not None else None,
        'confidence_calibration':reliability,'per_asset':per_asset,
        'utility_scope':'DINO feature errors on proposal-hit queries; this is not SS generation improvement'}
    del model;torch.cuda.empty_cache();return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','panels','labels','file4-mirror','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--arms',required=True);p.add_argument('--wandb',action='store_true')
    a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_device(0)
    results=[audit(arm,a) for arm in a.arms.split(',')]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'heads':results},indent=2)+'\n')
    if a.wandb:
        import wandb
        run=wandb.init(entity='youngwoojeon-unist',project='pixal3d-deformnext-retargeting',id='ss-matching-v2-head-audit',resume='allow',dir=str(a.output.parent))
        for row in results:
            payload={row['arm']+'/'+k:v for k,v in row.items() if isinstance(v,(float,int))}
            payload.update({row['arm']+'/confidence/'+k:v for k,v in (row['confidence'] or {}).items() if isinstance(v,(float,int))})
            run.log(payload)
        run.finish()


if __name__=='__main__':main()
