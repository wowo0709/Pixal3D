"""Matched-budget H1/H2/H3 training on cached DeformNeXt FRAME descriptors."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .matching_v2_head import MatchingV2Head, SCHEMA, masked_probabilities, ranking_loss


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def atomic_save(value, path):
    tmp = path.with_name('.' + path.name + '.tmp'); torch.save(value, tmp); os.replace(tmp, path)


def load(row, a, device, with_values):
    path = a.file4_mirror / row['transport_cache'].removeprefix('/file4/')
    def tensor(array):
        return torch.from_numpy(np.ascontiguousarray(array)).pin_memory().to(device, non_blocking=True)
    with np.load(path) as z:
        if str(z['conditioning_input_sha256']) != row['conditioning_input_sha256']: raise ValueError('RGB cache fingerprint differs')
        q = tensor(z['target_keys'][..., :1024]).float()
        source = tensor(z['source_keys'][..., :1024]).float().flatten(0, 1)
        values = tensor(z['source_values']).float().flatten(0, 1) if with_values else None
    with np.load(a.proposals / (row['sample_id'] + '.npz')) as z:
        if str(z['input_sha256']) != row['conditioning_input_sha256']: raise ValueError('proposal input differs')
        index = tensor(z['indices'][:, :a.topk].astype(np.int64)).long()
        raw = tensor(z['cosine'][:, :a.topk]).float()
        valid = tensor(z['valid'][:, :a.topk]).bool()
    with np.load(a.labels / (row['sample_id'] + '.npz')) as z:
        if str(z['sample_id']) != row['sample_id'] or str(z['split']) != row['split']: raise ValueError('teacher identity differs')
        pos = tensor(z['positive_indices']).long(); visible = tensor(z['positive_visible']).bool()
        foreground = tensor(z['target_foreground']).bool()
        source_masks_certain = bool(np.min(z['input_mask_iou'][1:]) >= .98)
    positive = ((index[..., None] == pos[:, None]) & visible[:, None]).any(-1) & valid
    trusted = visible.any(-1) | source_masks_certain
    return q, source, values, index, raw, valid, positive, visible, foreground, pos, trusted


def evaluate(model, rows, a, temperatures=(.005, .01, .02, .05, .1)):
    model.eval(); count = dict(foreground=0, visible=0, eligible=0, correct=0)
    feature_error = {str(t): 0. for t in temperatures}
    confidence, target, visibility, correct_list, candidate_hit = [], [], [], [], []
    with torch.no_grad():
        for row in rows:
            q, source, values, idx, raw, valid, positive, vis, fg, pos, trusted = load(row, a, 'cuda', True)
            selected = torch.where(fg & valid.any(-1))[0]
            count['foreground'] += int(fg.sum()); count['visible'] += int(vis.any(-1).sum()); count['eligible'] += int(positive.any(-1).sum())
            for chunk in selected.split(32):
                logits, conf = model(q[chunk], source[idx[chunk]], raw[chunk], valid[chunk])
                eligible = positive[chunk].any(-1)
                chosen = logits.argmax(-1)
                correct = positive[chunk].gather(1, chosen[:, None])[:, 0]
                count['correct'] += int(correct.sum())
                reliable = trusted[chunk]
                confidence.extend(conf[reliable, 0].cpu().tolist()); target.extend(correct[reliable].cpu().tolist())
                candidate_hit.extend(eligible[reliable].cpu().tolist())
                visibility.extend(vis[chunk][reliable].any(-1).cpu().tolist()); correct_list.extend(correct[reliable].cpu().tolist())
                gt = (values[pos[chunk]] * vis[chunk, :, None]).sum(1) / vis[chunk].sum(-1).clamp_min(1)[:, None]
                for temp in temperatures:
                    weights = masked_probabilities(logits, valid[chunk], temp)
                    pred = (values[idx[chunk]] * weights[..., None]).sum(1)
                    error = 1 - F.cosine_similarity(pred, gt)
                    feature_error[str(temp)] += float(error[eligible].sum())
    model.train()
    return {**count, 'top1_accuracy': count['correct']/max(1,count['visible']),
            'proposal_recall':count['eligible']/max(1,count['visible']),
            'feature_error':{t:v/max(1,count['eligible']) for t,v in feature_error.items()},
            'confidence_logits':confidence,'usable':target,'visible_labels':visibility,'candidate_hit':candidate_hit,'selected_correct':correct_list}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panels','proposals','labels','file4-mirror','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=['h1','h2','h3'],required=True)
    p.add_argument('--epochs',type=int,default=7);p.add_argument('--seed',type=int,default=42017)
    p.add_argument('--wandb',action='store_true')
    p.add_argument('--wandb-id')
    a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed)
    proposal=json.loads((a.proposals.parent/'proposal-audit.json').read_text());a.topk=proposal['selected_topk']
    train,calibration,validation=[read_rows(a.panels/(name+'.jsonl')) for name in ('head_train','head_calibration','head_validation')]
    model=MatchingV2Head().cuda()
    if a.arm=='h1':model.confidence.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    a.output.mkdir(parents=True,exist_ok=True)
    provenance={'schema':SCHEMA,'arm':a.arm,'seed':a.seed,'topk':a.topk,'epochs':a.epochs,
                'head_train_sha256':hashlib.sha256((a.panels/'head_train.jsonl').read_bytes()).hexdigest(),
                'ranking_temperature':1.,'feature_temperature':.05,'feature_lambda':1.,'query_budget':'up to32 eligible plus32 other foreground per asset',
                'uncertain_visibility_policy':'exclude unobservable NULL labels when any source raster/alpha IoU<0.98',
                'input':'RGB camera8 M1 + camera9/10/11 M0; frozen Omega L12 FRAME 1024',
                'train_assets':len(train),'calibration_assets':len(calibration),'validation_assets':len(validation)}
    (a.output/'recipe.json').write_text(json.dumps(provenance,indent=2)+'\n')
    latest=a.output/'training-state.pt';step=start_epoch=0;history=[]
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['provenance']!=provenance:raise ValueError('head resume provenance differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for state_value in optimizer.state.values():
            for key,value in state_value.items():
                if isinstance(value,torch.Tensor):state_value[key]=value.cuda()
        step=state['step'];start_epoch=state['epoch'];rng.bit_generator.state=state['numpy_rng_state'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);history=state['history']
    run=None
    if a.wandb:
        import wandb
        run=wandb.init(project='pixal3d-deformnext-retargeting',entity='youngwoojeon-unist',
                       id=a.wandb_id or ('ss-matching-v2-'+a.arm),name='SS matching-v2 '+a.arm,resume='allow',dir=str(a.output),config=provenance)
    for epoch in range(start_epoch+1,a.epochs+1):
        began=time.monotonic();losses=[];model.train()
        for offset,index in enumerate(rng.permutation(len(train))):
            row=train[int(index)]
            q,source,values,idx,raw,valid,positive,vis,fg,pos,trusted=load(row,a,'cuda',a.arm=='h3')
            eligible=positive.any(-1)
            primary=torch.where(eligible)[0].cpu().numpy();other=torch.where(fg&~eligible&valid.any(-1)&trusted)[0].cpu().numpy()
            chosen=np.concatenate((rng.permutation(primary)[:32],rng.permutation(other)[:32]))
            if not len(chosen):continue
            chunk=torch.as_tensor(chosen,device='cuda')
            logits,conf=model(q[chunk],source[idx[chunk]],raw[chunk],valid[chunk])
            ranking=ranking_loss(logits,positive[chunk],eligible[chunk]);loss=ranking
            null_loss=ranking.detach()*0;task_loss=null_loss
            if a.arm!='h1':
                selected_correct=positive[chunk].gather(1,logits.detach().argmax(-1,keepdim=True))[:,0]
                labels=torch.stack((selected_correct,vis[chunk].any(-1),eligible[chunk]),-1).float()
                null_loss=F.binary_cross_entropy_with_logits(conf,labels);loss=loss+null_loss
            if a.arm=='h3' and eligible[chunk].any():
                weights=masked_probabilities(logits,valid[chunk],.05)
                predicted=(values[idx[chunk]]*weights[...,None]).sum(1)
                teacher=(values[pos[chunk]]*vis[chunk,:,None]).sum(1)/vis[chunk].sum(-1).clamp_min(1)[:,None]
                task_loss=(1-F.cosine_similarity(predicted,teacher))[eligible[chunk]].mean();loss=loss+task_loss
            if not torch.isfinite(loss):raise ValueError('nonfinite matching-v2 loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1
            losses.append(float(loss.detach()))
            if step%100==0:
                record={'epoch':epoch,'step':step,'loss':float(loss.detach()),'ranking':float(ranking.detach()),'null':float(null_loss.detach()),'task':float(task_loss.detach()),'assets_done':offset+1,'sec_per_asset':(time.monotonic()-began)/(offset+1)}
                print(json.dumps(record),flush=True)
                if run:run.log({f'train/{k}':v for k,v in record.items() if k!='step'},step=step)
        cal=evaluate(model,calibration,a);val=evaluate(model,validation,a)
        if a.arm=='h1':
            for result in (cal,val):result['confidence_untrained']=True
        epoch_record={'epoch':epoch,'step':step,'train_loss':float(np.mean(losses)),'calibration':{k:v for k,v in cal.items() if not isinstance(v,list)},'validation':{k:v for k,v in val.items() if not isinstance(v,list)},'seconds':time.monotonic()-began}
        history.append(epoch_record);print(json.dumps(epoch_record),flush=True)
        state={'provenance':provenance,'epoch':epoch,'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'numpy_rng_state':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'history':history}
        atomic_save(state,latest);atomic_save({'schema':SCHEMA,'provenance':provenance,'model':model.state_dict()},a.output/f'epoch-{epoch:02d}.pt')
        (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if run:run.log({'epoch':epoch,'val/top1_accuracy':val['top1_accuracy'],'val/proposal_recall':val['proposal_recall'],'train/epoch_loss':epoch_record['train_loss']},step=step)
        if epoch==a.epochs:
            (a.output/'calibration.json').write_text(json.dumps(cal)+'\n');(a.output/'validation.json').write_text(json.dumps(val)+'\n')
            atomic_save({'schema':SCHEMA,'provenance':provenance,'model':model.state_dict()},a.output/'final.pt')
    if run:run.finish()


if __name__=='__main__':main()
