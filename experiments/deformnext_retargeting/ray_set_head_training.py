"""Train only the five ray utility tensors with separate multi-positive loss labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from pixal3d.modules.task_matching_attention import TaskCandidateAttention


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def ray_set_logit_loss(logits,relation_queries,positives):
    """Stable per-target positive-set loss, averaged within ray then across rays."""
    if (logits.ndim!=2 or positives.shape!=(len(relation_queries),logits.shape[1])
            or positives.dtype!=torch.bool or relation_queries.dtype!=torch.long):
        raise ValueError('separate loss-only positive sets required')
    covered=positives.any(-1)
    stats=dict(observable_relations=len(positives),covered_relations=int(covered.sum()),
        proposal_misses=int((~covered).sum()),supervised_queries=int(relation_queries[covered].unique().numel()))
    if not covered.any():return logits.sum()*0,stats
    queries=relation_queries[covered]
    logp=torch.log_softmax(logits.float(),dim=-1)
    loss=-logp[queries].masked_fill(~positives[covered],-torch.inf).logsumexp(-1)
    totals=logits.new_zeros(len(logits),dtype=torch.float32).index_add(0,queries,loss)
    counts=torch.bincount(queries,minlength=len(logits))
    return (totals[counts>0]/counts[counts>0]).mean(),stats


def load_asset(root,row):
    with np.load(root/row['inputs_relative_path'],allow_pickle=False) as archive:
        if str(archive['schema'])!='deformnext-ray-set-inputs-v1' or str(archive['sample_id'])!=row['sample_id']:
            raise ValueError('input-cache identity differs')
        forbidden={'positive_indices','positive_visible','positive_near1_bits','relation_query_indices','teacher_input_sha256'}
        if forbidden&set(archive.files):raise ValueError('teacher leaked into model-input archive')
        inputs={key:archive[key].copy() for key in ['native_query','context_query','scalar_prior_logits',
            'candidate_indices','candidate_valid','source_value_bank']}
    with np.load(root/row['labels_relative_path'],allow_pickle=False) as archive:
        if str(archive['schema'])!='deformnext-ray-set-loss-labels-v1' or str(archive['sample_id'])!=row['sample_id']:
            raise ValueError('loss-label cache identity differs')
        if str(archive['teacher_input_sha256'])!=row['GT_teacher_fingerprint']:
            raise ValueError('teacher fingerprint differs')
        relation=archive['relation_query_indices'].astype(np.int64)
        labels={kind:np.unpackbits(archive['positive_'+kind+'_bits'],axis=1,count=64,bitorder='little').astype(bool)
            for kind in ['exact','near1']}
    n=len(inputs['native_query'])
    if (inputs['native_query'].shape!=(n,1024) or inputs['context_query'].shape!=(n,1024)
            or inputs['candidate_indices'].shape!=(n,64) or inputs['candidate_valid'].shape!=(n,64)
            or inputs['scalar_prior_logits'].shape!=(n,64) or inputs['source_value_bank'].shape!=(3072,1024)
            or not inputs['candidate_valid'].all() or (relation<0).any() or (relation>=n).any()):
        raise ValueError('actual complete native64 input/label dimensions differ')
    if not np.all(relation[:-1]<=relation[1:]):raise ValueError('ordered per-ray label CSR required')
    labels['relation']=relation;labels['pointer']=np.searchsorted(relation,np.arange(n+1))
    labels['eligible']=np.unique(relation[labels['near1'].any(-1)])
    return inputs,labels


def make_batch(root,rows,queries_per_asset,rng):
    fields={key:[] for key in ['native_query','context_query','candidates','valid','prior']}
    relation=[];positive=[];offset=0
    for row in rows:
        x,y=load_asset(root,row);available=y['eligible']
        if not len(available):raise ValueError('selected training asset has no supervised ray')
        chosen=rng.choice(available,size=queries_per_asset,replace=len(available)<queries_per_asset)
        fields['native_query'].append(x['native_query'][chosen]);fields['context_query'].append(x['context_query'][chosen])
        fields['candidates'].append(x['source_value_bank'][x['candidate_indices'][chosen].astype(np.int64)])
        fields['valid'].append(x['candidate_valid'][chosen]);fields['prior'].append(x['scalar_prior_logits'][chosen])
        lengths=y['pointer'][chosen+1]-y['pointer'][chosen]
        relation.append(np.repeat(np.arange(queries_per_asset)+offset,lengths))
        positive.append(np.concatenate([y['near1'][y['pointer'][q]:y['pointer'][q+1]] for q in chosen]))
        offset+=queries_per_asset
    batch={key:torch.from_numpy(np.concatenate(value)) for key,value in fields.items()}
    batch['relation']=torch.from_numpy(np.concatenate(relation).astype(np.int64))
    batch['positive']=torch.from_numpy(np.concatenate(positive))
    return batch


def forward_head(head,batch,device='cuda'):
    # Explicit argument list: teacher masks/indices never reach head.forward.
    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
        return head(batch['native_query'].to(device,dtype=torch.float32),
            batch['context_query'].to(device,dtype=torch.float32),
            batch['candidates'].to(device,dtype=torch.float32),
            batch['valid'].to(device),batch['prior'].to(device,dtype=torch.float32))


def initialize(conditioner):
    torch.manual_seed(101);torch.cuda.manual_seed_all(101)
    head=TaskCandidateAttention(content_mode='residual-zero',initial_content_scale=16.).cuda()
    source=torch.load(conditioner,map_location='cpu',weights_only=True)
    state={key[len('task_attention.'):]:value for key,value in source.items() if key.startswith('task_attention.')}
    if len(state)!=5:raise ValueError('exact five-tensor Full utility initialization required')
    head.load_state_dict(state,strict=True)
    if len(list(head.parameters()))!=5:raise ValueError('only five utility parameters may train')
    return head,source


def evaluate(head,root,rows):
    head.eval();total_loss=0.;supervised=0;observable=0;counts={kind:dict(hit1=0,hit64=0,mass=0.,union_mass=0.,rays=0) for kind in ['exact','near1']}
    with torch.no_grad():
        for row in rows:
            x,y=load_asset(root,row);probabilities=[];logits=[]
            for start in range(0,len(x['native_query']),64):
                take=slice(start,start+64)
                batch={key:torch.from_numpy(value) for key,value in dict(native_query=x['native_query'][take],context_query=x['context_query'][take],candidates=x['source_value_bank'][x['candidate_indices'][take].astype(np.int64)],valid=x['candidate_valid'][take],prior=x['scalar_prior_logits'][take]).items()}
                _,p,l=forward_head(head,batch);probabilities.append(p.cpu());logits.append(l.cpu())
            p=torch.cat(probabilities) if probabilities else torch.empty(0,64);l=torch.cat(logits) if logits else torch.empty(0,64)
            relation=torch.from_numpy(y['relation']);near=torch.from_numpy(y['near1'])
            loss,stats=ray_set_logit_loss(l,relation,near)
            total_loss+=float(loss)*stats['supervised_queries'];supervised+=stats['supervised_queries'];observable+=len(relation)
            for kind in counts:
                positive=torch.from_numpy(y[kind]);used=torch.zeros(len(p),dtype=torch.bool);used[relation]=True
                union=torch.zeros_like(p,dtype=torch.int32).index_add(0,relation,positive.int())>0
                if len(relation):
                    chosen=p[relation].argmax(-1);counts[kind]['hit1']+=int(positive.gather(1,chosen[:,None]).sum())
                counts[kind]['hit64']+=int(positive.any(-1).sum());counts[kind]['mass']+=float((p[relation]*positive).sum())
                counts[kind]['union_mass']+=float((p[used]*union[used]).sum());counts[kind]['rays']+=int(used.sum())
    head.train()
    return dict(loss=total_loss/max(1,supervised),supervised_queries=supervised,observable_relations=observable,
        **{kind:dict(top1_coverage=v['hit1']/max(1,observable),proposal64_coverage=v['hit64']/max(1,observable),
                    mean_target_patch_mass=v['mass']/max(1,observable),mean_ray_union_mass=v['union_mass']/max(1,v['rays'])) for kind,v in counts.items()})


def source_guard(cache_root,own_sha):
    if sha(__file__)!=own_sha:raise RuntimeError('training implementation changed during execution')
    for path,digest in read(cache_root/'source-contract.json')['source_SHA256'].items():
        if sha(path)!=digest:raise RuntimeError('cached-input source changed: '+path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['profile','train'],required=True)
    parser.add_argument('--cache-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--conditioner',type=Path,required=True);parser.add_argument('--microqueries',type=int,default=8192)
    parser.add_argument('--queries-per-asset',type=int,default=512);parser.add_argument('--epochs',type=int,default=7)
    parser.add_argument('--cache-receipt',type=Path);parser.add_argument('--campaign-root',type=Path)
    parser.add_argument('--profile-proof',type=Path);parser.add_argument('--wandb',action='store_true')
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or '0' not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('only explicitly assigned physicalGPU0 auxiliary training allowed')
    if args.microqueries%args.queries_per_asset or args.epochs!=7:raise ValueError('whole assets,seven declared head epochs required')
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.2,0)
    torch.use_deterministic_algorithms(True)
    own_sha=sha(__file__);source_guard(args.cache_root,own_sha)
    index=args.cache_root/('cache-index.jsonl' if args.mode=='train' else 'cache-index.partial.jsonl')
    rows=[json.loads(line) for line in index.read_text().splitlines()]
    train=[row for row in rows if row['role']=='train' and row['usable_queries']>0];cal=[row for row in rows if row['role']=='calibration']
    if not train:raise ValueError('no supervised training rays')
    args.output.mkdir(parents=True,exist_ok=False)
    head,base=initialize(args.conditioner);optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01)
    if args.mode=='profile':
        rng=np.random.default_rng(101);chosen=[train[i%len(train)] for i in range(args.microqueries//args.queries_per_asset)]
        batch=make_batch(args.cache_root,chosen,args.queries_per_asset,rng)
        torch.cuda.reset_peak_memory_stats();started=time.monotonic();optimizer.zero_grad(set_to_none=True)
        _,_,logits=forward_head(head,batch);loss,stats=ray_set_logit_loss(logits,batch['relation'].cuda(),batch['positive'].cuda())
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();torch.cuda.synchronize()
        if not math.isfinite(float(loss)) or not math.isfinite(float(norm)) or len(optimizer.state)!=5:
            raise ValueError('actual five-tensor optimizer/backward proof failed')
        result=dict(status='PASS_ACTUAL_AUXILIARY_HEAD_FORWARD_BACKWARD_OPTIMIZER_MEMORY',microqueries=args.microqueries,
            queries_per_asset=args.queries_per_asset,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            elapsed_seconds=time.monotonic()-started,trainable_tensors=5,optimizer_states=len(optimizer.state),
            temporary_optimizer_updates=1,profile_model_discarded=True,SS_optimizer_updates=0,
            GT_only_loss=True,loss=float(loss),gradient_norm=float(norm),training_source_SHA256=own_sha,
            conditioner_SHA256=sha(args.conditioner),scope='Actualcache-backed capacity proof;repeatedassetsifneeded;not the7epoch scientific training run.')
        write_new(args.output/'profile.json',result);print(json.dumps(result),flush=True);return
    done=read(args.cache_root/'completed.json')
    if done['status']!='CACHE_COMPLETE' or len(rows)!=544 or len(cal)!=32 or len(train)<448:
        raise ValueError('complete selected544 cache and at least448 supervised train assets required')
    closure=read(args.cache_receipt)
    if closure.get('status')!='PASS' or closure.get('local_equals_actual_file4_SHA256') is not True:
        raise ValueError('actual file4 cache closure required before training')
    closed={entry['relative_path']:entry for entry in closure['files']}
    for path in [index,args.cache_root/'completed.json',*[args.cache_root/row[k+'_relative_path'] for row in rows for k in ['inputs','labels']]]:
        key=str(path.relative_to(args.campaign_root.parent))
        if key not in closed or sha(path)!=closed[key]['sha256']:raise ValueError('closed data-cache bytes differ')
    fit=read(args.profile_proof)
    if (fit.get('status')!='PASS_ACTUAL_AUXILIARY_HEAD_FORWARD_BACKWARD_OPTIMIZER_MEMORY'
            or fit['microqueries']!=args.microqueries or fit['training_source_SHA256']!=own_sha
            or fit['conditioner_SHA256']!=sha(args.conditioner)):
        raise ValueError('same-source actual query microbatch profile required')
    run=None
    if args.wandb:
        import wandb
        run=wandb.init(entity='youngwoojeon-unist',project='pixal3d-deformnext-retargeting',id='ss-ray-set-head-v1-s101-512-e7',
            name='Ray set supervision utility5 train512 epoch7',resume='never',dir=str(args.output),config=dict(epochs=7,
                microqueries=args.microqueries,queries_per_asset=args.queries_per_asset,train_assets=len(train),calibration_assets=32,SS_updates=0))
    history=[];updates=0;started=time.monotonic();per_step=args.microqueries//args.queries_per_asset
    for epoch in range(8):
        losses=[]
        if epoch:
            rng=np.random.default_rng(101+epoch);order=rng.permutation(len(train));head.train()
            for begin in range(0,len(order),per_step):
                batch=make_batch(args.cache_root,[train[i] for i in order[begin:begin+per_step]],args.queries_per_asset,rng)
                optimizer.zero_grad(set_to_none=True);_,_,logits=forward_head(head,batch)
                loss,stats=ray_set_logit_loss(logits,batch['relation'].cuda(),batch['positive'].cuda())
                if not torch.isfinite(loss):raise ValueError('nonfinite set-supervision loss')
                loss.backward();gradient=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                if not torch.isfinite(gradient):raise ValueError('nonfinite head gradient')
                optimizer.step();updates+=1;losses.append(float(loss.detach()))
                if updates%8==0:
                    progress=dict(status='TRAINING',epoch=epoch,optimizer_updates=updates,mean_loss=float(np.mean(losses)),
                        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=time.monotonic()-started)
                    (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
                source_guard(args.cache_root,own_sha)
                del logits,loss,batch
        metrics=evaluate(head,args.cache_root,cal)
        row=dict(epoch=epoch,optimizer_updates=updates,training_loss=float(np.mean(losses)) if losses else None,calibration=metrics,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=time.monotonic()-started)
        history.append(row)
        with (args.output/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if run:
            run.log({'head/epoch':epoch,'head/optimizer_updates':updates,'head/calibration_loss':metrics['loss'],
                **{f'head/calibration_{kind}_{key}':value for kind in ['exact','near1'] for key,value in metrics[kind].items()},
                **({'head/training_loss':row['training_loss']} if losses else {})},step=epoch)
        path=args.output/f'head-epoch{epoch:02d}.pt'
        torch.save(dict(head={key:value.detach().cpu() for key,value in head.state_dict().items()},
            optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,source_SHA256=own_sha),path)
        print(json.dumps(row),flush=True)
    trained=head.state_dict();export={key:value.clone() for key,value in base.items()}
    for key,value in trained.items():export['task_attention.'+key]=value.detach().cpu()
    changed=[key for key in base if not torch.equal(base[key],export[key])]
    if not changed or any(not key.startswith('task_attention.') for key in changed) or len(changed)>5:
        raise ValueError('export changed a frozen R/fusion/matching tensor')
    final=args.output/'full-conditioner-head-epoch07.pt';torch.save(export,final)
    if run:run.finish()
    source_guard(args.cache_root,own_sha)
    result=dict(status='PASS_FIXED_SEVEN_EPOCH_RAY_SET_UTILITY_HEAD_TRAINING',epochs=7,optimizer_updates=updates,
        train_selected_assets=512,train_supervised_assets=len(train),calibration_assets=32,microqueries=args.microqueries,
        queries_per_asset=args.queries_per_asset,accumulation=1,changed_conditioner_tensors=changed,
        all_R_fusion_matching_tensors_exact=True,SS_LoRA_updates=0,GT_in_forward=False,
        conditioner_export=str(final),conditioner_export_SHA256=sha(final),source_SHA256=own_sha,
        final_calibration=history[-1]['calibration'],peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        elapsed_seconds=time.monotonic()-started,model_adopted=False,
        scope='Auxiliarysupervisionpilotonly;fixedepoch7,nobestepochselection;SSqualitymustbeevaluatedwithactualfrozenpipeline.')
    write_new(args.output/'completed.json',result);(args.output/'progress.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
