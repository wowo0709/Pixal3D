"""Preaggregate frozen reference R9 evidence without changing its attention.

Only RGB alpha, frozen values and existing inference match caches are read.
Top64 candidate values are expanded for a bounded query chunk, never SS batch96.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .matching_v2_finalize import sha


def load(row,a):
    resolve=lambda v:a.file4_mirror/v.removeprefix('/file4/')
    with np.load(resolve(row['transport_cache'])) as z:
        if str(z['conditioning_input_sha256'])!=row['conditioning_input_sha256']:raise ValueError('reference input fingerprint differs')
        values=torch.from_numpy(z['source_values'].copy()).cuda().float()
        source_valid=torch.from_numpy(z['source_valid'].copy()).cuda().bool()
    from pixal3d.modules.r9_matching import MATCHER_CONTRACT,ATTENTION_CONTRACT
    with np.load(a.reference_matches/(row['sample_id']+'.npz')) as z:
        if (str(z['schema'])!='deformnext-match-r9-v5' or str(z['aggregation'])!='global-categorical-v1'
            or int(z['topk'])!=64 or z['semantic_indices'].shape!=(1024,64)
            or str(z['conditioning_input_sha256'])!=row['conditioning_input_sha256'] or str(z['q2c_sha256'])!=a.head_sha256
            or json.loads(str(z['matcher_contract']))!=MATCHER_CONTRACT or json.loads(str(z['attention_contract']))!=ATTENTION_CONTRACT):
            raise ValueError('reference matching contract differs')
        indices=torch.from_numpy(z['semantic_indices'].astype(np.int64)).cuda()
        logits=torch.from_numpy(z['matching_logits'].copy()).cuda().float()
        valid=torch.from_numpy(z['candidate_valid'].copy()).cuda().bool()
    with Image.open(resolve(row['target_anchor'])) as im:
        alpha=np.asarray(im.convert('RGBA').resize((512,512),Image.Resampling.NEAREST))[8::16,8::16,3].copy().flatten()>127
    target_valid=torch.from_numpy(alpha).cuda()
    valid=valid&source_valid.flatten()[indices]&target_valid[:,None]&(source_valid.sum()>=64)
    return values,source_valid,indices,logits,valid,target_valid


@torch.inference_mode()
def evidence(loaded):
    from pixal3d.modules.r9_matching import categorical_weights
    values,_,indices,logits,valid,_=loaded;flat=values.flatten(0,1)
    soft=values.new_zeros(1024,1024);top1=torch.zeros_like(soft)
    usable=valid.any(-1)
    for q in torch.where(usable)[0].split(32):
        weights=categorical_weights(logits[q],valid[q],.005)
        soft[q]=(flat[indices[q]]*weights[...,None]).sum(1)
        chosen=logits[q].masked_fill(~valid[q],-torch.inf).argmax(-1)
        top1[q]=flat[indices[q].gather(1,chosen[:,None])[:,0]]
    return {'evidence_soft':soft.cpu().numpy(),'evidence_top1':top1.half().cpu().numpy(),
        'p_use':np.ones(1024,np.float32),'cached_valid':usable.cpu().numpy()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','root','reference-matches','file4-mirror'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--head-sha256',required=True);p.add_argument('--mode',choices=['cache','audit'],default='cache')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_device(0)
    contract=a.root/'reference-matches/contract.json';digest=sha(contract)
    if json.loads(contract.read_text())['head_sha256']!=a.head_sha256:raise ValueError('reference head identity differs')
    rows=[json.loads(l) for l in a.manifest.read_text().splitlines() if l]
    if a.mode=='cache':
        assigned=rows[a.rank::a.world_size]
        for n,row in enumerate(assigned):
            path=contract.parent/(row['sample_id']+'.npz')
            if path.exists():
                with np.load(path) as z:
                    if str(z['contract_sha256'])!=digest or str(z['conditioning_input_sha256'])!=row['conditioning_input_sha256']:
                        raise ValueError('stale reference preaggregation')
                continue
            arrays=evidence(load(row,a));temp=path.with_name('.'+path.name+'.tmp.npz')
            np.savez(temp,schema='deformnext-matching-v2-cache-v1',contract_sha256=digest,
                sample_id=row['sample_id'],split=row['split'],conditioning_input_sha256=row['conditioning_input_sha256'],**arrays)
            os.replace(temp,path)
            if n%100==0:print(json.dumps({'rank':a.rank,'completed':n+1,'assigned':len(assigned)}),flush=True)
        (a.root/f'reference-cache-rank{a.rank}.json').write_text(json.dumps({'status':'PASS','assigned':len(assigned),'contract_sha256':digest})+'\n')
    else:
        from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
        model=DeformationAwareConditioner(arm='b3',key_dim=1024,value_dim=1024,query_dim=1024).cuda().eval()
        report=[]
        with torch.inference_mode():
            for row in rows[:8]:
                loaded=load(row,a);values,source_valid,index,logits,valid,target_valid=loaded
                online=model(source_values=values[None],source_valid=source_valid[None],semantic_indices=index[None],
                    matching_logits=logits[None],candidate_valid=valid[None],target_valid=target_valid[None])
                with np.load(contract.parent/(row['sample_id']+'.npz')) as z:
                    if str(z['contract_sha256'])!=digest:raise ValueError('reference audit contract differs')
                    actual=torch.from_numpy(z['evidence_soft'].copy()).cuda()
                    error=float((actual-online.evidence[0]).abs().max())
                    if error>2e-6 or not np.array_equal(z['cached_valid'],online.query_valid[0].cpu().numpy()):
                        raise ValueError('reference preaggregation changes frozen R9 evidence')
                    report.append({'sample_id':row['sample_id'],'soft_max_abs_error':error})
        (a.root/'reference-cache-audit.json').write_text(json.dumps({'status':'PASS','samples':report,'contract_sha256':digest,
            'soft_dtype':'float32','source_teacher_used':False},indent=2)+'\n')


if __name__=='__main__':main()
