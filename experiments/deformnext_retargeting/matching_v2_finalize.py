"""Calibration-only selection and inference-only feature cache construction."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .matching_v2_head import MatchingV2Head, SCHEMA, masked_probabilities


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def calibrate(logits, labels):
    from scipy.optimize import minimize
    x,y=np.asarray(logits,dtype=np.float64),np.asarray(labels,dtype=np.float64)
    def loss(ab):
        z=ab[0]*x+ab[1]
        value=np.logaddexp(0,z)-y*z
        probability=1/(1+np.exp(-np.clip(z,-50,50)))
        delta=probability-y
        return value.mean()+1e-6*ab[0]**2,np.array([(delta*x).mean()+2e-6*ab[0],delta.mean()])
    fit=minimize(loss,np.array([1.,0.]),jac=True,method='L-BFGS-B',bounds=[(0.,100.),(-100.,100.)])
    if not np.isfinite(fit.fun):raise ValueError('confidence calibration failed')
    slope,bias=fit.x.tolist();probability=1/(1+np.exp(-np.clip(slope*x+bias,-50,50)))
    order=np.argsort(-probability);tp=np.cumsum(y[order]);n=np.arange(1,len(y)+1)
    precision=tp/n;recall=tp/max(1,y.sum());fscore=1.25*precision*recall/np.maximum(.25*precision+recall,1e-12)
    fscore[n<max(32,int(.05*len(y)))]=-1
    selected=int(fscore.argmax());threshold=float(probability[order[selected]])
    accepted=probability>=threshold
    return {'slope':slope,'bias':bias,'threshold':threshold,'selection':'maximum calibration F0.5, at least5% coverage and32 queries',
            'calibration_precision':float(y[accepted].mean()),'calibration_coverage':float(accepted.mean()),
            'calibration_brier':float(np.mean((probability-y)**2)),'optimizer_success':bool(fit.success)}


def select(root):
    candidates=[]
    for arm in ['h1','h2','h3']:
        directory=root/'heads'/arm
        history=json.loads((directory/'history.json').read_text())
        if history[-1]['epoch']!=7:raise ValueError('head selection requires fixed-final7 epochs')
        cal=json.loads((directory/'calibration.json').read_text())
        temperature=min(cal['feature_error'],key=lambda t:cal['feature_error'][t])
        record={'arm':arm,'head':str(directory/'final.pt'),'head_sha256':sha(directory/'final.pt'),
                'temperature':float(temperature),'calibration_feature_error':cal['feature_error'][temperature],
                'calibration_top1_accuracy':cal['top1_accuracy']}
        if arm!='h1':record['reliability']=calibrate(cal['confidence_logits'],cal['usable'])
        candidates.append(record)
    # The full SS ladder needs a supervised reliability output. H1 is retained
    # as a head ablation, never supplied with an untrained confidence fallback.
    chosen=min([x for x in candidates if x['arm']!='h1'],key=lambda x:x['calibration_feature_error'])
    record={'schema':'deformnext-matching-v2-cache-v1',**chosen,
            'topk':json.loads((root/'proposal-audit.json').read_text())['selected_topk'],
            'selection_uses':'train-internal calibration only; H2/H3 minimum transported feature cosine error',
            'candidate_results':candidates,'descriptor':'OMEGA epoch7 FRAME1024; C2 context FRAME+GLOBAL2048',
            'input_cameras':[8,9,10,11],'target_modality':'rgb'}
    cache=root/'matches';cache.mkdir(exist_ok=True)
    (cache/'contract.json').write_text(json.dumps(record,indent=2)+'\n')
    (root/'selected-head.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


@torch.inference_mode()
def cache_row(row, model, contract, root, mirror):
    path=mirror/row['transport_cache'].removeprefix('/file4/')
    device=lambda array:torch.from_numpy(np.ascontiguousarray(array)).pin_memory().cuda(non_blocking=True).float()
    with np.load(path) as z:
        target_context=device(z['target_keys']);source_context=device(z['source_keys']).flatten(0,1)
        values=device(z['source_values']).flatten(0,1)
        if str(z['conditioning_input_sha256'])!=row['conditioning_input_sha256']:raise ValueError('feature input mismatch')
    with np.load(root/'proposals'/(row['sample_id']+'.npz')) as z:
        if str(z['input_sha256'])!=row['conditioning_input_sha256']:raise ValueError('proposal input mismatch')
        k=contract['topk'];indices=torch.from_numpy(z['indices'][:,:k].astype(np.int64)).cuda()
        raw=device(z['cosine'][:,:k]);valid=torch.from_numpy(z['valid'][:,:k].copy()).cuda().bool()
    from PIL import Image
    image=mirror/row['target_anchor'].removeprefix('/file4/')
    alpha=np.asarray(Image.open(image).convert('RGBA').resize((512,512),Image.Resampling.NEAREST))[8::16,8::16,3].flatten()>127
    usable=torch.from_numpy(alpha.copy()).cuda()&valid.any(-1)
    top1=values.new_zeros(1024,1024);soft=torch.zeros_like(top1);deformation=values.new_zeros(1024,2048);p_use=values.new_zeros(1024)
    for chunk in torch.where(usable)[0].split(32):
        logits,confidence=model(target_context[chunk,:1024],source_context[indices[chunk],:1024],raw[chunk],valid[chunk])
        weights=masked_probabilities(logits,valid[chunk],contract['temperature'])
        chosen=indices[chunk].gather(1,logits.argmax(-1,keepdim=True))[:,0]
        top1[chunk]=values[chosen]
        soft[chunk]=(values[indices[chunk]]*weights[...,None]).sum(1)
        summary=(source_context[indices[chunk]]*weights[...,None]).sum(1)
        deformation[chunk]=target_context[chunk]-summary
        reliability=contract['reliability']
        p_use[chunk]=torch.sigmoid(reliability['slope']*confidence[:,0]+reliability['bias'])
    return {'evidence_top1':top1.cpu().half().numpy(),'evidence_soft':soft.cpu().half().numpy(),
            'p_use':p_use.cpu().numpy(),'cached_valid':usable.cpu().numpy(),
            'deformation':deformation.cpu().half().numpy()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['select','cache','audit'],required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path);p.add_argument('--file4-mirror',type=Path)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(4)
    if a.mode=='select':select(a.root);return
    torch.cuda.set_device(0)
    contract=json.loads((a.root/'matches/contract.json').read_text());digest=sha(a.root/'matches/contract.json')
    if sha(Path(contract['head']))!=contract['head_sha256']:raise ValueError('selected head changed')
    state=torch.load(contract['head'],map_location='cpu',weights_only=True)
    if state['schema']!=SCHEMA:raise ValueError('wrong head schema')
    model=MatchingV2Head().cuda().eval();model.load_state_dict(state['model'],strict=True)
    rows=[json.loads(l) for l in a.manifest.read_text().splitlines() if l]
    if a.mode=='cache':
        assigned=rows[a.rank::a.world_size]
        for i,row in enumerate(assigned):
            path=a.root/'matches'/(row['sample_id']+'.npz')
            if path.exists():
                with np.load(path) as z:
                    if str(z['contract_sha256'])!=digest or str(z['conditioning_input_sha256'])!=row['conditioning_input_sha256']:raise ValueError('stale matching-v2 cache')
                continue
            arrays=cache_row(row,model,contract,a.root,a.file4_mirror)
            if any(not np.isfinite(arrays[key]).all() for key in ['evidence_top1','evidence_soft','deformation','p_use']):raise ValueError('nonfinite preaggregated evidence')
            tmp=path.with_name('.'+path.name+'.tmp.npz')
            np.savez(tmp,schema='deformnext-matching-v2-cache-v1',contract_sha256=digest,
                conditioning_input_sha256=row['conditioning_input_sha256'],sample_id=row['sample_id'],split=row['split'],**arrays)
            os.replace(tmp,path)
            if i%100==0:print(json.dumps({'rank':a.rank,'completed':i+1,'assigned':len(assigned)}),flush=True)
        (a.root/f'cache-rank{a.rank}.json').write_text(json.dumps({'status':'PASS','assigned':len(assigned),'contract_sha256':digest})+'\n')
    else:
        report=[]
        for row in rows[:8]:
            expected=cache_row(row,model,contract,a.root,a.file4_mirror)
            with np.load(a.root/'matches'/(row['sample_id']+'.npz')) as z:
                if str(z['contract_sha256'])!=digest:raise ValueError('cache parity contract differs')
                differences={key:float(np.max(np.abs(z[key].astype(float)-value.astype(float)))) for key,value in expected.items()}
                if max(differences.values())>1e-5:raise ValueError('cache/online evidence differs')
                report.append({'sample_id':row['sample_id'],'max_errors':differences})
        (a.root/'cache-audit.json').write_text(json.dumps({'status':'PASS','contract_sha256':digest,'samples':report},indent=2)+'\n')


if __name__=='__main__':main()
