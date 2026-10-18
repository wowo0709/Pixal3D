"""Re-encode actual RGB with the selected frozen Omega7 aggregator.

This complements saved-head cache/CFG checks. It does not run the camera/depth
heads, train a model, or measure SS quality. Fresh proposal equality is reported
separately from bounded feature-storage/encoder numerical differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F

from .omega_loader import load_epoch7, layer12_patch_tokens
from .prepare_rgb_manifest import image_fingerprint
from .semantic_topk_search import write_new


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def compare_tokens(cached,fresh):
    if cached.shape!=(4,1024,2048) or fresh.shape!=cached.shape:
        raise ValueError('actual four-view32x32 FRAME+GLOBAL Omega L12 layout required')
    if not torch.isfinite(cached).all() or not torch.isfinite(fresh).all():
        raise ValueError('nonfinite Omega features')
    comparisons=[]
    for view in range(4):
        for name,start in [('FRAME',0),('GLOBAL',1024)]:
            old=cached[view,:,start:start+1024].float();new=fresh[view,:,start:start+1024].float()
            rms=float(new.square().mean().sqrt())
            if rms==0:raise ValueError('zero fresh Omega feature magnitude')
            delta=old-new
            relative=float(delta.square().mean().sqrt())/rms
            cosine=float(F.cosine_similarity(old,new,dim=-1).mean().clamp(-1,1))
            comparisons.append({'view':view,'component':name,'relative_RMS':relative,
                'mean_cosine':cosine,'max_abs_error':float(delta.abs().max()),
                'pass':relative<=.005 and cosine>=.9999})
    return {'comparisons':comparisons,'pass':all(r['pass'] for r in comparisons),
        'fresh_FP16_exact_to_cache':torch.equal(fresh.half(),cached.half()),
        'relative_RMS_limit':.005,'mean_cosine_minimum':.9999}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','weights-reference','vggt-root','sample-ids-file','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    allowed=os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(',')
    if len(visible)!=1 or not visible[0] or visible[0] not in allowed:
        raise ValueError('one explicit caller-authorized free GPU required')
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[-2:])
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.set_float32_matmul_precision('highest')
    cfg=json.loads(a.config.read_text());ref=json.loads(a.weights_reference.read_text())
    if ref['status']!='PASS_ACTUAL_BASE_AND_EPOCH7_WEIGHT_REFERENCES' or ref['completed_updates']!=23597:
        raise ValueError('selected actual Omega7 weight reference required')
    base=Path(ref['local_base_checkpoint']);epoch=Path(ref['local_epoch7_checkpoint'])
    if sha(base)!=ref['base_checkpoint_SHA256'] or sha(epoch)!=ref['epoch7_checkpoint_SHA256']:
        raise ValueError('actual selected weights changed')
    selected=json.loads(a.sample_ids_file.read_text())
    if selected['status']!='PASS_TRAINED_SS_CACHE_AMP_CFG_CONTRACT':
        raise ValueError('reuse the original completed first-two-active case selection')
    ids=selected['sample_ids']
    if len(ids)!=2 or len(set(ids))!=2:raise ValueError('exactly two fixed distinct active cases required')
    manifest=Path(cfg['validation']['roots_file'])
    rows={row['sample_id']:row for row in [json.loads(s) for s in manifest.read_text().splitlines()]}
    mappings=cfg['dataset']['args']['path_map']
    def resolve(value):
        for prefix,target in mappings.items():
            if value==prefix or value.startswith(prefix+'/'):return Path(target+value[len(prefix):])
        return Path(value)
    if (a.output/'summary.json').exists():raise FileExistsError('preserve previous successful backbone proof')
    a.output.mkdir(parents=True,exist_ok=True)
    # Reuse the exact actual-cache image preprocessing, without running its GT
    # route builder or its native DINO constructor.
    from .precompute_transport import load_rgba,patch_visibility
    from pixal3d.modules.r9_matching import frame_topk
    torch.cuda.reset_peak_memory_stats()
    model,checkpoint=load_epoch7(a.vggt_root,base,epoch)
    if (checkpoint['completed_updates']!=23597 or checkpoint['base_checkpoint_sha256']!=ref['base_checkpoint_SHA256']
        or len(checkpoint['lora'])!=192 or len(checkpoint['frame_ffn_lora'])!=96
        or any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules())):
        raise ValueError('actual frozen Omega7 aggregator construction differs')
    if not all(torch.isfinite(t).all() for family in ['lora','frame_ffn_lora'] for t in checkpoint[family].values()):
        raise ValueError('nonfinite selected Omega7 adapters')
    measurements=[]
    with torch.inference_mode():
        for ix,sid in enumerate(ids):
            row=rows[sid]
            if (row.get('target_anchor_kind')!='rgb' or row['conditioning_camera_ids']!=[8,9,10,11]
                or image_fingerprint(row,resolve)!=row['conditioning_input_sha256']):
                raise ValueError('actual four RGB bytes/cameras differ from cache input')
            paths=[resolve(path) for path in [row['target_anchor'],*row['source_images']]]
            loaded=[load_rgba(path,512) for path in paths]
            images=torch.stack([item[0] for item in loaded]).cuda()
            alpha=torch.stack([item[1] for item in loaded])
            fresh=layer12_patch_tokens(model,images[None])[0].cpu()
            cache=resolve(row['transport_cache'])
            with np.load(cache) as z:
                if int(z['omega_completed_updates'])!=23597 or str(z['conditioning_input_sha256'])!=row['conditioning_input_sha256']:
                    raise ValueError('cache checkpoint/input metadata differs')
                cached=torch.from_numpy(np.concatenate([z['target_keys'][None],z['source_keys']],0))
                valid=torch.from_numpy(z['source_valid'].copy())
            if not torch.equal(valid,patch_visibility(alpha[1:].cuda()).cpu()):raise ValueError('observed source RGB alpha differs')
            feature=compare_tokens(cached,fresh)
            active=(alpha[0,8::16,8::16].flatten()>127/255.) & (valid.sum()>=64)
            if not active.any():raise ValueError('selected case has no active source-supported query')
            def proposals(tokens):
                keys=tokens[...,:1024].half().float().cuda()
                return frame_topk(keys[0:1],keys[None,1:],valid[None].cuda(),64)[1][0].cpu()
            old_index=proposals(cached);new_index=proposals(fresh)
            match=Path(cfg['dataset']['args']['match_root'])/(sid+'.npz')
            with np.load(match) as z:reference=torch.from_numpy(z['semantic_indices'].astype(np.int64))
            cached_exact=torch.equal(old_index[active],reference[active])
            left,right=old_index[active],new_index[active]
            overlap=(left[:,:,None]==right[:,None,:]).any(-1).float().mean()
            result={'sample_id':sid,'source_cluster_id':row['source_cluster_id'],'conditioning_input_sha256':row['conditioning_input_sha256'],
                'actual_four_RGB_SHA256':[sha(path) for path in paths],'cache_SHA256':sha(cache),
                'active_queries':int(active.sum()),'feature_parity':feature,'cached_proposals_exact_to_match_cache':cached_exact,
                'fresh_FP16_boundary_proposals_ordered_exact':torch.equal(left,right),
                'fresh_proposal_ordered_slot_agreement':float((left==right).float().mean()),
                'fresh_proposal_set_overlap':float(overlap)}
            write_new(a.output/f'case{ix}.json',result)
            if not cached_exact:raise ValueError('cached RAW FRAME proposals do not replay the actual matching cache')
            if not feature['pass']:raise ValueError('fresh Omega7 feature difference exceeds declared per-view/component limits')
            measurements.append(result);del images,fresh,cached
    imported={str(Path(module.__file__).resolve()):sha(Path(module.__file__))
        for module in sys.modules.copy().values() if getattr(module,'__file__',None)
        and str(module.__file__).endswith('.py') and Path(module.__file__).resolve().is_relative_to(a.vggt_root.resolve())}
    write_new(a.output/'summary.json',{'status':'PASS_FRESH_OMEGA7_FEATURE_CACHE_PARITY','completed_updates':23597,
        'base_checkpoint_SHA256':ref['base_checkpoint_SHA256'],'epoch7_checkpoint_SHA256':ref['epoch7_checkpoint_SHA256'],
        'strict_base_and_adapter_key_load':True,'all_parameters_frozen_eval':True,
        'cases':measurements,'Omega_forward_passes':2,'SS_generations':0,'optimizer_updates':0,
        'feature_layout':'Layer12[FRAME1024,GLOBAL1024],four RGB512 views,targetM1+source3M0',
        'input_camera_ids':[8,9,10,11],'input_states':['M1','M0','M0','M0'],
        'matmul_precision':torch.get_float32_matmul_precision(),'allow_TF32':torch.backends.cuda.matmul.allow_tf32,
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'fresh_proposals_exact_all_cases':all(x['fresh_FP16_boundary_proposals_ordered_exact'] for x in measurements),
        'imported_Omega_source_SHA256':imported,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'weights_reference_SHA256':sha(a.weights_reference),'sample_selection_SHA256':sha(a.sample_ids_file),
        'manifest_SHA256':sha(manifest),'config_SHA256':sha(a.config),
        'scope':'Two actual fixed active cases. Fresh selected frozen Omega7 aggregator versus existing FP16 keys;per-view FRAME/GLOBAL numerical parity. Proposal equality is separately reported,not exact-Top1 correctness. Camera/depth heads are not executed. Not all-asset or SS-quality evidence.',
        'full_goal_complete':False})
    print('PASS_FRESH_OMEGA7_FEATURE_CACHE_PARITY',flush=True)


if __name__=='__main__':main()
