"""Read-only probability audit on the explicit auxiliary calibration32 cache.

These32 assets are held out from utility-head fitting but belong to SS TRAIN.
GT sets enter CPU metrics only. This is neither reconstruction validation nor
evidence that a probability intervention improves the generated model.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import time

import numpy as np
import torch

from pixal3d.modules.ray_feature_attention import RayFeatureReadout
from . import ray_set_head_training as cache


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--cold-conditioner',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES','')
    if physical not in {'0','1','2','3'} or physical not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one explicitly assigned GPU0-3 required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.1,0)
    torch.use_deterministic_algorithms(True)
    bound = {str(p.resolve()):sha(p) for p in [Path(__file__),Path(cache.__file__),
        Path(__file__).parents[2]/'pixal3d/modules/ray_feature_attention.py',
        args.config,args.cold_conditioner,args.checkpoint,args.cache_root/'cache-index.jsonl']}
    cfg = json.loads(args.config.read_text())
    if (cfg['trainer']['args']['deform_conditioner']['score_temperature']!=.05
            or not args.checkpoint.name.endswith('step0000192.pt')):
        raise ValueError('this declared audit compares the fixed192 head with native temperature0.05')
    manifest = Path(cfg['data_dir'])
    metadata = {row['sample_id']:row for row in map(json.loads,manifest.read_text().splitlines())}
    bound[str(manifest)] = sha(manifest)
    selected = [r for r in map(json.loads,(args.cache_root/'cache-index.jsonl').read_text().splitlines())
                if r['role']=='calibration']
    if len(selected)!=32 or len({r['source_cluster_id'] for r in selected})!=32:
        raise ValueError('the exact32 auxiliary calibration source clusters required')

    def head(path):
        state = torch.load(path,map_location='cpu',weights_only=True)
        state = {key[len('ray_readout.'):]:value for key,value in state.items() if key.startswith('ray_readout.')}
        if len(state)!=9:
            raise ValueError('exact nine-tensor feature readout checkpoint required')
        model = RayFeatureReadout(1024).cuda()
        model.load_state_dict(state,strict=True)
        model.eval().requires_grad_(False)
        return model

    cold,trained = head(args.cold_conditioner),head(args.checkpoint)
    if torch.count_nonzero(cold.output.weight):
        raise ValueError('untouched zero-output initial feature head required')
    anchored = copy.deepcopy(cold)
    anchored.query.weight.zero_()
    trained_anchored = copy.deepcopy(trained)
    trained_anchored.query.weight.zero_()
    args.output.mkdir(parents=True,exist_ok=False)
    rows = []
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for row in selected:
        original = metadata[row['sample_id']]
        if original['split']!='train' or row['query_unit']!='edited-target-patch-ray':
            raise ValueError('explicit SS-TRAIN auxiliary calibration domain required')
        for field in ['inputs','labels']:
            path = args.cache_root/row[field+'_relative_path']
            if sha(path)!=row[field+'_SHA256']:
                raise ValueError('original closed input/loss-label bytes changed')
        x,y = cache.load_asset(args.cache_root,row)
        with np.load(args.cache_root/row['inputs_relative_path'],allow_pickle=False) as archive:
            patches = archive['query_patch_indices'].astype(np.int64)
        raw_path = original['transport_cache']
        pathmap = cfg['dataset']['args']['path_map']
        prefix = '/file4'
        if not raw_path.startswith(prefix+'/') or prefix not in pathmap:
            raise ValueError('explicit file4 transport mapping required')
        transport = Path(pathmap[prefix]+raw_path[len(prefix):])
        match = Path(cfg['dataset']['args']['match_root'])/(row['sample_id']+'.npz')
        bound[str(transport)] = sha(transport)
        bound[str(match)] = sha(match)
        with np.load(transport,allow_pickle=False) as archive:
            if str(archive['conditioning_input_sha256'])!=original['conditioning_input_sha256']:
                raise ValueError('current RGB fingerprint differs')
            source_keys = archive['source_keys'][...,:1024].reshape(3072,1024).copy()
            target_keys = archive['target_keys'][patches,:1024].copy()
            if not np.array_equal(archive['source_values'].reshape(3072,1024),x['source_value_bank']):
                raise ValueError('cached source values differ from current transport')
        with np.load(match,allow_pickle=False) as archive:
            if (not np.array_equal(archive['semantic_indices'][patches],x['candidate_indices'])
                    or not np.array_equal(archive['candidate_valid'][patches],x['candidate_valid'])
                    # Native CUDA divides by0.05 using reciprocal20. NumPy
                    # divide can differ by one ULP; multiplication reproduces
                    # the actual cached/native FP32 operation exactly.
                    or not np.array_equal(archive['matching_logits'][patches]*20.,x['scalar_prior_logits'])):
                raise ValueError('cached proposals or prior differ from current native bank')
        parts = {name:[] for name in ['prior','utility7','cold_random_QK','cold_Qzero',
            'trained192','trained192_Qzero','trained192_score_prior_off','trained192_decoder_context_raw']}
        with torch.no_grad():
            for begin in range(0,len(patches),64):
                take = slice(begin,begin+64)
                native = torch.from_numpy(x['native_query'][take]).cuda().float()
                context = torch.from_numpy(x['context_query'][take]).cuda().float()
                indices = x['candidate_indices'][take].astype(np.int64)
                values = torch.from_numpy(x['source_value_bank'][indices]).cuda().float()
                keys = torch.from_numpy(source_keys[indices]).cuda().float()
                valid = torch.from_numpy(x['candidate_valid'][take]).cuda()
                prior = torch.from_numpy(x['scalar_prior_logits'][take]).cuda().float()
                raw = torch.from_numpy(target_keys[take]).cuda().float()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    parts['prior'].append(torch.softmax(prior,dim=-1).cpu())
                    utility = cold.score_selector(native,context,values,valid,prior)[1]
                    parts['utility7'].append(utility.cpu())
                    for name,model,ctx,score in [('cold_random_QK',cold,context,prior),
                            ('cold_Qzero',anchored,context,prior),('trained192',trained,context,prior),
                            ('trained192_Qzero',trained_anchored,context,prior),
                            ('trained192_score_prior_off',trained,context,torch.zeros_like(prior)),
                            ('trained192_decoder_context_raw',trained,raw,prior)]:
                        probability = model(native,ctx,keys,values,valid,score)[1].mean(1)
                        parts[name].append(probability.cpu())
        probability = {name:torch.cat(part).float() if part else torch.empty(0,64) for name,part in parts.items()}
        if not torch.allclose(probability['cold_Qzero'],probability['utility7'],atol=1e-7,rtol=0):
            raise ValueError('zeroQ must reproduce the pretrained utility probabilities')
        observed = torch.from_numpy(y['relation'])
        metrics = {}
        for name,p in probability.items():
            if not torch.isfinite(p).all() or not torch.allclose(p.sum(-1),torch.ones(len(p)),atol=1e-5,rtol=0):
                raise ValueError('finite normalized actual head probabilities required')
            metrics[name] = dict(observable_relations=len(observed),query_count=len(p),
                effective_K=float((-(p*p.clamp_min(1e-30).log()).sum(-1)).exp().mean()) if len(p) else 0.,
                **{kind:dict(mass_sum=float((p[observed]*torch.from_numpy(y[kind])).sum()),
                    hit1=int(torch.from_numpy(y[kind]).gather(1,p[observed].argmax(-1)[:,None]).sum()),
                    proposal64_hits=int(y[kind].any(-1).sum())) for kind in ['exact','near1']})
        rows.append(dict(sample_id=row['sample_id'],source_cluster_id=row['source_cluster_id'],
            split='train',role='auxiliary-calibration',metrics=metrics))
    summary = {}
    for name in rows[0]['metrics']:
        items = [r['metrics'][name] for r in rows]
        count = sum(r['observable_relations'] for r in items)
        summary[name] = dict(observable_relations=count,**{kind:dict(
            mean_probability_mass=sum(r[kind]['mass_sum'] for r in items)/count,
            top1_coverage=sum(r[kind]['hit1'] for r in items)/count,
            proposal64_coverage=sum(r[kind]['proposal64_hits'] for r in items)/count)
            for kind in ['exact','near1']})
    if any(sha(path)!=digest for path,digest in bound.items()):
        raise ValueError('read-only audit source/checkpoint changed')
    result = dict(status='PASS_ACTUAL_FEATURE_HEAD_PROBABILITY_AUDIT_AUXILIARY_CALIBRATION32',
        assets=32,source_clusters=32,split='SS-TRAIN;heldoutonlyfromauxiliaryutilityfit',summary=summary,
        per_asset=rows,source_SHA256=bound,physical_GPU=physical,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        seconds=time.monotonic()-began,optimizer_updates=0,GT_in_model_forward=False,
        Qzero_reproduces_trained_utility7=True,reconstruction_or_validation_gain_claimed=False,
        score_off_not_user_Top1_NoScore=True,decoder_bypass_retains_D1_candidate_bank=True,
        model_adopted=False,full_goal_complete=False)
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],summary=summary,peak_GiB=result['peak_allocated_GiB'])),flush=True)


if __name__=='__main__':
    main()
