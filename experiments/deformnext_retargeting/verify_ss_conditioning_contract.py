"""Post-training SS cache/AMP/CFG checks bound to actual saved weights."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .prepare_ss_seed_confirmation import read, write_new


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def tensor_fingerprints(tree):
    if torch.is_tensor(tree):
        raw=tree.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return {'shape':list(tree.shape),'dtype':str(tree.dtype),'sha256':hashlib.sha256(raw).hexdigest()}
    if isinstance(tree,dict): return {k:tensor_fingerprints(v) for k,v in tree.items()}
    if isinstance(tree,(list,tuple)): return [tensor_fingerprints(v) for v in tree]
    if tree is None or isinstance(tree,(str,int,float,bool)): return tree
    raise TypeError('unexpected conditioning-pack value: '+type(tree).__name__)


def online_dataset_args(cached):
    result=copy.deepcopy(cached);result['match_root']=None
    # Online R9 C1 already loads Omega keys through its normal schema. The
    # task flag explicitly requires a cache and therefore must not be reused.
    result.pop('task_matching_features',None)
    return result


def verify_saved_conditioner(initial,saved,trainable,task):
    frozen=[k for k in initial if k not in trainable]
    if not frozen and not task:raise ValueError('static matcher has no frozen initializer state')
    if any(not torch.equal(initial[k],saved[k]) for k in frozen):
        raise ValueError('saved frozen state differs from the selected initializer')
    changed=[k for k in initial if k.startswith('q2c.') and k in trainable and not torch.equal(initial[k],saved[k])]
    if any(k.startswith('q2c.') for k in trainable) and not changed:raise ValueError('task matcher weights did not change from initialization')
    attention_changed=[k for k in initial if k.startswith('task_attention.') and k in trainable and not torch.equal(initial[k],saved[k])]
    if task and not attention_changed:raise ValueError('task attention weights did not change from initialization')
    return {'frozen_initializer_state_checked_keys':frozen,
        'frozen_initializer_state_exact':True if frozen else None,
        'actual_trainable_matcher_saved_state_changed_keys':changed,
        'actual_task_attention_saved_state_changed_keys':attention_changed}


def check_native_tokens(cached,fresh):
    if cached.shape!=fresh.shape or cached.shape[-1]!=1024 or not torch.isfinite(cached).all() or not torch.isfinite(fresh).all():
        raise ValueError('native DINO token shape/finiteness differs')
    delta=cached.float()-fresh.float();rms=float(fresh.float().square().mean().sqrt())
    relative=float(delta.square().mean().sqrt())/max(rms,1e-8)
    cosine=float(torch.nn.functional.cosine_similarity(cached.float(),fresh.float(),dim=-1).mean())
    if relative>.005 or cosine<.9999:
        raise ValueError(f'cached native DINO differs from fresh FP32 RGB encoding:relative_RMS={relative},cosine={cosine}')
    return {'relative_RMS':relative,'mean_token_cosine':cosine,'max_abs_error':float(delta.abs().max()),
        'relative_RMS_limit':.005,'mean_cosine_minimum':.9999,'scope':'Fresh FP32 DINO from actual four RGB tensors versus FP16-stored native tokens;not bit-exact BF16/FP32 encoder equality.'}


def compare_trace_tensors(left, right, *, name):
    """Compare scores with identical invalid masks; report finite errors only."""
    if left.shape != right.shape:
        raise ValueError('trace shape differs: '+name)
    if torch.isnan(left).any() or torch.isnan(right).any():
        raise ValueError('NaN trace tensor: '+name)
    if name == 'matching_logits':
        if torch.isposinf(left).any() or torch.isposinf(right).any():
            raise ValueError('positive infinity is not an invalid matching logit')
        masked = torch.isneginf(left)
        if not torch.equal(masked, torch.isneginf(right)):
            raise ValueError('matching-logit negative-infinity masks differ')
    else:
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError('nonfinite trace tensor: '+name)
        masked = torch.zeros_like(left, dtype=torch.bool)
    finite = ~masked
    if not finite.any():
        raise ValueError('no finite trace values: '+name)
    # Keep the original equality assertion, including the masked infinities.
    torch.testing.assert_close(left, right, rtol=0, atol=1e-5)
    error = float((left.float()[finite]-right.float()[finite]).abs().max())
    if not np.isfinite(error):
        raise ValueError('nonfinite trace error statistic: '+name)
    return {'max_abs_error': error, 'finite_values': int(finite.sum()),
            'masked_negative_infinity_values': int(masked.sum())}


def attention_weight_statistics(weights, query_valid):
    """Describe the actual categorical trace, excluding inactive query rows."""
    if weights is None:
        return None  # Hard Top1 does not return a categorical-weight trace.
    if weights.ndim!=3 or query_valid.shape!=weights.shape[:2] or query_valid.dtype!=torch.bool:
        raise ValueError('attention weight/query mask shape differs')
    p=weights.detach().float()
    if not torch.isfinite(p).all() or (p<0).any():raise ValueError('nonfinite/negative attention weight')
    def describe(rows):
        count=len(rows)
        if not count:return {'active_queries':0,'mean_max_weight':None,'mean_entropy_nats':None,'mean_effective_candidates':None,'max_row_sum_error':None}
        total=rows.sum(-1,keepdim=True);error=float((total-1).abs().max())
        if error>.005:raise ValueError('active attention weights are not normalized')
        rows=rows/total
        entropy=-(rows*rows.clamp_min(1e-30).log()).sum(-1)
        return {'active_queries':count,'mean_max_weight':float(rows.amax(-1).mean()),
            'mean_entropy_nats':float(entropy.mean()),'mean_effective_candidates':float(entropy.exp().mean()),
            'max_row_sum_error':error}
    return {'candidate_slots':p.shape[-1],**describe(p[query_valid]),
        'per_case':[describe(case[valid]) for case,valid in zip(p,query_valid)],
        'scope':'Actual two-case saved-model inference trace;FP32 row-normalized statistics over active queries only,not dataset coverage or quality evidence.'}


def check_conditions(cached, online, train0, train1, mixed, native, mask):
    keys = {'global', 'proj'}
    if set(cached) != {'cond','neg_cond'} or set(online) != set(cached):
        raise ValueError('expected native SS conditional/unconditional branches')
    values = [cached['cond'], cached['neg_cond'], online['cond'], online['neg_cond'],
              train0, train1, mixed, native['cond']]
    if any(set(value) != keys for value in values):
        raise ValueError('SS conditioning keys changed')
    if any(not torch.isfinite(tensor).all() for value in values for tensor in value.values()):
        raise ValueError('nonfinite conditioning tensor')
    for key in keys:
        torch.testing.assert_close(cached['cond'][key], online['cond'][key], rtol=0, atol=1e-5)
        torch.testing.assert_close(cached['neg_cond'][key], online['neg_cond'][key], rtol=0, atol=0)
        torch.testing.assert_close(train0[key], cached['cond'][key], rtol=0, atol=1e-5)
        if torch.count_nonzero(cached['neg_cond'][key]) or torch.count_nonzero(train1[key]):
            raise ValueError('CFG all-drop leaked active global/transport conditioning')
        shape = [len(mask)] + [1] * (train0[key].ndim-1)
        expected = torch.where(torch.tensor(mask, device=train0[key].device).reshape(shape),
                               cached['neg_cond'][key], train0[key])
        torch.testing.assert_close(mixed[key], expected, rtol=0, atol=1e-5)
    torch.testing.assert_close(cached['cond']['global'], native['cond']['global'], rtol=0, atol=0)
    delta = cached['cond']['proj'].float()-native['cond']['proj'].float()
    if not torch.count_nonzero(delta):
        raise ValueError('inactive path cannot prove trained-module cache/CFG parity')
    return {'active_projection_update_rms': float(delta.square().mean().sqrt()),
        'cached_online_condition_max_abs_errors': {k:float((cached['cond'][k]-online['cond'][k]).abs().max()) for k in keys},
        'p0_train_inference_max_abs_errors': {k:float((train0[k]-cached['cond'][k]).abs().max()) for k in keys},
        'p1_all_branches_exact_zero': True, 'mixed_sample_mask': list(map(bool, mask)),
        'mixed_global_and_projection_mask_verified': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--conditioner-checkpoint', type=Path, required=True)
    p.add_argument('--panel-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if (args.output/'summary.json').exists(): raise FileExistsError('preserve the previous contract proof')
    from pixal3d import datasets, models, trainers
    from pixal3d.utils.data_utils import recursive_to_device
    torch.set_num_threads(2);torch.cuda.set_device(0)
    cfg = read(args.config);spec = cfg['trainer']['args']['deform_conditioner']
    task=spec.get('task_attention')
    if (spec['implementation']!='r9-native-v5' or spec['arm'] not in {'b2','b3','c1'} or spec['value_dim']!=1024
        or (not task and not spec['freeze_q2c'])):
        raise ValueError('this proof requires native R9 SS Top1/soft/C1 with frozen or task-trained matching')
    head = Path(spec['q2c_checkpoint']);head_ref = fingerprint(head)
    if head_ref['sha256'] != cfg['experiment']['head_sha256']:
        raise ValueError('selected head fingerprint differs')
    ds_args = copy.deepcopy(cfg['validation']['dataset_args'])
    if ds_args['expected_q2c_sha256'] != head_ref['sha256'] or not ds_args['match_root']:
        raise ValueError('dataset cache/head contract differs')
    manifest = Path(cfg['validation']['roots_file'])
    cached_ds = getattr(datasets,cfg['dataset']['name'])(str(manifest), **ds_args)
    online_args = online_dataset_args(ds_args)
    online_ds = getattr(datasets,cfg['dataset']['name'])(str(manifest), **online_args)
    indices = {row['sample_id']:i for i,row in enumerate(cached_ds.rows)}
    if [r['sample_id'] for r in online_ds.rows] != [r['sample_id'] for r in cached_ds.rows]:
        raise ValueError('cached/online dataset identity changed')
    selected = []
    for sid in read(args.panel_file)['sample_ids']:
        index = indices[sid];item = cached_ds[index]
        if item['transport_target_valid'].any() and int(item['transport_source_valid'].sum()) >= 64:
            selected.append((sid,index,item))
        if len(selected) == 2: break
    if len(selected) != 2: raise ValueError('two real active cases are required; no fabricated confidence')
    with torch.device('meta'):
        md = {name:getattr(models,s['name'])(**s['args']) for name,s in cfg['models'].items()}
    kw = copy.deepcopy(cfg['trainer']['args'])
    kw.update(continuation_ckpt={'denoiser':str(args.checkpoint)}, batch_size_per_gpu=2,
        batch_split=1, num_workers=0, prefetch_data=False, max_steps=1)
    trainer = getattr(trainers,cfg['trainer']['name'])(md,cached_ds,**kw,
        output_dir=str(args.output/'runtime'),load_dir=None,step=None,wandb_run=None)
    trainable={k for k,v in trainer.deform_conditioner.named_parameters() if v.requires_grad}
    initialized={k:v.detach().cpu().clone() for k,v in trainer.deform_conditioner.state_dict().items()}
    saved_state=torch.load(args.conditioner_checkpoint,map_location='cpu',weights_only=True)
    trainer.deform_conditioner.load_state_dict(saved_state,strict=True)
    state_checks=verify_saved_conditioner(initialized,saved_state,trainable,task)
    for model in trainer.models.values(): model.eval().requires_grad_(False)
    if len(trainer.optimizer.state): raise ValueError('proof must not load training optimizer state')

    def pack(ds,items):
        value=recursive_to_device(ds.collate_fn(items),'cuda')
        for key in ['x_0','_dataset_name','_sha256']: value.pop(key,None)
        return value
    cached_pack=pack(cached_ds,[item for _,_,item in selected])
    online_pack=pack(online_ds,[online_ds[index] for _,index,_ in selected])
    if tuple(cached_pack['cond'].shape[:3])!=(2,4,3):
        raise ValueError('two actual four-view RGB inputs are required')
    if any(cached_ds.rows[index]['conditioning_camera_ids']!=[8,9,10,11] for _,index,_ in selected):
        raise ValueError('actual camera order changed')
    cached_inputs,online_inputs=map(tensor_fingerprints,[cached_pack,online_pack])
    native_inputs=lambda value:{k:v for k,v in value.items() if not k.startswith('transport_')}
    if native_inputs(cached_inputs)!=native_inputs(online_inputs):
        raise ValueError('cached/online real RGB or camera input differs')
    trace_errors={};trace_comparisons={};trainer.p_uncond=0
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        traces=[];conditions=[]
        for one in [cached_pack,online_pack]:
            # Use the public camera/ProjGrid route also for native-spatial;
            # calling the conditioner directly would skip its spatial inputs.
            conditions.append(trainer.get_inference_cond(**one))
            traces.append(trainer._last_deform_condition)
        torch.testing.assert_close(traces[0].candidate_indices,traces[1].candidate_indices,rtol=0,atol=0)
        for key in ['evidence','matching_logits','candidate_weights','view_weights','retarget_residual']:
            a,b=getattr(traces[0],key),getattr(traces[1],key)
            if a is None or b is None:
                if a is not None or b is not None:raise ValueError('optional trace differs: '+key)
                continue
            trace_comparisons[key]=compare_trace_tensors(a,b,name=key)
            trace_errors[key]=trace_comparisons[key]['max_abs_error']
        attention_stats=attention_weight_statistics(traces[0].candidate_weights,traces[0].query_valid)
        if spec['arm']=='c1' and not traces[0].retarget_residual.abs().sum()>0:
            raise ValueError('actual trained retargeter is inactive on selected cases')
        cached,online=conditions
        task_cache_checks=None
        if task:
            altered=dict(cached_pack)
            altered['transport_matching_logits']=torch.zeros_like(altered['transport_matching_logits'])+123.
            if spec.get('correspondence_mode','semantic')=='native-spatial':
                altered['transport_semantic_indices']=(altered['transport_semantic_indices']+17)%3072
                altered['transport_candidate_valid']=torch.zeros_like(altered['transport_candidate_valid'])
            ignored=trainer.get_inference_cond(**altered)
            for branch in ['cond','neg_cond']:
                for key in cached[branch]:torch.testing.assert_close(cached[branch][key],ignored[branch][key],rtol=0,atol=0)
            task_cache_checks={'stale_cached_matching_logits_ignored':True,
                'native_spatial_ignores_semantic_cached_candidates':spec.get('correspondence_mode','semantic')=='native-spatial'}
        trainer.deform_conditioner.train()
        train0=trainer.get_cond(**cached_pack)
        trainer.p_uncond=1.;train1=trainer.get_cond(**cached_pack)
        trainer.p_uncond=.5;np_state=np.random.get_state()
        try:
            np.random.seed(1);mixed=trainer.get_cond(**cached_pack)
        finally: np.random.set_state(np_state)
        trainer.deform_conditioner.eval()
        expected_mask=(np.random.RandomState(1).rand(2)<.5).tolist()
        saved=trainer.deform_conditioner_config
        try:
            trainer.deform_conditioner_config=None
            native=trainer.get_inference_cond(**{k:v for k,v in cached_pack.items() if not k.startswith('transport_')})
        finally: trainer.deform_conditioner_config=saved
        checks=check_conditions(cached,online,train0,train1,mixed,native,expected_mask)
    # Validate the native value feature contract independently of matching
    # cache/online parity. Precomputation uses FP32 DINO then FP16 storage.
    extractor=trainer._image_cond_module
    if any(p.dtype!=torch.float32 for p in extractor.model.parameters()):
        raise ValueError('fresh native-token reference requires original FP32 DINO parameters')
    token_cases=[]
    for _,index,_ in selected:
        with np.load(cached_ds._resolve(cached_ds.rows[index]['transport_cache'])) as z:
            token_cases.append(torch.cat([torch.from_numpy(z['target_query'].copy())[None],torch.from_numpy(z['source_values'].copy())],0))
    token_cache=torch.stack(token_cases).to('cuda').float()
    with torch.no_grad(),torch.autocast('cuda',enabled=False):
        images=cached_pack['cond'].flatten(0,1).float()
        hidden=extractor.extract_features(extractor.transform(images))
        registers=getattr(extractor.model.config,'num_register_tokens',4)
        fresh=hidden[:,1+registers:].float().reshape_as(token_cache)
        native_token_check=check_native_tokens(token_cache,fresh)
    if len(trainer.optimizer.state): raise ValueError('conditioning check applied an optimizer update')
    result={'status':'PASS_CURRENT_C1_CACHE_AMP_CFG_CONTRACT' if not task and spec['arm']=='c1' else 'PASS_TRAINED_SS_CACHE_AMP_CFG_CONTRACT',
        'amp':'bf16','sample_ids':[sid for sid,_,_ in selected],'selection':'first2 fixed-panel cases with actual target/source validity',
        'trained_denoiser':fingerprint(args.checkpoint),'trained_conditioner':fingerprint(args.conditioner_checkpoint),
        'selected_head':head_ref,'source_recipe':fingerprint(args.config),'manifest':fingerprint(manifest),
        'head_reference_role':'Initializer checkpoint;task-trained matching state is taken from the actual saved conditioner.',
        'saved_frozen_head_exact_to_selected_checkpoint':bool(state_checks['frozen_initializer_state_checked_keys']),
        'frozen_head_state_keys':sorted(state_checks['frozen_initializer_state_checked_keys']),
        **state_checks,'task_cache_checks':task_cache_checks,'task_attention_configuration':task,
        'actual_attention_weight_statistics':attention_stats,
        'actual_content_attention_scale':float(trainer.deform_conditioner.task_attention.log_content_scale.exp().clamp(max=20.)) if task else None,
        'fresh_native_DINO_token_check':native_token_check,
        'actual_float32_matmul_precision':torch.get_float32_matmul_precision(),
        'actual_allow_TF32':torch.backends.cuda.matmul.allow_tf32,
        'actual_constructor_trainable_conditioner_keys':sorted(trainable),
        'panel':fingerprint(args.panel_file),'candidate_indices_exact':True,'active_trace_max_abs_errors':trace_errors,
        'active_trace_comparisons':trace_comparisons,'trace_nonfinite_masks_verified':True,
        'actual_cached_input_tensor_fingerprints':cached_inputs,
        'actual_online_input_tensor_fingerprints':online_inputs,'actual_four_RGB_cameras_exact':True,
        **checks,'optimizer_updates':0,'scope':'Two actual active SS cases and actual saved weights;cached versus online proposals/current head,BF16,train/inference/CFG contract. Not quality evaluation or all-asset parity.'}
    json.dumps(result,allow_nan=False)
    write_new(args.output/'summary.json',result)


if __name__ == '__main__': main()
