"""Post-training SS C1 cache/AMP/CFG checks bound to actual saved weights."""
from __future__ import annotations

import argparse
import copy
import hashlib
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
    if spec['implementation'] != 'r9-native-v5' or spec['arm'] != 'c1' or spec['value_dim'] != 1024 or not spec['freeze_q2c']:
        raise ValueError('this proof targets the actual original R9 SS C1 candidate')
    head = Path(spec['q2c_checkpoint']);head_ref = fingerprint(head)
    if head_ref['sha256'] != cfg['experiment']['head_sha256']:
        raise ValueError('selected head fingerprint differs')
    ds_args = copy.deepcopy(cfg['validation']['dataset_args'])
    if ds_args['expected_q2c_sha256'] != head_ref['sha256'] or not ds_args['match_root']:
        raise ValueError('dataset cache/head contract differs')
    manifest = Path(cfg['validation']['roots_file'])
    cached_ds = getattr(datasets,cfg['dataset']['name'])(str(manifest), **ds_args)
    online_args = copy.deepcopy(ds_args);online_args['match_root'] = None
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
    frozen_head={k:v.detach().cpu().clone() for k,v in trainer.deform_conditioner.state_dict().items() if k not in trainable}
    if not frozen_head: raise ValueError('selected frozen head has no actual state to verify')
    trainer.deform_conditioner.load_state_dict(torch.load(args.conditioner_checkpoint,map_location='cpu',weights_only=True),strict=True)
    actual=trainer.deform_conditioner.state_dict()
    if any(not torch.equal(value,actual[key].detach().cpu()) for key,value in frozen_head.items()):
        raise ValueError('saved conditioner head differs from the selected frozen head checkpoint')
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
    trace_errors={};trainer.p_uncond=0
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        traces=[]
        for one in [cached_pack,online_pack]:
            transport=trainer._pop_deform_transport(dict(one))
            traces.append(trainer.deform_conditioner(**{k:v for k,v in transport.items() if k not in {'patch_grid','patch_ids'}}))
        torch.testing.assert_close(traces[0].candidate_indices,traces[1].candidate_indices,rtol=0,atol=0)
        for key in ['evidence','matching_logits','candidate_weights','view_weights','retarget_residual']:
            a,b=getattr(traces[0],key),getattr(traces[1],key)
            torch.testing.assert_close(a,b,rtol=0,atol=1e-5)
            trace_errors[key]=float((a-b).abs().max())
        if not traces[0].retarget_residual.abs().sum()>0:
            raise ValueError('actual trained retargeter is inactive on selected cases')
        cached=trainer.get_inference_cond(**cached_pack);online=trainer.get_inference_cond(**online_pack)
        train0=trainer.get_cond(**cached_pack)
        trainer.p_uncond=1.;train1=trainer.get_cond(**cached_pack)
        trainer.p_uncond=.5;np_state=np.random.get_state()
        try:
            np.random.seed(1);mixed=trainer.get_cond(**cached_pack)
        finally: np.random.set_state(np_state)
        expected_mask=(np.random.RandomState(1).rand(2)<.5).tolist()
        saved=trainer.deform_conditioner_config
        try:
            trainer.deform_conditioner_config=None
            native=trainer.get_inference_cond(**{k:v for k,v in cached_pack.items() if not k.startswith('transport_')})
        finally: trainer.deform_conditioner_config=saved
        checks=check_conditions(cached,online,train0,train1,mixed,native,expected_mask)
    if len(trainer.optimizer.state): raise ValueError('conditioning check applied an optimizer update')
    write_new(args.output/'summary.json',{'status':'PASS_CURRENT_C1_CACHE_AMP_CFG_CONTRACT',
        'amp':'bf16','sample_ids':[sid for sid,_,_ in selected],'selection':'first2 fixed-panel cases with actual target/source validity',
        'trained_denoiser':fingerprint(args.checkpoint),'trained_conditioner':fingerprint(args.conditioner_checkpoint),
        'selected_head':head_ref,'source_recipe':fingerprint(args.config),'manifest':fingerprint(manifest),
        'saved_frozen_head_exact_to_selected_checkpoint':True,'frozen_head_state_keys':sorted(frozen_head),
        'panel':fingerprint(args.panel_file),'candidate_indices_exact':True,'active_trace_max_abs_errors':trace_errors,
        'actual_cached_input_tensor_fingerprints':cached_inputs,
        'actual_online_input_tensor_fingerprints':online_inputs,'actual_four_RGB_cameras_exact':True,
        **checks,'optimizer_updates':0,'scope':'Two actual active original C1 cases, exact selected head and trained checkpoint; conditioning contract, not quality evaluation or all-asset parity.'})


if __name__ == '__main__': main()
