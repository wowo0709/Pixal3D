"""Frozen SS diagnostic separating nearest-slot writes from native ProjGrid.

GT addresses are used only as a diagnostic. All four RGB views, the native
global condition, denoiser weights and per-asset sampling noise remain fixed.
"""
import argparse
import copy
import json
from pathlib import Path
import types

import numpy as np
import torch

from .diagnose_matching_v2 import OracleDataset


def project_patchmap(extractor, patches, camera):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import compute_multiview_projection_matrices
    projection, _ = compute_multiview_projection_matrices(
        camera['transform_matrix'], camera['distance'], extractor.fixed_projection_transform)
    if patches.ndim != 3 or patches.shape[1] != 1024:
        raise ValueError('SS transport diagnostic requires the actual 32x32 patch map')
    return extractor.proj_grid(patches.reshape(patches.shape[0], 32, 32, -1),
        camera['camera_angle_x'][:, 0], camera['distance'][:, 0],
        camera['mesh_scale'], projection[:, 0])


def normalized_evidence(project, values, valid):
    coverage = project(valid[..., None].to(values.dtype)).clamp(0, 1)
    evidence = project(values * valid[..., None]) / coverage.clamp_min(1e-8)
    return evidence, coverage


class DiagnosticDataset(OracleDataset):
    def __init__(self,dataset,labels,oracle=True,teacher_support=False):
        super().__init__(dataset,labels);self.oracle=oracle;self.teacher_support=teacher_support

    def __getitem__(self, index):
        item = super().__getitem__(index) if self.oracle else self.dataset[index]
        row = self.rows[index]
        if self.teacher_support and not self.oracle:
            with np.load(self.labels/(row['sample_id']+'.npz')) as z:
                if str(z['sample_id'])!=row['sample_id'] or str(z['split'])!=row['split']:
                    raise ValueError('diagnostic support teacher identity differs')
                positive=torch.from_numpy(z['positive_indices'].copy()).long()
                visible=torch.from_numpy(z['positive_visible'].copy()).bool()
            item['transport_target_valid']=visible.any(-1)
            valid=item['transport_source_valid'].flatten().clone()
            valid[positive[visible]]=True
            item['transport_source_valid']=valid.reshape_as(item['transport_source_valid'])
        with np.load(self.dataset._resolve(row['transport_cache'])) as z:
            if str(z['conditioning_input_sha256']) != row['conditioning_input_sha256']:
                raise ValueError('target query RGB fingerprint differs')
            item['transport_target_query'] = torch.from_numpy(z['target_query'].copy())
        return item


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config-root', 'checkpoint', 'panels', 'labels', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--calibration', action='store_true')
    p.add_argument('--arm', choices=['b3','c1'],default='b3')
    p.add_argument('--conditioner-checkpoint',type=Path)
    p.add_argument('--correspondence',choices=['gt','native'],default='gt')
    p.add_argument('--support',choices=['native','teacher-visible'],default='native')
    p.add_argument('--scale', type=float)
    p.add_argument('--generation-batch', type=int, default=8)
    p.add_argument('--cases', default='native,nearest,bilinear,normalized,wrong-object')
    a = p.parse_args()
    if a.scale is not None and not 0 < a.scale < 1:
        raise ValueError('a fixed nonzero diagnostic scale in (0,1) is required')
    if a.arm=='c1' and a.conditioner_checkpoint is None:
        raise ValueError('trained C1 diagnostic requires the paired conditioner checkpoint')
    if a.arm=='b3' and a.scale is None and a.conditioner_checkpoint is None:
        raise ValueError('untrained B3 transport requires an explicit nonzero diagnostic scale')
    cases = a.cases.split(',')
    if set(cases) - {'native','nearest','bilinear','normalized','wrong-object'}:
        raise ValueError('unknown projection diagnostic')
    from pixal3d import datasets, models, trainers
    from .quick_metrics import evaluate_quick
    from PIL import Image
    torch.set_num_threads(4); torch.cuda.set_device(0)
    cfg = json.loads((a.config_root / f'ss64-{a.arm}.json').read_text())
    args = copy.deepcopy(cfg['validation']['dataset_args'])
    manifest = cfg['validation']['roots_file']
    if a.calibration:
        manifest = str(a.panels / 'head_calibration.jsonl'); args['expected_split'] = 'train'
    raw = getattr(datasets, cfg['dataset']['name'])(manifest, **args)
    ds = DiagnosticDataset(raw, a.labels,oracle=a.correspondence=='gt',teacher_support=a.support=='teacher-visible')
    quick = cfg['validation']['quick_metrics']
    quick.update(panel_file=str(a.panels / ('calibration-32.json' if a.calibration else 'panel-32.json')),
        seed_map_file=str(a.panels / ('calibration-seeds.json' if a.calibration else 'seed-map.json')),
        generation_batch_size=a.generation_batch, log_progress=True)
    with torch.device('meta'):
        md = {n: getattr(models, s['name'])(**s['args']) for n,s in cfg['models'].items()}
    kw = copy.deepcopy(cfg['trainer']['args'])
    kw.update(continuation_ckpt={'denoiser':str(a.checkpoint)},batch_size_per_gpu=1,batch_split=1,
        num_workers=0,prefetch_data=False,max_steps=1)
    a.output.mkdir(parents=True,exist_ok=True)
    trainer = getattr(trainers,cfg['trainer']['name'])(md,raw,**kw,output_dir=str(a.output),
        load_dir=None,step=None,wandb_run=None)
    trainer.models['denoiser'].eval(); trainer.p_uncond=0
    if a.conditioner_checkpoint:
        trainer.deform_conditioner.load_state_dict(torch.load(a.conditioner_checkpoint,map_location='cpu',weights_only=True),strict=True)
    if a.scale is not None:trainer.deform_conditioner.fusion_gate.data.fill_(np.arctanh(a.scale))
    a.scale=float(torch.tanh(trainer.deform_conditioner.fusion_gate.detach()))
    if a.scale==0:raise ValueError('a zero transport gate cannot test projection utility')
    encode = trainer._encode_deform_views; inject = trainer._inject_deform_transport
    state = {'case':'native','parity':[]}

    def wrapped_encode(self, images, camera_info, transport, **kwargs):
        state['camera'] = camera_info
        return encode(images,camera_info,transport,**kwargs)

    def wrapped_inject(self, condition, transport, coords, *, source_views, control, anchor_condition):
        if coords is not None:
            raise ValueError('this audit is restricted to dense SS')
        camera = state['camera']; extractor=self._image_cond_module
        project=lambda values:project_patchmap(extractor,values,camera)
        key='proj'; native=condition[key]
        if state['case']=='native':
            # A self map is a projection-only check, not a target-only model arm.
            query=transport['target_query'].float()
            near=query.gather(1,transport['patch_grid'].flatten(1)[...,None].expand(-1,-1,1024))
            linear=project(query)
            anchor=anchor_condition[key].float(); rms=anchor.square().mean().sqrt()
            state['parity'].append({'nearest_relative_rmse':float((near-anchor).square().mean().sqrt()/rms),
                'native_bilinear_relative_rmse':float((linear-anchor).square().mean().sqrt()/rms),
                'anchor_rms':float(rms)})
            return condition
        if state['case']=='nearest':
            return inject(condition,transport,coords,source_views=source_views,
                control=control,anchor_condition=anchor_condition)
        call={k:v for k,v in transport.items() if k not in {'patch_grid','patch_ids'}}
        if state['case']=='wrong-object':
            if call['source_values'].shape[0]<2:
                raise ValueError('wrong-object requires a batch of distinct assets')
            call['source_values']=torch.roll(call['source_values'],1,0)
        result=self.deform_conditioner(**call)
        residual=result.retarget_residual
        residual=torch.zeros_like(result.evidence) if residual is None else residual
        combined=torch.cat((result.evidence-residual,residual),-1)
        if state['case']=='normalized':
            value,coverage=normalized_evidence(project,combined,result.query_valid)
            valid=coverage>0; usage=coverage
        else:
            value=project(combined)
            valid=result.query_valid.gather(1,transport['patch_grid'].flatten(1))[...,None]
            usage=1
        value,residual=value.split((1024,1024),-1)
        semantic=(anchor_condition[key]+source_views*value.to(native.dtype))/(source_views+1)
        update=(result.fusion_scale.to(native.dtype)*(semantic-native)
            +source_views/(source_views+1)*residual.to(native.dtype))*usage
        condition[key]=native+torch.where(valid,update,torch.zeros_like(update))
        return condition

    trainer._encode_deform_views=types.MethodType(wrapped_encode,trainer)
    trainer._inject_deform_transport=types.MethodType(wrapped_inject,trainer)
    results={}
    for case in cases:
        state['case']=case;out=a.output/case;out.mkdir(exist_ok=True)
        with torch.inference_mode():
            metrics,panel,ids=evaluate_quick(trainer,ds,cfg['validation'])
        result={'status':'PASS','case':case,'diagnostic_only':True,'GT_addresses':a.correspondence=='gt',
            'optimizer_updates':0,'checkpoint':str(a.checkpoint),'scale':a.scale,
            'arm':a.arm,'conditioner_checkpoint':str(a.conditioner_checkpoint) if a.conditioner_checkpoint else None,
            'support':'teacher-visible' if a.correspondence=='gt' or a.support=='teacher-visible' else 'native',
            'split':'train-calibration' if a.calibration else 'validation',
            'sampling_steps':12,'guidance':3.,'sample_ids':ids,
            'quick_metrics':metrics,'per_asset':trainer._quick_per_asset}
        (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
        Image.fromarray(panel).save(out/'panel.png');results[case]=metrics
        print(json.dumps({'case':case,'metrics':metrics}),flush=True)
    (a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    report={'status':'finished','cases':results,'self_projection':state['parity'],
        'fixed_scale':a.scale,'selection_scope':'train-calibration' if a.calibration else 'validation',
        'GT_diagnostic_not_deployable':a.correspondence=='gt','original_arms_unchanged':True,
        'arm':a.arm,'correspondence':a.correspondence}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
