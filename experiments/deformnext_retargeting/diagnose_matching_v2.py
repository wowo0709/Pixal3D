"""Frozen-checkpoint SS controls; GT visibility is diagnostic-only."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch


class OracleDataset:
    def __init__(self, dataset, labels):
        self.dataset, self.labels = dataset, labels
    def __getattr__(self, name):
        return getattr(self.dataset, name)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        item = self.dataset[index]
        row = self.rows[index]
        with np.load(self.labels / (row['sample_id'] + '.npz')) as label:
            if str(label['sample_id']) != row['sample_id'] or str(label['split']) != row['split']:
                raise ValueError('diagnostic teacher identity mismatch')
            positive = torch.from_numpy(label['positive_indices'].copy()).long()
            visible = torch.from_numpy(label['positive_visible'].copy()).bool()
        indices = torch.zeros(1024, 64, dtype=torch.long)
        valid = torch.zeros(1024, 64, dtype=torch.bool)
        indices[:, :3], valid[:, :3] = positive, visible
        item['transport_semantic_indices'] = indices
        item['transport_candidate_valid'] = valid
        item['transport_matching_logits'] = torch.zeros(1024, 64).masked_fill(~valid, -torch.inf)
        item['transport_target_valid'] = visible.any(-1)
        source_valid = item['transport_source_valid'].flatten().clone()
        source_valid[positive[visible]] = True
        item['transport_source_valid'] = source_valid.reshape_as(item['transport_source_valid'])
        return item


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--conditioner-checkpoint', type=Path)
    p.add_argument('--group', choices=['initializer', 'v1', 'b3', 'c1', 'baseline'], required=True)
    p.add_argument('--arm', choices=['b1','b2','b3','c1','c2'])
    p.add_argument('--cases')
    p.add_argument('--calibration', action='store_true')
    p.add_argument('--oracle-scales')
    p.add_argument('--oracle-gate-file',type=Path)
    p.add_argument('--panels', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--panel-size', type=int, default=128)
    p.add_argument('--generation-batch', type=int, default=8)
    a = p.parse_args()
    from pixal3d import datasets, models, trainers
    from .quick_metrics import evaluate_quick
    from .conditioning_controls import ConditioningControl
    from PIL import Image
    torch.set_num_threads(4); torch.cuda.set_device(0)
    arm = a.arm or ('v1' if a.group == 'initializer' else ('c1' if a.group == 'c1' else 'b3'))
    cfg = json.loads((a.config_root / f'ss64-{arm}.json').read_text())
    dataset_args=copy.deepcopy(cfg['validation']['dataset_args'])
    manifest=cfg['validation']['roots_file']
    if a.calibration:
        manifest=str(a.panels/'head_calibration.jsonl');dataset_args['expected_split']='train'
    ds = getattr(datasets, cfg['dataset']['name'])(manifest, **dataset_args)
    cfg['validation']['quick_metrics'].update(panel_file=str(a.panels / f'panel-{a.panel_size}.json'), seed_map_file=str(a.panels / 'seed-map.json'), generation_batch_size=a.generation_batch, log_progress=True)
    if a.calibration:
        cfg['validation']['quick_metrics'].update(panel_file=str(a.panels/'calibration-32.json'),seed_map_file=str(a.panels/'calibration-seeds.json'))
    with torch.device('meta'):
        md = {name: getattr(models, spec['name'])(**spec['args']) for name, spec in cfg['models'].items()}
    kw = copy.deepcopy(cfg['trainer']['args'])
    kw.update(continuation_ckpt={'denoiser': str(a.checkpoint)}, batch_size_per_gpu=1, batch_split=1, num_workers=0, prefetch_data=False, max_steps=1)
    a.output.mkdir(parents=True, exist_ok=True)
    trainer = getattr(trainers, cfg['trainer']['name'])(md, ds, **kw, output_dir=str(a.output), load_dir=None, step=None, wandb_run=None)
    trainer.models['denoiser'].eval(); trainer.p_uncond = 0
    module = getattr(trainer, 'deform_conditioner', None)
    if a.conditioner_checkpoint:
        module.load_state_dict(torch.load(a.conditioner_checkpoint, map_location='cpu', weights_only=True), strict=True)
    native_gate = float(module.fusion_gate.detach()) if module is not None else 0.
    cases = {'initializer': ['edited-rgb', 'source-only'],
             'v1': ['native', 'current', 'oracle'],
             'b3': ['normal', 'source-off', 'wrong-corr', 'oracle'],
             'c1': ['normal', 'adapter-off', 'source-off', 'oracle', 'wrong-object'], 'baseline':['normal']}[a.group]
    if a.cases:
        selected=a.cases.split(',')
        if any(case not in cases for case in selected):raise ValueError('unsupported case for diagnostic group')
        cases=selected
    if a.oracle_scales or a.oracle_gate_file:
        scales=[float(v) for v in a.oracle_scales.split(',')] if a.oracle_scales else [json.loads(a.oracle_gate_file.read_text())['scale']]
        if any(not 0<s<1 for s in scales):raise ValueError('diagnostic fusion scales must be between0 and1')
        cases=['native']+[f'oracle-{s}' for s in scales]
    original_forward = module.forward if module is not None else None
    results = {}
    for case in cases:
        out = a.output / case; out.mkdir(exist_ok=True)
        if (out / 'evaluation.json').exists():
            results[case] = json.loads((out / 'evaluation.json').read_text()); continue
        current = OracleDataset(ds, a.labels) if case.startswith('oracle') else ds
        if case == 'source-only': current = ConditioningControl(ds, 'source-only')
        hook = None
        if module is not None:
            module.fusion_gate.data.fill_(0. if case == 'native' else (np.arctanh(.04306826368) if a.group == 'v1' else native_gate))
            if case.startswith('oracle-'):module.fusion_gate.data.fill_(np.arctanh(float(case.removeprefix('oracle-'))))
            control = case if case in ('source-off', 'wrong-corr', 'wrong-object') else 'normal'
            def forward(*args, _control=control, **kwargs):
                if _control=='wrong-object':
                    if kwargs['source_values'].shape[0]<2:raise ValueError('wrong-object diagnostic requires batched distinct assets')
                    kwargs['source_values']=torch.roll(kwargs['source_values'],1,0)
                    kwargs['control']='normal'
                else:kwargs['control'] = _control
                return original_forward(*args, **kwargs)
            module.forward = forward
            if case == 'adapter-off':
                hook = module.retargeter.register_forward_hook(lambda m, inputs, output: torch.zeros_like(output))
        try:
            with torch.no_grad():
                metrics, panel, ids = evaluate_quick(trainer, current, cfg['validation'])
            record = {'status': 'PASS', 'group': a.group, 'case': case, 'checkpoint': str(a.checkpoint),
                      'conditioner_checkpoint': str(a.conditioner_checkpoint) if a.conditioner_checkpoint else None,
                      'optimizer_updates_during_evaluation': 0, 'diagnostic_only': case.startswith('oracle'),
                      'split':'train-calibration' if a.calibration else 'validation',
                      'quick_metrics': metrics, 'sample_ids': ids,
                      'per_asset': trainer._quick_per_asset,
                      'semantic_gate': float(torch.tanh(module.fusion_gate.detach())) if module is not None else None,
                      'sampling_steps': 12, 'guidance': 3., 'generation_batch_size':a.generation_batch,'input_camera_ids': [8, 9, 10, 11]}
            (out / 'evaluation.json').write_text(json.dumps(record, indent=2) + '\n')
            Image.fromarray(panel).save(out / 'panel.png')
            results[case] = record
            print(json.dumps({'group': a.group, 'case': case, 'metrics': metrics}), flush=True)
        finally:
            if hook is not None: hook.remove()
    (a.output / 'summary.json').write_text(json.dumps({k: v['quick_metrics'] for k, v in results.items()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
