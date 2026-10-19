"""Inspect actual frozen-utility Adam moments without changing training."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from .semantic_topk_search import write_new


def fingerprint(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def inspect(config, checkpoint, step):
    cfg = json.loads(config.read_text())
    spec = cfg['trainer']['args']['deform_conditioner']
    task = spec.get('task_attention') or {}
    if (spec.get('arm') != 'c1' or spec.get('implementation') != 'r9-native-v5'
            or task.get('content_mode') != 'residual-zero'
            or task.get('train_decoder') is not False or task.get('train_score') is not False):
        raise ValueError('explicit frozen-matcher C1 residual-utility profile required')
    module = DeformationAwareConditioner(**spec)
    named = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if step <= 0 or payload['step'] != step:
        raise ValueError('actual checkpoint step differs')
    opt = payload['optimizer']
    groups = [g for g in opt['param_groups'] if g['model_name'] == 'deform_conditioner']
    if len(groups) != 1 or len(named) != 11 or len(groups[0]['params']) != 11:
        raise ValueError('declared original11-parameter conditioner group required')
    group = groups[0]
    if len(opt['state']) != 371:
        raise ValueError('actual360 LoRA plus11 conditioner states required')
    for state in opt['state'].values():
        if (float(state['step']) != step or not torch.isfinite(state['exp_avg']).all()
                or not torch.isfinite(state['exp_avg_sq']).all() or not (state['exp_avg_sq'] >= 0).all()):
            raise ValueError('nonfinite/different actual optimizer state')
    b1, b2 = group['betas']
    eps = group['eps']
    rows = []
    for (name, parameter), pid in zip(named, group['params']):
        state = opt['state'][pid]
        if state['exp_avg'].shape != parameter.shape:
            raise ValueError('actual optimizer enumeration/shape differs: ' + name)
        rms = (state['exp_avg_sq'].double() / (1 - b2 ** step)).sqrt()
        damping = rms / (rms + eps)
        mhat = state['exp_avg'].double() / (1 - b1 ** step)
        direction = mhat / (rms + eps)
        rows.append({
            'parameter': name, 'elements': parameter.numel(),
            'median_bias_corrected_gradient_RMS': float(rms.median()),
            'fraction_RMS_below_epsilon': float((rms < eps).double().mean()),
            'median_epsilon_damping_factor': float(damping.median()),
            'normalized_Adam_direction_RMS': float(direction.square().mean().sqrt()),
        })
    repo = Path(__file__).resolve().parents[2]
    return {
        'status': 'PASS_ACTUAL_FROZEN_UTILITY_OPTIMIZER_MOMENT_INSPECTION',
        'actual_step': step, 'group_learning_rate': group['lr'],
        'actual_epsilon': eps, 'actual_betas': group['betas'],
        'parameter_rows': rows, 'all371_optimizer_states_finite_and_step_matched': True,
        'inputs': {'recipe': fingerprint(config), 'checkpoint': fingerprint(checkpoint)},
        'driver': fingerprint(Path(__file__)),
        'group_enumeration_source': fingerprint(repo / 'pixal3d/trainers/basic.py'),
        'conditioner_source': fingerprint(repo / 'pixal3d/modules/deform_conditioner.py'),
        'mapping': 'BasicTrainer enumerates trainable model.parameters();all11 corresponding actual state shapes checked.',
        'scope': 'Bias-corrected accumulated second-moment RMS and epsilon-normalizer damping atone actual checkpoint. Not raw single-step gradients or a counterfactual retraining outcome. Training is unchanged.',
        'optimizer_updates_during_inspection': 0, 'GPU_used': False,
        'model_adopted': False, 'full_goal_complete': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve actual optimizer inspection')
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    result = inspect(args.config, args.checkpoint, args.step)
    write_new(args.output, result)
    print(json.dumps({'status': result['status'], 'step': result['actual_step'],
                      'utility_rows': [row for row in result['parameter_rows']
                                       if row['parameter'].startswith('task_attention.')]}), flush=True)


if __name__ == '__main__':
    main()
