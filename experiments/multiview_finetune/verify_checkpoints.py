"""Verify saved raw/EMA weights and native resume state without CUDA."""
import argparse
import io
import json
import math
from pathlib import Path

import torch
from configure import STAGES


def load_state(path, weights_only):
    # Read NFS checkpoints sequentially; mmap scans incurred serial page-fault I/O.
    with Path(path).open('rb') as stream:
        return torch.load(io.BytesIO(stream.read()), map_location='cpu', weights_only=weights_only)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--stage', choices=STAGES, nargs='+', default=list(STAGES))
    parser.add_argument('--step', type=int, default=3, help='Saved update to verify; default: debug update 3')
    parser.add_argument('--configs-dir', type=Path, help='Explicit stage JSON directory; default: per-stage native config.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    for stage in args.stage:
        root = args.result / stage
        config_path = args.configs_dir / f'{stage}.json' if args.configs_dir else root / 'config.json'
        stage_config = json.loads(config_path.read_text())
        config = stage_config['trainer']['args']
        weights = {}
        for name in ['denoiser'] + [f'denoiser_ema{rate}' for rate in config['ema_rate']]:
            checkpoint = root / 'ckpts' / f'{name}_step{args.step:07d}.pt'
            state = load_state(checkpoint, weights_only=True)
            assert all(bool(torch.isfinite(value).all()) for value in state.values())
            weights[name] = {'tensors': len(state), 'parameters': sum(v.numel() for v in state.values()),
                             'dtypes': sorted({str(v.dtype) for v in state.values()}), 'finite': True}
            if name == 'denoiser':
                raw = state
        pretrained = load_state(config['finetune_ckpt']['denoiser'], weights_only=True)
        names = [key for key in raw if key in pretrained and raw[key].is_floating_point()]
        selected = names[:10] + names[-10:]
        changed = {key: float((raw[key] - pretrained[key]).abs().max()) for key in selected}
        lora_report = None
        if config.get('lora') is not None:
            # Compare after the intentional BF16 cast, not against FP32 rounding.
            for key in names:
                assert torch.equal(raw[key], pretrained[key].to(raw[key].dtype)), f'Frozen base changed: {key}'
            adapters = [key for key in raw if key.endswith(('.lora_A', '.lora_B'))]
            assert adapters and any(bool(raw[key].abs().sum() > 0) for key in adapters if key.endswith('.lora_B'))
            lora_report = {'frozen_base_unchanged': True, 'adapter_tensors': len(adapters),
                           'adapter_parameters': sum(raw[key].numel() for key in adapters),
                           'maximum_adapter_B': max(float(raw[key].abs().max()) for key in adapters if key.endswith('.lora_B'))}
        else:
            assert any(value > 0 for value in changed.values())
        misc = load_state(root / 'ckpts' / f'misc_step{args.step:07d}.pt', weights_only=False)
        assert misc['step'] == misc['lr_scheduler']['last_epoch'] == args.step
        epoch_updates = stage_config['experiment']['updates_per_epoch']
        # Native cycle increments epoch only when requesting the next batch.
        sampler_epoch = (args.step - 1) // epoch_updates
        assert misc['data_sampler']['epoch'] == sampler_epoch
        assert misc['data_sampler']['idx'] == (args.step - sampler_epoch * epoch_updates) * config['batch_size_per_gpu']
        expected_lr = config['optimizer']['args']['lr'] * min(1.0, (args.step + 1) / config['lr_scheduler']['args']['warmup_steps'])
        actual_lr = misc['optimizer']['param_groups'][0]['lr']
        assert math.isclose(actual_lr, expected_lr)
        tensors = [value for state in misc['optimizer']['state'].values() for key, value in state.items()
                   if isinstance(value, torch.Tensor) and key != 'step']
        assert all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in tensors)
        report = {'stage': stage, 'checkpoint_step': args.step, 'weights': weights,
                  'changed_from_official_pretrained': changed, 'sampler': misc['data_sampler'],
                  'scheduler_last_epoch': args.step, 'lr_next_update': actual_lr,
                  'optimizer_state': 'finite FP32', 'cuda_initialized': torch.cuda.is_initialized()}
        if lora_report is not None:
            report['lora'] = lora_report
        assert not report['cuda_initialized']
        (root / 'checkpoint_integrity.json').write_text(json.dumps(report, indent=2) + '\n')
        print(stage, 'finite weights/FP32 optimizer; adapter/base or full-finetune checks and resume state passed', flush=True)


if __name__ == '__main__':
    main()
