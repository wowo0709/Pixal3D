"""Retain only selected frozen spatial rows in feature-head checkpoints.

The original module, candidates, probabilities and tensor operations inside
the head are unchanged. Full spatial key/value maps are unnecessary backward
inputs because they are frozen image features. Gathering before checkpoint
prevents those full maps from surviving the whole denoiser backward pass.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch


@contextmanager
def compact_spatial_checkpoint_inputs():
    from pixal3d.modules import member_feature_controls as module
    original = module.checkpoint
    statistics = dict(checkpoints=0, selected_rows=0, maximum_full_frozen_input_bytes=0,
                      cumulative_compact_input_bytes=0, evaluated_native_numerical_path_unchanged=True)

    def compact(function, *args, **kwargs):
        if function is not module._spatial_feature_chunk or not torch.is_grad_enabled():
            return original(function, *args, **kwargs)
        if len(args) != 9:
            raise ValueError('original nine spatial feature-head checkpoint arguments required')
        head, native, context, keys, values, indices, valid, prior, control = args
        if (keys.requires_grad or values.requires_grad or indices.dtype != torch.long
                or keys.shape != values.shape or keys.ndim != 3 or keys.shape[1:] != (3, 1024)
                or indices.ndim != 1 or len(indices) != len(native)
                or kwargs.get('use_reentrant') is not False):
            raise ValueError('frozen original spatial3 inputs and native nonreentrant checkpoint required')
        small_keys, small_values = keys[indices], values[indices]
        local = torch.arange(len(indices), device=indices.device, dtype=indices.dtype)
        statistics['checkpoints'] += 1
        statistics['selected_rows'] += len(indices)
        statistics['maximum_full_frozen_input_bytes'] = max(
            statistics['maximum_full_frozen_input_bytes'],
            keys.numel()*keys.element_size()+values.numel()*values.element_size())
        statistics['cumulative_compact_input_bytes'] += (
            small_keys.numel()*small_keys.element_size()+small_values.numel()*small_values.element_size())
        return original(function, head, native, context, small_keys, small_values,
                        local, valid, prior, control, **kwargs)

    try:
        module.checkpoint = compact
        yield statistics
    finally:
        module.checkpoint = original


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--native-entry', choices=['member_feature_control_runtime', 'native_source_seed_runtime'],
                        default='member_feature_control_runtime')
    local, remaining = parser.parse_known_args()
    config_arg = '--config' if '--config' in remaining else '--arm-config'
    config_path = Path(remaining[remaining.index(config_arg)+1])
    config = json.loads(config_path.read_text())
    if config['trainer']['args'].get('feature_control') != 'spatial-feature-attention':
        raise ValueError('explicit original spatial-feature control required')
    bound = {str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             str(config_path): hashlib.sha256(config_path.read_bytes()).hexdigest()}
    native = importlib.import_module('.'+local.native_entry, __package__)
    argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        with compact_spatial_checkpoint_inputs() as statistics:
            native.main()
    finally:
        sys.argv = argv
    for path, digest in bound.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('spatial storage adapter or bound recipe changed')
    phase = remaining[remaining.index('--phase')+1]
    if phase in {'native', 'evaluation'}:
        output = Path(remaining[remaining.index('--output')+1])
    else:
        output = Path(config['output_dir'])
    (output/'spatial-checkpoint-storage.json').write_text(json.dumps(dict(
        status='PASS_COMPLETED_COMPACT_SPATIAL_CHECKPOINT_RUNTIME', phase=phase,
        statistics=statistics, source_SHA256=bound,
        frozen_rows_gathered_before_checkpoint=True, candidates_or_weights_or_projection_changed=False,
        learned_parameters_or_optimizer_or_sampler_changed=False,
        byte_counters_are_cumulative_argument_sizes_not_measured_GPU_peak=True,
        capacity_and_quality_require_separate_actual_reports=True, full_goal_complete=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
