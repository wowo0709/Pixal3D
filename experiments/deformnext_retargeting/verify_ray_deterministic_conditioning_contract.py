"""Run every original ray saved-model assertion with reproducible GPU reductions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from . import verify_ss_conditioning_contract as original


def validate_environment(config, environment):
    if config['trainer']['name'] != 'RayGroupImageConditionedProjFlowMatchingCFGTrainer':
        raise ValueError('explicit ray-group saved-model QA required')
    visible = environment.get('CUDA_VISIBLE_DEVICES', '').split(',')
    allowed = environment.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')
    if (len(visible) != 1 or not visible[0].isdigit()
            or visible[0] not in allowed or any(not value.isdigit() for value in allowed)):
        raise ValueError('one explicitly assigned physical GPU required')
    if environment.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('the verified deterministic CUBLAS workspace must be set before CUDA starts')


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--cuda-memory-fraction', type=float, default=1.)
    local, remaining = parser.parse_known_args()
    if not 0 < local.cuda_memory_fraction <= 1:
        raise ValueError('CUDA allocator fraction must be in (0,1]')
    config_path = Path(remaining[remaining.index('--config') + 1])
    validate_environment(json.loads(config_path.read_text()), os.environ)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(local.cuda_memory_fraction, 0)
    torch.use_deterministic_algorithms(True)
    template, wrapper = Path(original.__file__), Path(__file__)
    template_sha = hashlib.sha256(template.read_bytes()).hexdigest()
    wrapper_sha = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    initial_write, initial_arguments = original.write_new, sys.argv

    def write_result(path, value):
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError('deterministic algorithms were disabled during QA')
        if hashlib.sha256(template.read_bytes()).hexdigest() != template_sha:
            raise RuntimeError('original QA template changed')
        if hashlib.sha256(wrapper.read_bytes()).hexdigest() != wrapper_sha:
            raise RuntimeError('deterministic QA adapter changed')
        value = {**value, 'deterministic_ray_QA': {
            'actual_deterministic_algorithms': True,
            'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'CUDA_visible_physical_GPU': os.environ['CUDA_VISIBLE_DEVICES'],
            'CUDA_allocator_fraction': local.cuda_memory_fraction,
            'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
            'original_trace_and_all_other_assertions_unchanged': True,
            'original_template_SHA256': template_sha, 'adapter_SHA256': wrapper_sha,
            'scope': 'Functional saved-model parity with deterministic reductions;not a quality evaluation or bit-exact claim for legacy nondeterministic generation.'}}
        return initial_write(path, value)

    original.write_new, sys.argv = write_result, [str(wrapper), *remaining]
    try:
        original.main()
    finally:
        original.write_new, sys.argv = initial_write, initial_arguments


if __name__ == '__main__':
    main()
