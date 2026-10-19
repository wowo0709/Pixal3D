"""Versioned SS evaluation using a pinned evaluator and deterministic reductions.

This file can run directly with PYTHONPATH/cwd pointing at a frozen source
checkout. No model/data/trainer file or existing evaluation is modified.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time

import torch


ENTRY = 'experiments.deformnext_retargeting.evaluate_native_checkpoint'


def validate_environment(environment):
    visible = environment.get('CUDA_VISIBLE_DEVICES', '').split(',')
    allowed = environment.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')
    if (len(visible) != 1 or not visible[0].isdigit() or visible[0] not in allowed
            or any(not value.isdigit() for value in allowed)):
        raise ValueError('one explicitly assigned physical GPU required')
    if environment.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('verified CUBLAS workspace must be set before CUDA initialization')


def validate_entry_path(path, root, expected_sha):
    expected = root.resolve() / 'experiments/deformnext_retargeting/evaluate_native_checkpoint.py'
    if path.resolve() != expected:
        raise ValueError('evaluator did not resolve inside its explicit frozen checkout')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('pinned evaluator source changed')


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--expected-stage-root', type=Path, required=True)
    parser.add_argument('--expected-entry-sha', required=True)
    parser.add_argument('--cuda-memory-fraction', type=float, default=1.)
    local, remaining = parser.parse_known_args()
    validate_environment(os.environ)
    if not 0 < local.cuda_memory_fraction <= 1:
        raise ValueError('CUDA allocator fraction must be in (0,1]')
    output = Path(remaining[remaining.index('--output') + 1])
    result_path = output / 'evaluation.json'
    if result_path.exists():
        raise FileExistsError('preserve the earlier evaluation')
    spec = importlib.util.find_spec(ENTRY)
    if spec is None or spec.origin is None:
        raise ValueError('explicit frozen evaluator unavailable')
    entry_path = Path(spec.origin)
    validate_entry_path(entry_path, local.expected_stage_root, local.expected_entry_sha)
    wrapper = Path(__file__)
    wrapper_sha = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(local.cuda_memory_fraction, 0)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    started, previous_arguments = time.monotonic(), sys.argv
    print(json.dumps({'status':'BEGIN_DETERMINISTIC_SS_EVALUATION','output':str(output),
        'physical_GPU':os.environ['CUDA_VISIBLE_DEVICES']}), flush=True)
    try:
        sys.argv = [str(entry_path), *remaining]
        with (output / 'evaluator.log').open('x') as log, contextlib.redirect_stdout(log):
            runpy.run_module(ENTRY, run_name='__main__')
    finally:
        sys.argv = previous_arguments
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('deterministic algorithms disabled during evaluation')
    validate_entry_path(entry_path, local.expected_stage_root, local.expected_entry_sha)
    if hashlib.sha256(wrapper.read_bytes()).hexdigest() != wrapper_sha:
        raise RuntimeError('evaluation adapter changed during execution')
    raw = result_path.read_bytes()
    result = json.loads(raw)
    if result['status'] != 'PASS' or result['stage'] != 'ss64' or result['optimizer_updates_during_evaluation'] != 0:
        raise ValueError('actual SS evaluator failed its protocol')
    result['deterministic_evaluation'] = {
        'protocol':'ss-deterministic-v1','actual_deterministic_algorithms':True,
        'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'CUDA_visible_physical_GPU':os.environ['CUDA_VISIBLE_DEVICES'],
        'CUDA_allocator_fraction':local.cuda_memory_fraction,
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'elapsed_seconds':time.monotonic()-started,
        'entry_source':str(entry_path),'entry_source_SHA256':local.expected_entry_sha,
        'adapter_SHA256':wrapper_sha,'pre_provenance_evaluation_SHA256':hashlib.sha256(raw).hexdigest(),
        'model_data_trainer_sources_modified':False,'optimizer_updates':0,
        'scope':'Fresh output namespace;unchanged saved weights,RGB protocol,GT/noise and sampler. Reproducibility of final metrics must be verified by separate paired replays.'}
    temporary = result_path.with_suffix('.provenance.tmp')
    with temporary.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,result_path)
    print(json.dumps({'status':'PASS_DETERMINISTIC_SS_EVALUATION','output':str(output),
        'assets':len(result['per_asset']),'metrics':result['quick_metrics'],
        'runtime':result['deterministic_evaluation']}), flush=True)


if __name__ == '__main__':
    main()
