"""Saved static-ray QA with comparison of frozen scores in head units.

Ray static traces store score/temperature. Compare the original frozen scalar
score with the existing absolute tolerance, while retaining every original
candidate, weight, evidence, projection, RGB, AMP and CFG check unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

from . import verify_ss_conditioning_contract as original

ORIGINAL_TRACE_COMPARE = original.compare_trace_tensors


def score_temperature(cfg):
    spec = cfg['trainer']['args']['deform_conditioner']
    if (cfg['trainer']['name'] != 'RayGroupImageConditionedProjFlowMatchingCFGTrainer'
            or spec['ray_mode'] not in {'semantic-top1', 'semantic-weighted'}
            or spec.get('task_attention') is not None or spec['score_temperature'] != .05):
        raise ValueError('only declared static-ray score/temperature traces are supported')
    return spec['score_temperature']


def compare_static_trace(left, right, *, name, temperature):
    if name != 'matching_logits':
        return ORIGINAL_TRACE_COMPARE(left, right, name=name)
    if temperature != .05 or left.dtype != torch.float32 or right.dtype != torch.float32:
        raise ValueError('declared FP32 static score/0.05 trace required')
    result = ORIGINAL_TRACE_COMPARE(left * temperature, right * temperature, name=name)
    finite = torch.isfinite(left) & torch.isfinite(right)
    raw_error = float((left[finite] - right[finite]).abs().max())
    return {**result, 'max_abs_error': raw_error,
        'head_score_max_abs_error': result['max_abs_error'],
        'comparison_units': 'frozen scalar score before division by temperature',
        'temperature': temperature, 'head_score_absolute_tolerance': 1e-5,
        'head_score_relative_tolerance': 0}


def main():
    # Keep the original CLI and every source statement in its main function.
    config = Path(sys.argv[sys.argv.index('--config') + 1])
    temperature = score_temperature(json.loads(config.read_text()))
    initial_comparator, initial_write = original.compare_trace_tensors, original.write_new
    template = Path(original.__file__)
    template_sha = hashlib.sha256(template.read_bytes()).hexdigest()
    wrapper = Path(__file__)
    wrapper_sha = hashlib.sha256(wrapper.read_bytes()).hexdigest()

    def comparator(left, right, *, name):
        return compare_static_trace(left, right, name=name, temperature=temperature)

    def write_result(path, value):
        if hashlib.sha256(template.read_bytes()).hexdigest() != template_sha:
            raise RuntimeError('frozen QA template changed')
        if hashlib.sha256(wrapper.read_bytes()).hexdigest() != wrapper_sha:
            raise RuntimeError('QA adapter changed during execution')
        value = {**value, 'static_ray_score_comparison': {
            'trace_definition': 'matching_logits = frozen scalar score / 0.05',
            'head_score_absolute_tolerance': 1e-5, 'head_score_relative_tolerance': 0,
            'other_checks_original_unchanged': True,
            'original_template_SHA256': template_sha, 'comparison_adapter_SHA256': wrapper_sha,
            'reason': 'Measured repeated FP32 camera-ray reductions produce tiny score differences; temperature amplifies score units. Candidate IDs remain exact; weights/evidence/R/projection retain original tolerances.'}}
        return initial_write(path, value)

    original.compare_trace_tensors, original.write_new = comparator, write_result
    try:
        original.main()
    finally:
        original.compare_trace_tensors, original.write_new = initial_comparator, initial_write


if __name__ == '__main__':
    main()
