"""Run unchanged official GPU metrics on explicitly bound final508 pixels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np

from .bind_generated_pbr_dataset import load_QA
from .cascade_view_roles import ClosedInputs, require
from .evaluate_bound_population import pinned, sha
from .perceptual_image_inputs import paired_images
from .summarize_frozen_inception_pools import validate_frozen_models


def validate_observations(plan, *, ids, clusters):
    alias = plan['frozen_alias']
    expected = [(alias, sid, clusters[sid], cid) for sid in ids for cid in (8,9,10)]
    actual = [(row['arm'], row['sample_id'], row['source_cluster_id'], row['camera_id']) for row in plan['inputs']]
    require(len(ids) == 508 and len(set(clusters.values())) == 134
            and actual == expected and len({r['shared_npz'] for r in plan['inputs']}) == 1524,
            'exactly1524 distinct ordered final508 model/camera images required')


def validate_inputs(plan):
    require(plan['scope'] == 'final-test508' and plan['full_panel'], 'explicit full final-test508 perceptual panel required')
    freeze = json.loads(pinned(plan['model_freeze']).read_text())
    ids, clusters, _ = validate_frozen_models(freeze, freeze['models'])
    alias = plan['frozen_alias']
    require(alias in freeze['models'], 'perceptual model must be development-frozen')
    for evidence in freeze['development_evidence']: pinned(evidence)
    compiled = json.loads(pinned(plan['native_case_manifest']).read_text())
    require(compiled['status'] == 'PREPARED_ALL_FROZEN_TEST508_NATIVE_CASCADE_CASES'
            and compiled['scope'] == 'final-test508' and tuple(compiled['models']) == tuple(freeze['models'])
            and compiled['model_freeze'] == plan['model_freeze']
            and plan['manifest']['SHA256'] == freeze['population']['manifest_SHA256']
            and compiled['source_inputs']['manifest'] == plan['manifest']
            and compiled['source_inputs']['cache_QA'] == plan['completed_cache_QA']
            and compiled['source_inputs']['cache_QA_plan'] == plan['cache_QA_plan'],
            'official metrics must bind the completed native case population')
    load_QA(dict(cache_QA=plan['completed_cache_QA'], cache_QA_plan=plan['cache_QA_plan'], manifest=plan['manifest']))
    evaluator = Path(__file__).with_name('evaluate_official_perceptual.py')
    require(pinned(plan['evaluator_source']).resolve() == evaluator.resolve()
            and pinned(plan['wrapper_source']).resolve() == Path(__file__).resolve(), 'explicit unchanged official evaluator/wrapper source required')
    registry = pinned(plan['metric_registry'])
    require(registry.resolve() == (Path(plan['metric_models'])/'registry.json').resolve(), 'same explicit official metric registry required')
    validate_observations(plan, ids=ids, clusters=clusters)
    closed = ClosedInputs(plan['canonical_local_root'])
    for row in plan['inputs']:
        path = row['shared_npz']
        require(closed.check(path, plan['closed_inputs'][path]) == row['shared_npz_SHA256'], 'actual perceptual image archive changed')
        with np.load(path, allow_pickle=False) as arrays:
            _, audit = paired_images(arrays['prediction'], arrays['GT_shared_renderer'], arrays['GT_alpha'])
        require(audit == row['image_audit'], 'actual fullframe/GT-only-crop pixel identity changed')
    return freeze


def verify_result(plan, plan_spec, report, binding):
    require(binding['status'] == 'PASS_ACTUAL_BOUND_OFFICIAL_TEST508_PERCEPTUAL_METRICS'
            and binding['executed_plan'] == plan_spec and binding['model_freeze'] == plan['model_freeze']
            and binding['alias'] == plan['frozen_alias'] and binding['evaluator_source'] == plan['evaluator_source']
            and binding['wrapper_source'] == plan['wrapper_source'] and binding['views'] == 1524
            and binding['physical_GPU'] in {'0','1','2','3'}
            and binding['evaluation_SHA256'] == sha(Path(plan['output'])/'evaluation.json'),
            'completed official metric execution binding differs')
    require(report['status'] == 'PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION'
            and report['scope'] == 'final-test508' and report['full_panel'] and report['fullframe_primary']
            and report['metric_registry_SHA256'] == plan['metric_registry']['SHA256']
            and report['model_precision'] == 'float32,TF32off,noAMP'
            and report['GPU_sanity']['status'] == 'PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY'
            and len(report['per_view']) == len(plan['inputs']) == 1524,
            'completed full508 official metric protocol differs')
    for expected, actual in zip(plan['inputs'], report['per_view']):
        require(all(expected[key] == actual[key] for key in
                    ['arm','sample_id','source_cluster_id','camera_id','shared_npz_SHA256','image_audit']),
                'completed perceptual score belongs to a different model/sample/camera/pixel array')


def run(plan_spec):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(visible in {'0','1','2','3'} and visible in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','),
            'one explicitly assigned physical GPU0-3 required')
    path = pinned(plan_spec); plan = json.loads(path.read_text())
    validate_inputs(plan)
    output = Path(plan['output'])
    if output.exists(): raise FileExistsError('preserve existing official metric outputs')
    previous = sys.argv
    try:
        sys.argv = [str(pinned(plan['evaluator_source'])), '--plan', str(path)]
        runpy.run_module('experiments.deformnext_retargeting.evaluate_official_perceptual', run_name='__main__')
    finally:
        sys.argv = previous
    pinned(plan_spec); pinned(plan['evaluator_source']); pinned(plan['wrapper_source']); pinned(plan['metric_registry'])
    # Immutable images are checked again after the actual GPU forwards.
    validate_inputs(plan)
    report = json.loads((output/'evaluation.json').read_text())
    binding = dict(status='PASS_ACTUAL_BOUND_OFFICIAL_TEST508_PERCEPTUAL_METRICS', executed_plan=plan_spec,
        model_freeze=plan['model_freeze'], alias=plan['frozen_alias'], views=1524, physical_GPU=visible,
        evaluator_source=plan['evaluator_source'], wrapper_source=plan['wrapper_source'],
        evaluation_SHA256=sha(output/'evaluation.json'), native_metric_code_unchanged=True,
        actual_image_bytes_checked_before_and_after_GPU=True, full_goal_complete=False)
    verify_result(plan, plan_spec, report, binding)
    with (output/'binding-audit.json').open('x') as stream: json.dump(binding, stream, indent=2); stream.write('\n')
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(run(dict(path=str(args.plan), SHA256=args.plan_sha256))), flush=True)


if __name__ == '__main__': main()
