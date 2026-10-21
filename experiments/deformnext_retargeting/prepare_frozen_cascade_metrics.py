"""Bind completed native508 cases to official perceptual and eight-metric plans.

This entrypoint reads completed, canonically closed predictions. It never
selects a model, renders an image, computes a neural feature or changes a
native numerical plan. Missing cases and failed outputs are preserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bind_generated_pbr_dataset import load_QA
from .cascade_view_roles import ClosedInputs, require
from .evaluate_bound_population import pinned, sha
from .frozen_cascade_metrics import PinnedFiles, read_terminal
from .perceptual_image_inputs import paired_images
from .summarize_frozen_inception_pools import validate_frozen_models


PACKAGE = 'experiments.deformnext_retargeting.'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')
    return dict(path=str(path), SHA256=sha(path))


def image_input(path, *, alias, sample_id, cluster, closed):
    """Fingerprint the actual shared RGB/GT/alpha consumed by every metric."""
    path = Path(path)
    digest = closed.check(path, path.parent/'file4-closure.json')
    with np.load(path, allow_pickle=False) as arrays:
        require(all(arrays[key].dtype == np.float32 for key in
                    ['prediction', 'prediction_alpha', 'GT_shared_renderer', 'GT_alpha'])
                and arrays['prediction'].shape == arrays['GT_shared_renderer'].shape == (3, 512, 512)
                and arrays['prediction_alpha'].shape == arrays['GT_alpha'].shape == (512, 512),
                'actual native512 FP32 paired RGB/alpha required')
        _, audit = paired_images(arrays['prediction'], arrays['GT_shared_renderer'], arrays['GT_alpha'])
    return dict(arm=alias, sample_id=sample_id, source_cluster_id=cluster,
                camera_id=int(path.stem.removeprefix('camera')), shared_npz=str(path),
                shared_npz_SHA256=digest, image_audit=audit)


def bind_case(spec, *, alias, model, source_inputs, prepared_receipt,
              SS_receipt, closed):
    path = pinned(spec)
    closed.check(path, prepared_receipt)
    case = json.loads(path.read_text())
    require(case['alias'] == alias and case['model_freeze'] == source_inputs['model_freeze']
            and case['source_inputs'] == source_inputs, 'completed case source binding differs')
    phases = {phase['name']: phase for phase in case['phases']}
    names = ['generation', 'PBR', 'render', 'shared'] if model['texture_protocol'] == 'native-replay' else ['generation', 'render', 'shared']
    require([p['name'] for p in case['phases']] == names, 'complete ordered native phase sequence required')
    native = {name: json.loads(pinned(phase['native_plan']).read_text()) for name, phase in phases.items()}
    root = Path(native['generation']['output']).parent
    completed_path = root/'case-completed.json'
    root_receipt = root/'file4-closure.json'
    closed.check(completed_path, root_receipt)
    completed = json.loads(completed_path.read_text())
    require(completed['status'] == 'PASS_ACTUAL_FROZEN_NATIVE_CASCADE_CASE'
            and completed['alias'] == alias and completed['sample_id'] == case['sample_id']
            and completed['source_cluster_id'] == case['source_cluster_id']
            and completed['model_freeze'] == case['model_freeze']
            and [t['phase'] for t in completed['terminals']] == names,
            'actual completed native case identity/phase sequence differs')
    inputs = {str(path): str(prepared_receipt), str(completed_path): str(root_receipt)}

    def add(path, receipt):
        path = str(path)
        closed.check(path, receipt)
        inputs[path] = str(receipt)
        return json.loads(Path(path).read_text())

    IO = {}
    for name, phase in phases.items():
        pinned(phase['native_source']); pinned(phase['execution_source'])
        plan_path = phase['native_plan']['path']
        add(plan_path, prepared_receipt)
        folder = Path(native[name]['output'])
        terminal = add(folder/'terminal.json', folder/'file4-closure.json')
        require(terminal == completed['terminals'][names.index(name)]
                and terminal['phase'] == name and terminal['native_plan'] == phase['native_plan']
                and terminal['native_source'] == phase['native_source']
                and terminal['physical_GPU'] in {'0','1','2','3'},
                'completed case must retain the exact actual phase terminal')
        binding = None
        if phase['remote_stage'] is not None:
            binding = str(root/(name+'-IO.json'))
            add(binding, root_receipt)
        read_terminal(terminal, plan_path, binding, read=lambda p: json.loads(Path(p).read_text()), module=phase['module'])
        result = add(folder/phase['result'], folder/'file4-closure.json')
        require(result['status'].startswith('PASS_'), 'unsuccessful actual native phase result')
        IO[name] = binding
    SS_path = native['generation']['SS_evaluation']
    require(SS_path == source_inputs['SS_evaluations'][alias]['path'], 'case consumed a different SS evaluation')
    pinned(source_inputs['SS_evaluations'][alias])
    add(SS_path, SS_receipt)
    generation = phases['PBR'] if 'PBR' in phases else phases['generation']
    row = dict(sample_id=case['sample_id'], source_cluster_id=case['source_cluster_id'],
        generation=str(Path(native[generation['name']]['output'])/'generation.json'),
        native_generation_plan=phases['generation']['native_plan']['path'],
        generation_terminal=str(Path(native['generation']['output'])/'terminal.json'),
        generation_IO_binding=IO['generation'],
        render_camera_audit=str(Path(native['render']['output'])/'render-evaluation.json'),
        native_render_plan=phases['render']['native_plan']['path'],
        render_terminal=str(Path(native['render']['output'])/'terminal.json'), render_IO_binding=IO['render'],
        render=str(Path(native['shared']['output'])/'render-evaluation.json'), closed_inputs=inputs)
    if 'PBR' in phases:
        row.update(shape_generation=str(Path(native['generation']['output'])/'generation.json'),
            PBR_replay_plan=phases['PBR']['native_plan']['path'],
            PBR_replay_terminal=str(Path(native['PBR']['output'])/'terminal.json'), PBR_IO_binding=IO['PBR'])
    observations = [image_input(Path(row['render']).parent/f'camera{cid:02d}.npz', alias=alias,
                    sample_id=case['sample_id'], cluster=case['source_cluster_id'], closed=closed) for cid in (8,9,10)]
    return row, observations


def prepare(plan, output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('preserve existing metric plans/outputs')
    require(plan['scope'] == 'final-test508', 'explicit final-test508 scope required')
    compiled = json.loads(pinned(plan['case_manifest']).read_text())
    source = compiled['source_inputs']
    freeze = json.loads(pinned(source['model_freeze']).read_text())
    ids, clusters, _ = validate_frozen_models(freeze, compiled['models'])
    require(compiled['status'] == 'PREPARED_ALL_FROZEN_TEST508_NATIVE_CASCADE_CASES'
            and compiled['scope'] == 'final-test508' and compiled['assets'] == 508
            and compiled['model_freeze'] == source['model_freeze']
            and source['manifest']['SHA256'] == freeze['population']['manifest_SHA256']
            and compiled['source_clusters'] == 134 and compiled['total_cases'] == 508*len(freeze['models'])
            and tuple(compiled['cases']) == tuple(freeze['models'])
            and tuple(plan['SS_evaluation_receipts']) == tuple(freeze['models']),
            'complete compiled frozen model population required')
    load_QA(source)
    rows = [json.loads(line) for line in pinned(source['manifest']).read_text().splitlines() if line.strip()]
    require([row['sample_id'] for row in rows] == ids
            and all(row['source_cluster_id'] == clusters[row['sample_id']] for row in rows),
            'native metric population must retain the exact frozen manifest order/clusters')
    closed = ClosedInputs(plan['canonical_local_root'])
    closed.check(plan['case_manifest']['path'], plan['prepared_cases_receipt'])
    pins = PinnedFiles()
    registry_path = pins.check(plan['metric_registry'])
    require(registry_path.name == 'registry.json', 'official perceptual registry.json required')
    registry = json.loads(registry_path.read_text())
    require(registry['status'] == 'PASS_OFFICIAL_PERCEPTUAL_WEIGHT_SOURCE_CACHE_AND_CPU_LOAD',
            'prepared official metric packages/weights required')
    for value in registry['files']:
        path = registry_path.parent/value['path']
        require(path.stat().st_size == value['bytes'], 'official metric file size differs')
        pins.check(dict(path=str(path), SHA256=value['sha256']))
    for evidence in freeze['development_evidence']: pins.check(evidence)
    references, models, perceptuals = {}, {}, {}
    for alias, model in freeze['models'].items():
        require([c['sample_id'] for c in compiled['cases'][alias]] == ids,
                'all508 ordered case descriptors required: '+alias)
        rows, observations = [], []
        for descriptor in compiled['cases'][alias]:
            row, images = bind_case(descriptor['plan'], alias=alias, model=model, source_inputs=source,
                prepared_receipt=plan['prepared_cases_receipt'], SS_receipt=plan['SS_evaluation_receipts'][alias], closed=closed)
            require(row['sample_id'] == descriptor['sample_id'] and row['source_cluster_id'] == clusters[row['sample_id']],
                    'actual case sample/source cluster differs')
            for image in images:
                audit = image['image_audit']
                identity = {k:v for k,v in audit.items() if k in {'original_float_GT_SHA256','original_GT_alpha_SHA256','GT_only_crop_box'}}
                identity['converted_reference'] = {k:v for k,v in audit['converted_image_SHA256'].items() if k.startswith('reference_')}
                key = (row['sample_id'], image['camera_id'])
                require(references.setdefault(key, identity) == identity, 'cross-model actual reference pixels/crop differ')
            rows.append(row); observations.extend(images)
        models[alias] = rows; perceptuals[alias] = observations
    pins.finish()
    evaluator = Path(__file__).with_name('evaluate_official_perceptual.py')
    wrapper = Path(__file__).with_name('bound_official_perceptual.py')
    output.mkdir(parents=True, exist_ok=False)
    descriptors = {}
    for i, (alias, observations) in enumerate(perceptuals.items()):
        folder = output/f'model{i:03d}'; folder.mkdir()
        metric = dict(scope='final-test508', full_panel=True, metric_models=str(registry_path.parent),
            output=str(folder/'actual'), inputs=observations, frozen_alias=alias,
            model_freeze=source['model_freeze'], metric_registry=plan['metric_registry'],
            native_case_manifest=plan['case_manifest'], completed_cache_QA=source['cache_QA'],
            cache_QA_plan=source['cache_QA_plan'], manifest=source['manifest'],
            evaluator_source=dict(path=str(evaluator), SHA256=sha(evaluator)),
            wrapper_source=dict(path=str(wrapper), SHA256=sha(wrapper)),
            closed_inputs={row['shared_npz']:str(Path(row['shared_npz']).parent/'file4-closure.json') for row in observations},
            canonical_local_root=plan['canonical_local_root'])
        spec = dump(folder/'perceptual-plan.json', metric)
        descriptors[alias] = dict(plan=spec, output=metric['output'])
    result = dict(status='PREPARED_COMPLETED_NATIVE_TEST508_METRIC_INPUTS', scope='final-test508',
        source_inputs=plan, native_source_inputs=source, model_freeze=source['model_freeze'],
        models=models, perceptuals=descriptors, assets=508, source_clusters=134,
        total_cases=len(models)*508, total_perceptual_views=len(models)*1524,
        no_model_selection_or_neural_metric_execution=True, full_goal_complete=False)
    dump(output/'manifest.json', result)
    return result


def bind_completed(manifest_spec):
    from .bound_official_perceptual import verify_result
    prepared = json.loads(pinned(manifest_spec).read_text())
    require(prepared['status'] == 'PREPARED_COMPLETED_NATIVE_TEST508_METRIC_INPUTS'
            and prepared['scope'] == 'final-test508', 'completed frozen native metric preparation required')
    source = prepared['native_source_inputs']
    freeze = json.loads(pinned(source['model_freeze']).read_text())
    ids, _, _ = validate_frozen_models(freeze, prepared['models'])
    require(tuple(prepared['perceptuals']) == tuple(freeze['models'])
            and all([r['sample_id'] for r in rows] == ids for rows in prepared['models'].values()),
            'complete frozen model case/metric population required')
    closed = ClosedInputs(prepared['source_inputs']['canonical_local_root'])
    perceptuals = {}
    for alias, descriptor in prepared['perceptuals'].items():
        metric = json.loads(pinned(descriptor['plan']).read_text())
        root = Path(descriptor['output']); report = root/'evaluation.json'
        receipt = root/'file4-closure.json'
        closed.check(report, receipt)
        closed.check(root/'binding-audit.json', receipt)
        verify_result(metric, descriptor['plan'], json.loads(report.read_text()), json.loads((root/'binding-audit.json').read_text()))
        require(metric['frozen_alias'] == alias, 'completed perceptual model alias differs')
        perceptuals[str(report)] = str(receipt)
    pinned(manifest_spec)
    return dict(scope='final-test508', models=prepared['models'], perceptuals=perceptuals,
        model_freeze=source['model_freeze'], manifest=source['manifest'], cache_QA=source['cache_QA'],
        cache_QA_plan=source['cache_QA_plan'], file4_cache_root=source['file4_cache_root'],
        canonical_local_root=prepared['source_inputs']['canonical_local_root'], prepared_metric_manifest=manifest_spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    first = sub.add_parser('prepare'); first.add_argument('--plan', type=Path, required=True); first.add_argument('--output', type=Path, required=True)
    last = sub.add_parser('bind'); last.add_argument('--manifest', type=Path, required=True)
    last.add_argument('--manifest-sha256', required=True); last.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(json.loads(args.plan.read_text()), args.output)
        print(json.dumps(dict(status=result['status'], total_cases=result['total_cases'])), flush=True)
    else:
        if args.output.exists(): raise FileExistsError('preserve existing eight-metric plan')
        result = bind_completed(dict(path=str(args.manifest), SHA256=args.manifest_sha256))
        args.output.parent.mkdir(parents=True, exist_ok=True); dump(args.output, result)
        print(json.dumps(dict(status='BOUND_COMPLETED_NATIVE_TEST508_EIGHT_METRIC_INPUTS', models=list(result['models']))), flush=True)


if __name__ == '__main__': main()
