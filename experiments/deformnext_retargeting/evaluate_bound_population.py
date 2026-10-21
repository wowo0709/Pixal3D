"""Explicit held-out SS data binding around unchanged native evaluation.

Training recipes keep their validation-only guards. The native evaluator sees
only four explicit dataset-field substitutions in memory. Its model, sampler,
noise construction and decoder code reverse exactly to the frozen source.
Final test execution additionally requires a frozen model set and completed
all508 cache QA; engineering replays accept validation data only.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys


ENTRY = 'experiments.deformnext_retargeting.evaluate_native_checkpoint'
MODULES = {
    'evaluate_deterministic_checkpoint', 'evaluate_consistent_ray_retargeting',
    'evaluate_direct_ray_retargeting', 'evaluate_native_prior_source_ray',
    'member_posterior_runtime', 'member_feature_runtime',
    'member_feature_deformation_runtime', 'member_feature_control_runtime',
}
SS_DATASETS = {'DeformNeXtSS64', 'DeformNeXtMemberRawSS64'}
IDENTITIES = ('sample_id', 'source_uid', 'source_cluster_id')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pinned(spec):
    path = Path(spec['path'])
    if sha(path) != spec['SHA256']:
        raise ValueError('explicit file byte identity differs: ' + str(path))
    return path


def argument(arguments, flag, *, optional=False):
    count = arguments.count(flag)
    if optional and not count:
        return None
    if count != 1 or arguments.index(flag) + 1 >= len(arguments):
        raise ValueError('one explicit argument required: ' + flag)
    return arguments[arguments.index(flag) + 1]


def validate_population(rows, *, scope, panel, seeds, train_rows, validation_rows):
    if scope not in {'engineering-validation32', 'final-test508'}:
        raise ValueError('explicit engineering validation or final-test scope required')
    expected_split = 'test' if scope == 'final-test508' else 'validation'
    if (not rows or any(row.get('split') != expected_split for row in rows)
            or any(not isinstance(row.get(field), str) or not row[field]
                   for row in rows for field in IDENTITIES)):
        raise ValueError('complete explicit population identity and split required')
    ids = [row['sample_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate population sample IDs')
    if (not isinstance(panel, list) or len(panel) != len(set(panel))
            or not set(panel).issubset(ids) or set(seeds) != set(panel)
            or any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**31
                   for value in seeds.values())):
        raise ValueError('explicit matched unique panel and integer noise map required')
    if scope == 'engineering-validation32':
        if len(panel) != 32:
            raise ValueError('engineering replay uses exactly32 validation assets')
        return expected_split
    if len(rows) != 508 or len(panel) != 508 or panel != ids:
        raise ValueError('all508 test samples in the prescribed manifest order required')
    if len(train_rows) != 9149 or len(validation_rows) != 508:
        raise ValueError('complete original9149 training and508 validation manifests required')
    for values, split in [(train_rows, 'train'), (validation_rows, 'validation')]:
        if (not values or any(row.get('split') != split for row in values)
                or len({row['sample_id'] for row in values}) != len(values)
                or any(not isinstance(row.get(field), str) or not row[field]
                       for row in values for field in IDENTITIES)):
            raise ValueError('original training and validation populations required')
        for field in IDENTITIES:
            if {row[field] for row in rows}.intersection(row[field] for row in values):
                raise ValueError('held-out test identity overlaps ' + split + ': ' + field)
    if any({row[field] for row in train_rows}.intersection(row[field] for row in validation_rows)
           for field in IDENTITIES):
        raise ValueError('original training and validation identities overlap')
    return expected_split


def project_dataset(cfg, *, manifest, expected_split, match_root, projection_root):
    if (cfg['validation']['dataset_name'] not in SS_DATASETS
            or cfg['validation']['dataset_args']['expected_split'] != 'validation'
            or cfg['validation']['quick_metrics']['sampling_steps'] != 12
            or expected_split not in {'validation', 'test'}):
        raise ValueError('unchanged native SS12-step validation recipe required')
    result = copy.deepcopy(cfg)
    result['validation']['roots_file'] = str(manifest)
    result['validation']['dataset_args']['expected_split'] = expected_split
    # None is part of the original model's input contract, not a missing path:
    # source-ray models match online, and physical RAW-DQ forbids old sidecars.
    for key, available in [('match_root', match_root), ('projection_root', projection_root)]:
        if cfg['validation']['dataset_args'].get(key) is not None:
            if not isinstance(available, str) or not available:
                raise ValueError('an explicitly rebound active dataset root is required: ' + key)
            result['validation']['dataset_args'][key] = available
    reverse = copy.deepcopy(result)
    reverse['validation']['roots_file'] = cfg['validation']['roots_file']
    for key in ['expected_split', 'match_root', 'projection_root']:
        if key in cfg['validation']['dataset_args']:
            reverse['validation']['dataset_args'][key] = cfg['validation']['dataset_args'][key]
        else:
                reverse['validation']['dataset_args'].pop(key, None)
    if reverse != cfg:
        raise ValueError('evaluation binding changed fields outside explicit dataset selection')
    return result


def native_data_tree(source):
    original = ast.parse(source)
    count = 0

    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal count
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'cfg' and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == 'json' and node.value.func.attr == 'loads'):
                count += 1
                added = ast.parse('cfg = _bound_dataset(cfg)').body[0]
                added._bound_data_only = True
                return [node, added]
            return self.generic_visit(node)

    tree = ast.fix_missing_locations(Rewrite().visit(copy.deepcopy(original)))
    if count != 1:
        raise ValueError('exactly one native recipe-read site required')

    class Reverse(ast.NodeTransformer):
        def visit(self, node):
            if getattr(node, '_bound_data_only', False):
                return None
            return super().visit(node)

    if ast.dump(Reverse().visit(copy.deepcopy(tree))) != ast.dump(original):
        raise ValueError('native evaluator changed outside dataset binding')
    return tree


def validate_model_arguments(plan, model):
    args = plan['native_arguments']
    if plan['evaluation_module'] not in MODULES or plan['evaluation_module'] != model['evaluation_module']:
        raise ValueError('explicit supported and frozen evaluator required')
    if (argument(args, '--stage') != 'ss64' or int(argument(args, '--step')) != model['step']
            or argument(args, '--output') != plan['output']
            or argument(args, '--panel-file') != plan['panel']['path']
            or argument(args, '--seed-map-file') != plan['seeds']['path']
            or int(argument(args, '--panel-size')) != plan['assets']
            or argument(args, '--generation-batch') != '8'
            or any(flag in args for flag in ['--transport-control', '--r9-path-control', '--task-value-control'])):
        raise ValueError('frozen native SS8/12step/CFG3 read-only argument contract differs')
    mode = argument(args, '--conditioning-mode', optional=True) or 'edited-rgb'
    if (mode not in {'edited-rgb', 'source-only'} or mode != model['conditioning_mode']
            or (mode == 'source-only' and model['conditioner_checkpoint'] is not None)):
        raise ValueError('frozen edited/source-only RGB input mode differs')
    native = pinned(plan['native_entry'])
    if (Path(argument(args, '--expected-stage-root')).resolve() / 'experiments/deformnext_retargeting/evaluate_native_checkpoint.py'
            != native.resolve() or argument(args, '--expected-entry-sha') != plan['native_entry']['SHA256']):
        raise ValueError('explicit pinned native evaluator source required')
    pinned(model['evaluator_source'])
    if not model['source_SHA256'] or any(sha(path) != digest for path, digest in model['source_SHA256'].items()):
        raise ValueError('frozen model runtime source byte identity differs')
    for flag, field in [('--checkpoint', 'checkpoint'), ('--arm-config', 'arm_config'),
                        ('--conditioner-checkpoint', 'conditioner_checkpoint')]:
        selected = argument(args, flag, optional=field == 'conditioner_checkpoint')
        spec = model[field]
        if spec is None:
            if selected is not None:
                raise ValueError('unexpected conditioner checkpoint')
        elif selected is None or Path(selected).resolve() != pinned(spec).resolve():
            raise ValueError('actual checkpoint/config differs from frozen model binding')
    expected = argument(args, '--expected-assets', optional=True)
    phase = argument(args, '--phase', optional=True)
    runtime_cfg = argument(args, '--config', optional=True)
    if ((expected is not None and int(expected) != plan['assets'])
            or (phase is not None and phase != 'evaluation')
            or (runtime_cfg is not None and Path(runtime_cfg).resolve() != Path(model['arm_config']['path']).resolve())
            or (argument(args, '--source-value-control', optional=True) not in (None, 'normal'))):
        raise ValueError('no training phase, value intervention or alternate runtime recipe allowed')


def load_binding(plan):
    if plan['scope'] == 'final-test508' and not {'model_freeze', 'cache_QA', 'cache_QA_plan', 'model_alias'}.issubset(plan):
        raise ValueError('final test requires a frozen model set and complete native cache QA')
    manifest = pinned(plan['manifest'])
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    panel = json.loads(pinned(plan['panel']).read_text())['sample_ids']
    seeds = json.loads(pinned(plan['seeds']).read_text())
    train = [json.loads(line) for line in pinned(plan['train_manifest']).read_text().splitlines() if line.strip()]
    validation = [json.loads(line) for line in pinned(plan['validation_manifest']).read_text().splitlines() if line.strip()]
    split = validate_population(rows, scope=plan['scope'], panel=panel, seeds=seeds,
                                train_rows=train, validation_rows=validation)
    if len(panel) != plan['assets']:
        raise ValueError('complete selected population required')
    if split == 'test':
        freeze = json.loads(pinned(plan['model_freeze']).read_text())
        qa = json.loads(pinned(plan['cache_QA']).read_text())
        qa_plan = json.loads(pinned(plan['cache_QA_plan']).read_text())
        if (freeze['status'] != 'FROZEN_FROM_DEVELOPMENT_ONLY'
                or freeze['heldout_quality_read_before_freeze'] is not False
                or freeze['population']['manifest_SHA256'] != plan['manifest']['SHA256']
                or freeze['population']['seed_map_SHA256'] != plan['seeds']['SHA256']
                or freeze['population']['sample_ids'] != panel
                or not freeze['development_evidence']):
            raise ValueError('prespecified immutable development-only model freeze required')
        for evidence in freeze['development_evidence']:
            pinned(evidence)
        if (qa['status'] != 'PASS_ACTUAL_TEST508_NATIVE_CACHE_AND_LOADER_QA'
                or qa['assets'] != 508 or not qa['SS_PBR_cache_ready'] or not qa['no_test_asset_dropped']
                or not qa['train_validation_identity_disjoint'] or not qa['all_input_RGB_fingerprints_verified']
                or not qa['no_candidate_model_forward_or_test_quality_metric']
                or qa['matching_head_training_or_calibration_updates'] != 0 or qa['Omega_completed_updates'] != 23597
                or qa['published_manifest_SHA256'] != plan['manifest']['SHA256']
                or Path(qa['published_manifest']).resolve() != manifest.resolve()
                or Path(qa_plan['published_manifest']).resolve() != manifest.resolve()
                or plan['dataset_roots'] != {key: qa_plan[key] for key in ['match_root', 'projection_root']}):
            raise ValueError('actual all508 native cache QA and exact test data roots required')
        model = freeze['models'][plan['model_alias']]['ss64']
    else:
        model = plan['engineering_model_binding']
    validate_model_arguments(plan, model)
    if split == 'test':
        cfg = json.loads(Path(model['arm_config']['path']).read_text())
        dataset = cfg['validation']['dataset_args']
        if (dataset['path_map'] != {'/file4': qa_plan['file4_cache']}
                or (dataset.get('expected_q2c_sha256') is not None
                    and dataset['expected_q2c_sha256'] != qa['frozen_TRAIN_q2c_SHA256'])):
            raise ValueError('native dataset path/matching-head binding differs from actual test cache QA')
    return rows, panel, seeds, split, model


@contextmanager
def bound_native_entry(source_path, projector):
    source = source_path.read_bytes()
    tree = native_data_tree(source.decode())
    original = runpy.run_module
    calls = []

    def execute(name, init_globals=None, run_name=None, alter_sys=False):
        if name != ENTRY:
            return original(name, init_globals=init_globals, run_name=run_name, alter_sys=alter_sys)
        if init_globals is not None or run_name != '__main__' or alter_sys:
            raise ValueError('preserve the exact native evaluator invocation')
        calls.append(name)
        namespace = dict(__name__='__main__', __package__=__package__, __file__=str(source_path),
                         _bound_dataset=projector)
        exec(compile(tree, str(source_path), 'exec'), namespace)
        return namespace

    runpy.run_module = execute
    try:
        yield calls
    finally:
        runpy.run_module = original
    if calls != [ENTRY] or source_path.read_bytes() != source:
        raise ValueError('exactly one unchanged native evaluator execution required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    rows, panel, seeds, split, model = load_binding(plan)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (visible not in {'0','1','2','3'} or visible not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(',')
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise ValueError('one assigned physical GPU0-3 with pinned CUBLAS workspace required')
    output = Path(plan['output'])
    if output.exists():
        raise FileExistsError('fresh population evaluation namespace required')
    entry = importlib.import_module('.'+plan['evaluation_module'], __package__)
    if Path(entry.__file__).resolve() != Path(model['evaluator_source']['path']).resolve():
        raise ValueError('evaluation module resolved outside its frozen source binding')
    from . import quick_metrics as quick
    from .evaluate_ss_artifact_subset import capture_ss_artifacts
    from .member_raw_metric_dispatch import raw_ss_metric_dispatch
    native = Path(__file__).with_name('evaluate_native_checkpoint.py')
    bindings = {str(path): sha(path) for path in [Path(__file__), Path(entry.__file__), native,
                                                 Path(__file__).with_name('member_raw_metric_dispatch.py'),
                                                 Path(model['arm_config']['path']), Path(args.plan)]}
    captured = []
    metric_dispatch_events = []
    previous = sys.argv
    effective_roots = []
    def projector(cfg):
        projected = project_dataset(cfg, manifest=plan['manifest']['path'], expected_split=split,
                                      **plan['dataset_roots'])
        effective_roots.append({key: projected['validation']['dataset_args'].get(key)
                                for key in ['match_root', 'projection_root']})
        return projected
    try:
        with raw_ss_metric_dispatch(metric_dispatch_events), bound_native_entry(native, projector), capture_ss_artifacts(quick, output, captured):
            sys.argv = [entry.__file__, *plan['native_arguments']]
            entry.main()
    finally:
        sys.argv = previous
    result = json.loads((output/'evaluation.json').read_text())
    if (len(effective_roots) != 1 or [row['sample_id'] for row in result['per_asset']] != panel
            or result['quick_panel_ids'] != panel or len(captured) != len(panel)
            or result['optimizer_updates_during_evaluation'] != 0
            or result['generation_batch_size'] != 8 or result['sampling_steps'] != 12 or result['guidance'] != 3.
            or result['manifest_sha256'] != plan['manifest']['SHA256']):
        raise ValueError('complete actual native population/sampler/data identity differs')
    for artifact, record in zip(captured, result['per_asset']):
        if (artifact['sample_id'] != record['sample_id'] or artifact['seed'] != seeds[record['sample_id']]
                or artifact['target_voxels_SHA256'] != record['target_voxels_sha256']
                or any(artifact['occupancy_counts'][key] != record[key] for key in ['tp','fp','fn'])):
            raise ValueError('actual SS archive and native count/noise/GT records differ')
    validate_model_arguments(plan, model)
    if any(sha(path) != digest for path, digest in bindings.items()):
        raise ValueError('native population evaluation source/config/plan changed')
    result['native_SS_artifacts'] = dict(status='PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_BOUND_POPULATION',
        artifacts=captured, archived_asset_count=len(captured), native_occupancy_counts_and_GT_hashes_exact=True,
        predicted_SS_is_available_for_full_cascade=True, GT_SS_is_for_scoring_only=True, optimizer_updates=0,
        no_prior_test_model_reference_or_second_inference_required=True)
    result['bound_population'] = dict(scope=plan['scope'], split=split, assets=len(panel),
        source_clusters=len({row['source_cluster_id'] for row in rows if row['sample_id'] in set(panel)}),
        manifest=plan['manifest'], panel=plan['panel'], seeds=plan['seeds'], dataset_roots=plan['dataset_roots'],
        effective_dataset_roots=effective_roots[0], original_None_online_matching_mode_preserved=True,
        native_RAW_SS_metric_dispatch_events=metric_dispatch_events,
        original_training_recipe_preserved=True, only_four_runtime_dataset_fields_may_be_overridden=True,
        native_model_sampler_and_decoder_AST_reverse_exact=True, model_binding=model,
        model_freeze=plan.get('model_freeze'), source_SHA256=bindings,
        final_fullcascade_and_other_quality_metrics_still_required=True, full_goal_complete=False)
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['native_SS_artifacts']['status'],scope=plan['scope'],assets=len(panel),
                          metrics=result['quick_metrics'],full_goal_complete=False)),flush=True)


if __name__ == '__main__':
    main()
