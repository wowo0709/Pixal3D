"""Dataset-only final population binding for the native SS480 checkpoint schema."""
from pathlib import Path

from .evaluate_bound_population import project_dataset, validate_population


SCHEMA = 'native-cold-SS480-frozen-binding-v1'
MODE = 'bound-population'
ENGINEERING = 'engineering-validation36'
FINAL = 'final-test508'
VARIANTS = {'vanilla', 'R_top1_only', 'R_feature', 'R_raw_topk'}


def model_binding(bundle, cfg, hashes):
    variant = bundle['variant']
    root = Path(cfg['output_dir'])
    expected_source = None if variant == 'vanilla' else str(root/'ckpts/deform_conditioner_step0000480.pt')
    if (variant not in VARIANTS or type(bundle['step']) is not int or bundle['step'] != 480
            or cfg['training_seed'] not in (101, 202)
            or bundle['checkpoint'] != str(root/'ckpts/denoiser_step0000480.pt')
            or bundle['conditioner_checkpoint'] != expected_source
            or bundle['native_config'] != str(root/'config.json')):
        raise ValueError('own complete native480 NN/source/config binding required')
    def spec(path):
        if path is None:
            return None
        digest = hashes[path]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError('actual SHA256 required for every model file')
        return dict(path=path, SHA256=digest)
    return dict(schema=SCHEMA, variant=variant, training_seed=cfg['training_seed'], step=480,
                native_config=spec(bundle['native_config']), checkpoint=spec(bundle['checkpoint']),
                conditioner_checkpoint=spec(expected_source), conditioning_mode='edited-rgb',
                source_teacher_updates=0, native_task_objective='SS_flow_mse')


def validate_plan(plan, bundle, cfg, rows, panel, seeds, *, train_rows, validation_rows,
                  binding, freeze=None, cache_QA=None, cache_QA_plan=None, parity=None):
    scope = plan.get('population_scope')
    expected = 36 if scope == ENGINEERING else 508 if scope == FINAL else None
    if (expected is None or plan.get('mode') != MODE or plan.get('expected_assets') != expected
            or len(panel) != expected or len(set(panel)) != expected or set(seeds) != set(panel)
            or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds.values())
            or plan.get('generation_batch_size') != 8 or plan.get('deterministic_algorithms') is not True
            or plan.get('model_binding') != binding or bundle.get('step') != 480):
        raise ValueError('explicit native population, matched frozen model, seeds and original batch8 required')
    inputs = plan['population_inputs']
    if plan['panel'] != inputs['panel']['path'] or plan['seeds'] != inputs['seeds']['path']:
        raise ValueError('native panel/noise paths differ from pinned population inputs')
    roots = plan['dataset_roots']
    if set(roots) != {'match_root', 'projection_root'}:
        raise ValueError('exact explicit dataset roots required')
    if scope == ENGINEERING:
        if inputs['manifest']['path'] != cfg['validation']['roots_file']:
            raise ValueError('engineering must use the original validation manifest')
        if len(rows) != 508:
            raise ValueError('complete original validation508 required')
        validate_population(rows, scope='engineering-validation32', panel=panel[:32],
                            seeds={s: seeds[s] for s in panel[:32]},
                            train_rows=train_rows, validation_rows=validation_rows)
        if not set(panel).issubset(r['sample_id'] for r in rows):
            raise ValueError('engineering tail must also be validation data')
        rebound = project_dataset(cfg, manifest=inputs['manifest']['path'], expected_split='validation', **roots)
        if rebound != cfg:
            raise ValueError('engineering binding must preserve the entire original native recipe')
        return rebound
    validate_population(rows, scope=FINAL, panel=panel, seeds=seeds,
                        train_rows=train_rows, validation_rows=validation_rows)
    if (cache_QA is None or cache_QA_plan is None
            or cache_QA.get('status') != 'PASS_ACTUAL_TEST508_NATIVE_CACHE_AND_LOADER_QA'
            or cache_QA.get('published_manifest') != inputs['manifest']['path']
            or cache_QA.get('published_manifest_SHA256') != inputs['manifest']['SHA256']
            or not cache_QA.get('SS_PBR_cache_ready') or not cache_QA.get('all508_paired_coverage')
            or not cache_QA.get('no_test_asset_dropped')
            or not cache_QA.get('no_candidate_model_forward_or_test_quality_metric')
            or cache_QA.get('matching_head_training_or_calibration_updates') != 0
            or cache_QA.get('Omega_completed_updates') != 23597
            or roots != {k: cache_QA_plan[k] for k in roots}):
        raise ValueError('actual complete508 native cache QA and pinned dataset roots required')
    q2c = cfg['validation']['dataset_args'].get('expected_q2c_sha256')
    if q2c is not None and cache_QA.get('frozen_TRAIN_q2c_SHA256') != q2c:
        raise ValueError('test cache matcher differs from the actual trained model')
    clusters = {r['sample_id']: r['source_cluster_id'] for r in rows}
    if (freeze is None or freeze.get('status') != 'FROZEN_FROM_DEVELOPMENT_ONLY'
            or freeze.get('heldout_quality_read_before_freeze') is not False
            or not freeze.get('development_evidence')
            or freeze.get('models', {}).get(plan['model_alias'], {}).get('ss64') != binding
            or freeze.get('population', {}).get('sample_ids') != panel
            or freeze['population'].get('source_clusters_by_sample') != clusters
            or len(set(clusters.values())) != 134
            or freeze['population'].get('rendered_cameras') != [8, 9, 10]):
        raise ValueError('development-only frozen model and complete508/134-source population required')
    if (parity is None or parity.get('status') != 'PASS_ACTUAL_NATIVE_BOUND_VALIDATION36_REPEATABILITY'
            or parity.get('model_binding') != binding
            or parity.get('independent_model_loads') != 2
            or not parity.get('reference_prefix32_records_exact')
            or not parity.get('all36_records_and_archived_occupancy_exact')
            or parity.get('tail_generation_batch') != 4
            or parity.get('optimizer_updates') != 0):
        raise ValueError('actual own saved-model validation36/tail4 parity must precede final test')
    # project_dataset reverses exactly all changes to roots/split and never
    # turns an online matcher (None) into a cached correspondence loader.
    return project_dataset(cfg, manifest=inputs['manifest']['path'], expected_split='test', **roots)


def validate_capture(result, captured, count):
    if (result.get('status') != 'PASS' or result.get('optimizer_updates_during_evaluation') != 0
            or len(result['per_asset']) != count or len(captured) != count
            or result['quick_panel_ids'] != [r['sample_id'] for r in result['per_asset']]):
        raise ValueError('one complete actual native prediction/GT archive per population sample required')
    for row, artifact in zip(result['per_asset'], captured):
        if (artifact['sample_id'] != row['sample_id'] or artifact['seed'] != row['seed']
                or artifact['target_voxels_SHA256'] != row['target_voxels_sha256']
                or any(artifact['occupancy_counts'][k] != row[k] for k in ('tp', 'fp', 'fn'))):
            raise ValueError('actual archived sample/noise/GT/counts differ from measured predictions')


def validate_repeatability(results, audits):
    if len(results) != 2 or len(audits) != 2:
        raise ValueError('two independent actual native model loads required')
    for result, audit in zip(results, audits):
        if (audit.get('status') != 'PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION'
                or audit.get('population_scope') != ENGINEERING or audit.get('assets') != 36
                or audit.get('actual_generation_batch') != 8 or audit.get('tail_generation_batch') != 4
                or audit.get('sampling_steps') != 12 or audit.get('guidance') != 3.
                or audit.get('optimizer_updates') != 0
                or not audit.get('actual_reference_prefix32_records_exact')
                or not audit.get('actual_saved_SS_occupancy_counts_verified')
                or result.get('frozen_native_model_binding') != audit['model_binding']):
            raise ValueError('actual validation36/reference32/tail4 native execution required')
        artifacts = result['native_SS_artifacts']
        if artifacts.get('archived_asset_count') != 36:
            raise ValueError('complete36 actual SS archives required')
        validate_capture(result, artifacts['artifacts'], 36)
    a, b = results
    for key in ('per_asset', 'quick_metrics', 'quick_panel_ids', 'manifest_sha256',
                'actual_loaded_trainable_SHA256', 'frozen_native_model_binding'):
        if a[key] != b[key]:
            raise ValueError('independent native36 execution differs: '+key)
    for x, y in zip(a['native_SS_artifacts']['artifacts'], b['native_SS_artifacts']['artifacts']):
        for key in ('sample_id', 'seed', 'target_voxels_SHA256', 'prediction_voxels_SHA256',
                    'occupancy_counts', 'prediction_latent_SHA256', 'target_latent_SHA256'):
            if x[key] != y[key]:
                raise ValueError('actual decoded occupancy or native latent differs: '+key)
    return dict(status='PASS_ACTUAL_NATIVE_BOUND_VALIDATION36_REPEATABILITY',
                model_binding=audits[0]['model_binding'], independent_model_loads=2,
                reference_prefix32_records_exact=True, all36_records_and_archived_occupancy_exact=True,
                actual_native_prediction_and_target_latent_arrays_exact=True,
                generation_batch=8, tail_generation_batch=4, optimizer_updates=0,
                model_test_quality_evaluated=False, full_goal_complete=False)
