import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.deformnext_retargeting.evaluate_bound_population import sha
from experiments.deformnext_retargeting.metric_image_identity import scored_image_observation
from experiments.deformnext_retargeting.prepare_frozen_inception_pools import prepare, bind_completed_pools


def dump(path, value):
    path.write_text(json.dumps(value))
    return dict(path=str(path), SHA256=sha(path))


@pytest.fixture
def inputs(tmp_path):
    archive = tmp_path/'synthetic-only-scored-render.npz'
    arrays = dict(prediction=np.full((3,512,512), .2, dtype=np.float32),
        GT_shared_renderer=np.full((3,512,512), .8, dtype=np.float32),
        GT_alpha=np.ones((512,512), dtype=np.float32))
    np.savez_compressed(archive, **arrays)
    ids = [f'synthetic-asset{i}' for i in range(508)]
    clusters = {sid:f'synthetic-source{i%134}' for i,sid in enumerate(ids)}
    template = scored_image_observation(arrays, sample_id=ids[0], camera_id=8, source_cluster_id=clusters[ids[0]],
                                         archive=archive, archive_SHA256=sha(archive))
    models = {name:dict(synthetic_checkpoint=name) for name in ['vanilla','matched-R','candidate']}
    freeze = dict(status='FROZEN_FROM_DEVELOPMENT_ONLY', heldout_quality_read_before_freeze=False,
        development_evidence=[dump(tmp_path/'synthetic-development.json', dict(synthetic_only=True))],
        models=models, reference_binding=dict(synthetic_GT='original-material-M1'), selected_candidate='candidate',
        comparison_roles=dict(vanilla='vanilla', retargeting='matched-R', candidate='candidate'),
        contrasts=[['candidate','vanilla'],['candidate','matched-R']],
        population=dict(sample_ids=ids, source_clusters_by_sample=clusters, rendered_cameras=[8,9,10],
                        manifest_SHA256='synthetic-manifest'))
    freeze_spec = dump(tmp_path/'synthetic-freeze.json', freeze)
    rows = [dict(template, sample_id=sid, camera_id=cid, source_cluster_id=clusters[sid]) for sid in ids for cid in (8,9,10)]
    eight = dict(status='PASS_ACTUAL_BOUND_CASCADE_EIGHT_METRICS', scope='final-test508', model_bindings=models,
        model_freeze=freeze_spec, manifest=dict(SHA256='synthetic-manifest'), assets=508, source_clusters=134,
        actual_completed_cases=3*508, actual_perceptual_views=3*1524, native_stage_weight_hashes_verified=True,
        scored_image_observations={name:copy.deepcopy(rows) for name in models},
        input_SHA256={str(archive):sha(archive)})
    weight = tmp_path/'synthetic-only-weights'; weight.write_bytes(b'synthetic test bytes, no network or neural execution')
    registry = dump(tmp_path/'synthetic-registry.json', dict(package='torch-fidelity0.3.0', weights_SHA256=sha(weight)))
    plan = dict(scope='final-test508', model_freeze=freeze_spec, eight_metrics=dump(tmp_path/'synthetic-eight.json',eight),
        metric_registry=registry, weights=dict(path=str(weight),SHA256=sha(weight)), package_root=str(tmp_path), batch_size=32)
    return plan, eight, freeze, template


def update_eight(inputs):
    plan, eight, _, _ = inputs
    plan['eight_metrics'] = dump(Path(plan['eight_metrics']['path']), eight)


def test_prepare_validates_actual_render_bytes_and_writes_complete_explicit_feature_plans(inputs, tmp_path):
    plan, eight, freeze, template = inputs
    before = copy.deepcopy(inputs)
    manifest = prepare(plan, tmp_path/'plans')
    assert inputs == before and manifest['models'] == list(freeze['models'])
    assert manifest['no_neural_feature_extraction_or_quality_score_performed']
    assert len(manifest['pools']) == 4
    for pool in manifest['pools'].values():
        extract = json.loads(Path(pool['extract_plan']['path']).read_text())
        assert len(extract['observations']) == 1524 and extract['expected_sample_ids'] == freeze['population']['sample_ids']
        assert extract['model_freeze'] == plan['model_freeze'] and not Path(pool['output']).exists()
        array = 'GT_shared_renderer' if pool['alias'] is None else 'prediction'
        assert all(row['array'] == array for row in extract['observations'])
    with pytest.raises(FileExistsError): prepare(plan, tmp_path/'plans')


@pytest.mark.parametrize('problem', ['development', 'incomplete', 'wrong-freeze', 'archive-binding', 'actual-pixels',
    'raw-float', 'conversion', 'alpha', 'duplicate', 'source', 'bad-registry', 'bad-batch'])
def test_prepare_rejects_incomplete_or_changed_images_before_writing_any_plans(inputs, tmp_path, problem):
    plan, eight, freeze, template = inputs
    row = eight['scored_image_observations']['candidate'][0]
    if problem == 'development': eight['scope'] = 'engineering-validation32'
    elif problem == 'incomplete': eight['actual_completed_cases'] -= 1
    elif problem == 'wrong-freeze': eight['model_freeze'] = dict(path='different',SHA256='different')
    elif problem == 'archive-binding': eight['input_SHA256'].clear()
    elif problem == 'actual-pixels': Path(row['archive']).write_bytes(b'changed after completed eight-metric evaluation')
    elif problem == 'raw-float': row['prediction_float_RGB_SHA256'] = 'other-pixels'
    elif problem == 'conversion': row['prediction_CHW_uint8_SHA256'] = 'other-conversion'
    elif problem == 'alpha': row['reference_alpha_SHA256'] = 'other-alpha'
    elif problem == 'duplicate': eight['scored_image_observations']['candidate'][-1] = copy.deepcopy(row)
    elif problem == 'source': row['source_cluster_id'] = 'wrong-source'
    elif problem == 'bad-registry': plan['metric_registry'] = dump(tmp_path/'wrong-registry.json',dict(package='other',weights_SHA256=plan['weights']['SHA256']))
    else: plan['batch_size'] = True
    update_eight(inputs)
    with pytest.raises(ValueError): prepare(plan, tmp_path/'plans')
    assert not (tmp_path/'plans').exists()


@pytest.fixture
def completed_synthetic_pools(inputs, tmp_path):
    plan, eight, freeze, template = inputs
    output = tmp_path/'plans'; manifest = prepare(plan, output)
    ids = freeze['population']['sample_ids']; clusters = freeze['population']['source_clusters_by_sample']
    shared_features = tmp_path/'synthetic-only-features.npz'
    np.savez_compressed(shared_features, features=np.ones((1524,2048),dtype=np.float32),
        sample_id=np.asarray([sid for sid in ids for _ in (8,9,10)]),
        camera_id=np.asarray([cid for _ in ids for cid in (8,9,10)]),
        source_cluster_id=np.asarray([clusters[sid] for sid in ids for _ in (8,9,10)]))
    for descriptor in manifest['pools'].values():
        extract = json.loads(Path(descriptor['extract_plan']['path']).read_text())
        root = Path(descriptor['output']); root.mkdir()
        (root/'features.npz').write_bytes(shared_features.read_bytes())
        ref = descriptor['alias'] is None
        observations = [dict(row,converted_CHW_uint8_SHA256=template['reference_CHW_uint8_SHA256' if ref else 'prediction_CHW_uint8_SHA256'])
                        for row in extract['observations']]
        audit = dict(status='PASS_ACTUAL_OFFICIAL_INCEPTION2048_FEATURE_POOL', scope='final-test508',
            pool_role=extract['pool_role'], images=1524, assets=508, features_SHA256=sha(root/'features.npz'),
            TF32=False,autocast=False,fullframe=True,observations=observations,model_binding=extract['model_binding'],
            package='torch-fidelity0.3.0',feature_layer='2048',feature_dtype='float32',same_batch_repeat_exact=True,
            GT_crop_or_CLIP_preprocessing_used=False,optimizer_updates=0,weights_SHA256=plan['weights']['SHA256'],
            input_plan_SHA256=descriptor['extract_plan']['SHA256'])
        dump(root/'audit.json',audit)
    return inputs, manifest, dict(path=str(output/'manifest.json'),SHA256=sha(output/'manifest.json'))


def test_bind_requires_completed_actual_feature_schema_and_retains_exact_images(completed_synthetic_pools):
    inputs, manifest, spec = completed_synthetic_pools
    result = bind_completed_pools(spec)
    assert list(result['predictions']) == ['vanilla','matched-R','candidate']
    assert result['model_freeze'] == inputs[0]['model_freeze']
    assert result['actual_scored_input_image_identity']['prediction_images'] == 3*1524
    assert result['no_distribution_score_computed_during_binding']


@pytest.mark.parametrize('problem', ['unfinished', 'wrong-executed-plan', 'wrong-pixels', 'wrong-model', 'wrong-weights', 'stale-input'])
def test_bind_rejects_unfinished_or_misbound_GPU_feature_outputs(completed_synthetic_pools, problem):
    inputs, manifest, spec = completed_synthetic_pools
    descriptor = next(row for row in manifest['pools'].values() if row['alias']=='candidate')
    path = Path(descriptor['output'])/'audit.json'; audit = json.loads(path.read_text())
    if problem == 'unfinished': path.unlink()
    elif problem == 'wrong-executed-plan': audit['input_plan_SHA256'] = 'another-plan'; dump(path,audit)
    elif problem == 'wrong-pixels': audit['observations'][0]['original_float_RGB_SHA256'] = 'another-image'; dump(path,audit)
    elif problem == 'wrong-model': audit['model_binding'] = inputs[2]['models']['matched-R']; dump(path,audit)
    elif problem == 'wrong-weights': audit['weights_SHA256'] = 'other-weights'; dump(path,audit)
    else: Path(inputs[0]['eight_metrics']['path']).write_text('{}')
    with pytest.raises((ValueError,FileNotFoundError)): bind_completed_pools(spec)
