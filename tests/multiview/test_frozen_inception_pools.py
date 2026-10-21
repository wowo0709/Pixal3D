import copy
import json

import numpy as np
import pytest

from experiments.deformnext_retargeting.evaluate_bound_population import sha
from experiments.deformnext_retargeting.summarize_frozen_inception_pools import (
    load_frozen_pool, validate_frozen_models,
)


def freeze_fixture():
    ids = [f'asset{i}' for i in range(508)]
    models = {name: {'ss64': {'checkpoint_SHA256': name+'-fixture-only'}}
              for name in ['vanilla', 'matched-R', 'feature-DQ', 'zero-source4', 'zero-edited-plus3']}
    return dict(status='FROZEN_FROM_DEVELOPMENT_ONLY', heldout_quality_read_before_freeze=False,
        development_evidence=[{'path': 'synthetic-only'}], models=models, reference_binding={'GT': 'shared-fixture'},
        selected_candidate='feature-DQ', comparison_roles=dict(vanilla='vanilla', retargeting='matched-R', candidate='feature-DQ'),
        contrasts=[['feature-DQ', 'vanilla'], ['feature-DQ', 'matched-R'], ['vanilla', 'zero-source4']],
        population=dict(sample_ids=ids, source_clusters_by_sample={sid: 'source'+str(i % 134) for i, sid in enumerate(ids)},
                        rendered_cameras=[8, 9, 10]))


def test_new_candidate_alias_and_both_zero_controls_remain_frozen():
    freeze = freeze_fixture()
    ids, clusters, pairs = validate_frozen_models(freeze, freeze['models'])
    assert len(ids) == 508 and len(set(clusters.values())) == 134
    assert pairs == [('feature-DQ', 'vanilla'), ('feature-DQ', 'matched-R'), ('vanilla', 'zero-source4')]


@pytest.mark.parametrize('problem', ['unfrozen', 'test-used', 'no-development', 'drop-model', 'wrong-order',
                                   'duplicate-asset', 'missing-source', 'wrong-source-count', 'camera',
                                   'candidate', 'control-alias', 'missing-control-contrast', 'duplicate-contrast'])
def test_freeze_rejects_test_selection_or_missing_and_mislabelled_controls(problem):
    f = freeze_fixture(); names = list(f['models'])
    if problem == 'unfrozen': f['status'] = 'CANDIDATE'
    elif problem == 'test-used': f['heldout_quality_read_before_freeze'] = True
    elif problem == 'no-development': f['development_evidence'] = []
    elif problem == 'drop-model': names.pop()
    elif problem == 'wrong-order': names.reverse()
    elif problem == 'duplicate-asset': f['population']['sample_ids'][-1] = f['population']['sample_ids'][0]
    elif problem == 'missing-source': f['population']['source_clusters_by_sample'].pop('asset0')
    elif problem == 'wrong-source-count': f['population']['source_clusters_by_sample'] = {sid: sid for sid in f['population']['sample_ids']}
    elif problem == 'camera': f['population']['rendered_cameras'] = [8, 9]
    elif problem == 'candidate': f['selected_candidate'] = 'wrong-candidate'
    elif problem == 'control-alias': f['comparison_roles']['retargeting'] = 'feature-DQ'
    elif problem == 'missing-control-contrast': f['contrasts'].pop(1)
    else: f['contrasts'].append(copy.deepcopy(f['contrasts'][0]))
    with pytest.raises(ValueError): validate_frozen_models(f, names)


@pytest.fixture
def actual_pool_files(tmp_path):
    freeze = freeze_fixture(); ids = freeze['population']['sample_ids']; clusters = freeze['population']['source_clusters_by_sample']
    rows = [dict(sample_id=sid, camera_id=cid, source_cluster_id=clusters[sid], array='prediction')
            for sid in ids for cid in (8, 9, 10)]
    features_path = tmp_path/'features.npz'; audit_path = tmp_path/'audit.json'
    values = np.ones((1524, 2048), dtype=np.float32)
    np.savez(features_path, features=values, sample_id=np.asarray([r['sample_id'] for r in rows]),
             camera_id=np.asarray([r['camera_id'] for r in rows]), source_cluster_id=np.asarray([r['source_cluster_id'] for r in rows]))
    audit = dict(status='PASS_ACTUAL_OFFICIAL_INCEPTION2048_FEATURE_POOL', scope='final-test508',
        pool_role='generated-prediction', images=1524, assets=508, features_SHA256=sha(features_path),
        TF32=False, autocast=False, fullframe=True, observations=rows,
        model_binding=freeze['models']['feature-DQ'], package='torch-fidelity0.3.0', feature_layer='2048',
        feature_dtype='float32', same_batch_repeat_exact=True, GT_crop_or_CLIP_preprocessing_used=False, optimizer_updates=0)
    audit_path.write_text(json.dumps(audit))
    return freeze, audit, audit_path, features_path


def load_fixture(files):
    freeze, _, audit, features = files
    spec = {key: dict(path=str(path), SHA256=sha(path)) for key, path in [('audit', audit), ('features', features)]}
    return load_frozen_pool(spec, role='generated-prediction', binding=freeze['models']['feature-DQ'],
        sample_ids=freeze['population']['sample_ids'], clusters=freeze['population']['source_clusters_by_sample'])


def test_actual_npz_rows_are_paired_to_new_frozen_checkpoint_binding(actual_pool_files):
    values, audit = load_fixture(actual_pool_files)
    assert values.shape == (1524, 2048) and values.dtype == np.float32
    assert audit['model_binding'] == actual_pool_files[0]['models']['feature-DQ']


@pytest.mark.parametrize('problem', ['checkpoint', 'GT-array', 'cluster', 'crop', 'updated', 'nonrepeatable', 'feature-layer', 'nonfinite'])
def test_pool_rejects_mixed_checkpoint_reference_source_and_metric_protocol(actual_pool_files, problem):
    freeze, audit, audit_path, features_path = actual_pool_files
    if problem == 'checkpoint': audit['model_binding'] = freeze['models']['matched-R']
    elif problem == 'GT-array': audit['observations'][0]['array'] = 'GT_shared_renderer'
    elif problem == 'cluster':
        audit['observations'][0]['source_cluster_id'] = 'wrong-source'
        with np.load(features_path) as z: fields = {key: z[key] for key in z.files}
        fields['source_cluster_id'][0] = 'wrong-source'
        np.savez(features_path, **fields); audit['features_SHA256'] = sha(features_path)
    elif problem == 'crop': audit['GT_crop_or_CLIP_preprocessing_used'] = True
    elif problem == 'updated': audit['optimizer_updates'] = 1
    elif problem == 'nonrepeatable': audit['same_batch_repeat_exact'] = False
    elif problem == 'feature-layer': audit['feature_layer'] = '64'
    else:
        with np.load(features_path) as z: fields = {key: z[key] for key in z.files}
        fields['features'][0, 0] = np.nan
        np.savez(features_path, **fields); audit['features_SHA256'] = sha(features_path)
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError): load_fixture(actual_pool_files)
