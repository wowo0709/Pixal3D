"""A score-prior control must preserve every other trained component."""
import copy
import json

import pytest

from experiments.deformnext_retargeting.ray_group_study import recipe, source_record
from experiments.deformnext_retargeting.ray_score_prior_extension import verify_contrast, verify_lineage
from test_ray_group_study import source


def test_scalar_prior_control_rejects_extra_learning_rate_change(tmp_path):
    initial = source()
    initial.update(models={'denoiser': {}}, data_dir={'deform': '/fixed'})
    root = tmp_path / 'ray-group-study-v1'
    full = recipe(initial, root, 'ray-full')
    control = recipe(initial, root, 'ray-no-score-prior')
    verify_contrast(full, control)
    altered = copy.deepcopy(control)
    altered['trainer']['args']['lr'] = 2e-4
    with pytest.raises(ValueError):
        verify_contrast(full, altered)


def test_registry_extension_does_not_allow_changed_tensor_runtime(tmp_path):
    folder = tmp_path / 'score-prior-control-v1'
    folder.mkdir()
    parent = source_record()['component_SHA256'].copy()
    parent['experiments/deformnext_retargeting/ray_group_study.py'] = 'parent-registry'
    path = folder / 'source-isolation.json'
    path.write_text(json.dumps({'parent_components_SHA256': parent}))
    verify_lineage(tmp_path)
    parent['pixal3d/modules/ray_group_matching.py'] = 'changed-model'
    path.write_text(json.dumps({'parent_components_SHA256': parent}))
    with pytest.raises(ValueError, match='tensor'):
        verify_lineage(tmp_path)
