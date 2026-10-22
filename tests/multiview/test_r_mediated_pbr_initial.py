from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.evaluate_r_mediated_pbr_initial import (
    IDENTITY_FIELDS, MODES, audited_sampling, require_initial_identity,
)


def records():
    rows = [{**{key: f'{key}-{i}' for key in IDENTITY_FIELDS},
             'metrics': {'render_psnr': 24., 'render_lpips': .1}} for i in range(32)]
    return {mode: deepcopy(rows) for mode in MODES}


def test_initial_gate_requires_actual_predictions_and_all_input_identity():
    actual = records()
    require_initial_identity(actual)
    # Identical mean metrics cannot conceal different actual latent/noise/cameras.
    for key in IDENTITY_FIELDS:
        bad = deepcopy(actual)
        bad['top1-only'][17][key] = 'different'
        with pytest.raises(ValueError, match='input/noise/prediction'):
            require_initial_identity(bad)
    bad = deepcopy(actual)
    bad['raw-topk-plus-feature'][0]['metrics']['render_psnr'] += .001
    with pytest.raises(ValueError, match='rendering metrics'):
        require_initial_identity(bad)


def test_initial_gate_rejects_missing_duplicate_or_incomplete_assets():
    actual = records()
    bad = deepcopy(actual); bad.pop('top1-only')
    with pytest.raises(ValueError, match='four'):
        require_initial_identity(bad)
    bad = deepcopy(actual); bad['vanilla'][1]['sample_id'] = bad['vanilla'][0]['sample_id']
    with pytest.raises(ValueError, match='unique'):
        require_initial_identity(bad)
    bad = deepcopy(actual); bad['top1-plus-feature'].pop()
    with pytest.raises(ValueError, match='incomplete'):
        require_initial_identity(bad)


@pytest.mark.parametrize('fail', [False, True])
def test_actual_sample_capture_and_both_hooks_restore_on_failure(fail):
    def factory():
        return SimpleNamespace(sample=lambda **kw: kw['noise']+1)
    trainer = SimpleNamespace(get_sampler=factory)
    dataset = SimpleNamespace(rows=[dict(sample_id='one', source_cluster_id='cluster',
        conditioning_input_sha256='input')])
    pack = {key: torch.tensor([i], dtype=torch.float32) for i, key in enumerate([
        'cond', 'concat_cond', 'camera_angle_x', 'camera_distance', 'transform_matrix', 'mesh_scale'])}

    def original(trainer, dataset, index, steps, seed):
        prediction = trainer.get_sampler().sample(noise=torch.tensor([3.]))
        if fail:
            raise RuntimeError('actual generation failed')
        return prediction, torch.tensor([2.]), pack

    quick = SimpleNamespace(_sample=original)
    events = []
    try:
        with audited_sampling(quick, trainer, events):
            quick._sample(trainer, dataset, 0, 12, 42)
    except RuntimeError:
        assert fail
    assert trainer.get_sampler is factory
    assert quick._sample is original
    assert len(events) == (0 if fail else 1)
    if not fail:
        assert set(IDENTITY_FIELDS) == set(events[0])
        assert events[0]['actual_noise_SHA256'] != events[0]['actual_prediction_PBR_latent_SHA256']
        assert set(events[0]['actual_camera_SHA256']) == {
            'camera_angle_x', 'camera_distance', 'transform_matrix', 'mesh_scale'}
