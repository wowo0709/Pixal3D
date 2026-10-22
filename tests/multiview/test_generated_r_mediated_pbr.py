from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from experiments.deformnext_retargeting import generated_r_mediated_pbr as generated
from experiments.deformnext_retargeting.prepare_rgb_manifest import image_fingerprint
from experiments.deformnext_retargeting.projection_maps import anchor_patch_indices


@pytest.fixture
def case(tmp_path):
    paths = []
    for i in range(4):
        path = tmp_path/f'input{i}.png'
        Image.new('RGBA', (8, 8), (20+i, 40, 60, 255)).save(path); paths.append(str(path))
    transform = [[1., 0, 0, 0], [0, 0, -1., -2.], [0, 1., 0, 0], [0, 0, 0, 1.]]
    row = dict(sample_id='synthetic-unit-not-real-data', camera_angle_x=[.8]*4,
        transform_matrix=[transform]*4, mesh_scale=1., conditioning_input_schema='deformnext-edited-rgb-input-v1',
        target_anchor_kind='rgb', conditioning_camera_ids=[8, 9, 10, 11],
        target_anchor=paths[0], source_images=paths[1:],
        transport_cache=str(tmp_path/'image-omega.npz'), transport_cache_pbr=str(tmp_path/'image-pbr.npz'))
    row['conditioning_input_sha256'] = image_fingerprint(row, Path)
    for key in ['shape_latent', 'pbr_latent', 'target_geometry', 'source_geometry',
                'topology_correspondence', 'target_vertices', 'projection_root', 'match_root']:
        row[key] = '/forbidden-GT-or-legacy/'+key
    keys = np.zeros((1024, 2048), np.float16); keys[:, :1024] = 1.; keys[:, 1024:] = 9.
    ss = dict(schema=np.asarray('DEFORMNEXT_OMEGA_L12_PIXAL_VALUES_SS64_V1'),
        conditioning_input_sha256=np.asarray(row['conditioning_input_sha256']), omega_completed_updates=np.asarray(23597),
        target_keys=keys, source_keys=np.broadcast_to(keys, (3, 1024, 2048)),
        source_valid=np.ones((3, 1024), np.bool_), spatial_indices=np.asarray([-999]), patch_grid=np.asarray([-999]))
    values = np.zeros((3, 1024, 2048), np.float16); values[..., 1024:] = .75
    pbr = dict(schema=np.asarray('DEFORMNEXT_PIXAL_VALUES_PBR1024_V1'),
        conditioning_input_sha256=np.asarray(row['conditioning_input_sha256']),
        source_values=values, target_query=np.full((1024, 1024), 2., np.float16), patch_ids=np.asarray([-999]))
    np.savez(row['transport_cache'], **ss); np.savez(row['transport_cache_pbr'], **pbr)
    coords = torch.tensor([[0, 0, 0, 0], [0, 63, 63, 63], [0, 31, 32, 30]], dtype=torch.int32)
    return row, coords, ss, pbr


def test_generated_loader_opens_only_image_descriptors_and_keeps_NAF_and_support_order(case, monkeypatch):
    row, coords, _, _ = case
    original = np.load; opened, consumed = [], []
    forbidden = {'spatial_indices', 'patch_grid', 'patch_ids', 'semantic_indices', 'matching_logits', 'candidate_valid', 'deformation'}
    class Archive:
        def __init__(self, archive): self.archive = archive
        def __enter__(self): return self
        def __exit__(self, *args): self.archive.close()
        def __getitem__(self, key):
            consumed.append(key)
            assert key not in forbidden
            return self.archive[key]
    def load(path):
        opened.append(str(path)); assert str(path) in [row['transport_cache'], row['transport_cache_pbr']]
        return Archive(original(path))
    monkeypatch.setattr(generated.np, 'load', load)
    minimal = generated.image_only_row(row)
    assert set(minimal) == set(generated.IMAGE_ROW_FIELDS)
    assert not {'shape_latent', 'pbr_latent', 'target_geometry', 'source_geometry'} & set(minimal)
    actual = generated.load_generated_r_transport(minimal, coords, path_map={})
    assert set(opened) == {row['transport_cache'], row['transport_cache_pbr']}
    assert not forbidden & set(consumed)
    assert actual['source_values'].shape == (1, 3, 1024, 2048)
    assert torch.all(actual['source_values'][..., 1024:] == .75)
    assert torch.all(actual['target_keys'] == 1.) and torch.all(actual['target_query'] == 2.)
    assert torch.equal(actual['patch_ids'], torch.from_numpy(anchor_patch_indices(row, coords[:, 1:].numpy(), 64)))
    order = torch.tensor([2, 0, 1])
    again = generated.load_generated_r_transport(row, coords[order], path_map={})
    assert torch.equal(again['patch_ids'], actual['patch_ids'][order])


@pytest.mark.parametrize('change', ['RGB', 'camera', 'omega', 'cache-input'])
def test_changed_actual_input_or_old_Omega_cache_is_rejected(case, change):
    row, coords, ss, _ = case
    if change == 'RGB': Image.new('RGBA', (8, 8), (0, 0, 0, 255)).save(row['source_images'][0])
    elif change == 'camera': row['camera_angle_x'][0] += .1
    elif change == 'omega': ss['omega_completed_updates'] = np.asarray(0)
    elif change == 'cache-input': ss['conditioning_input_sha256'] = np.asarray('wrong')
    np.savez(row['transport_cache'], **ss)
    with pytest.raises(ValueError): generated.load_generated_r_transport(row, coords, path_map={})


@pytest.mark.parametrize('problem', ['float', 'duplicate', 'outside-grid', 'second-batch', 'empty'])
def test_actual_target_addresses_reject_invalid_generated_support(case, problem):
    row, coords, _, _ = case
    if problem == 'float': coords = coords.float()
    elif problem == 'duplicate': coords[1] = coords[0]
    elif problem == 'outside-grid': coords[0, 1] = 64
    elif problem == 'second-batch': coords[0, 0] = 1
    elif problem == 'empty': coords = coords[:0]
    with pytest.raises(ValueError): generated.actual_target_patch_ids(row, coords)


def test_generated_encoder_calls_actual_sparse_training_consumer_and_preserves_coords(case, monkeypatch):
    from pixal3d.trainers.flow_matching.r_mediated_pbr_source import RMediatedPBRNativeTrainer
    row, coords, _, _ = case
    transport = generated.load_generated_r_transport(row, coords, path_map={})
    calls = []
    def native(facade, image, camera, values):
        calls.append((facade, camera, values))
        facade._last_deform_condition = SimpleNamespace(query_valid=torch.ones(1, 3072, dtype=torch.bool))
        return ({'global': torch.ones(1, 5, 1024), 'proj': torch.ones(3, 2048)},
                {'global': torch.zeros(1, 5, 1024), 'proj': torch.zeros(3, 2048)})
    monkeypatch.setattr(RMediatedPBRNativeTrainer, '_encode_deform_views', native)
    extractor = SimpleNamespace(grid_resolution=64, image_size=1024)
    conditioner = SimpleNamespace(value_dim=2048, projection_mode='native-bilinear', appearance_mode='top1-only')
    images = torch.zeros(1, 4, 3, 1, 1).expand(1, 4, 3, 1024, 1024)
    result, facade = generated.encode_generated_r_pbr(extractor, images, {}, coords,
        conditioner=conditioner, transport=transport)
    assert calls[0][0] is facade and calls[0][1]['coords'] is coords and calls[0][2] is transport
    assert torch.equal(result['cond']['proj'].coords, coords) and result['cond']['proj'].feats.shape == (3, 2048)
    assert not result['neg_cond']['proj'].feats.any()


@pytest.mark.parametrize('mode', ['normal', 'encoder-error', 'CPU-noise-consumed'])
def test_hook_keeps_other_stages_and_restores_hook_and_offload_after_failure(case, monkeypatch, mode):
    from experiments.deformnext_retargeting import cascade_artifact_inputs
    row, coords, _, _ = case
    moved = []
    extractor = SimpleNamespace(use_naf_upsample=True, naf_model=object(),
        to=lambda device: moved.append('GPU'), cpu=lambda: moved.append('CPU'))
    expected = object()
    original = lambda *args, **kwargs: expected
    pipeline = SimpleNamespace(device=torch.device('cpu'), low_vram=True,
        image_cond_model_tex_1024=extractor, get_proj_cond_shape=original)
    monkeypatch.setattr(cascade_artifact_inputs, 'prepared_views',
        lambda *args: torch.zeros(1, 4, 3, 1, 1).expand(1, 4, 3, 1024, 1024))
    def encode(*args, **kwargs):
        if mode == 'encoder-error': raise RuntimeError('synthetic encoder error')
        if mode == 'CPU-noise-consumed': torch.randn(1)
        return expected, SimpleNamespace(_last_deform_condition=SimpleNamespace(query_valid=torch.ones(1, 3072, dtype=torch.bool)))
    monkeypatch.setattr(generated, 'encode_generated_r_pbr', encode)
    images = [Image.open(path).copy() for path in [row['target_anchor'], *row['source_images']]]
    events = []; rng = torch.get_rng_state().clone()
    def call():
        with generated.generated_r_pbr_conditioning(pipeline, conditioner=object(), row=row, path_map={}, events=events):
            assert pipeline.get_proj_cond_shape(object(), images, coords) is expected
            return pipeline.get_proj_cond_shape(extractor, images, coords,
                camera_angle_x=row['camera_angle_x'], distance=[2.]*4, mesh_scale=1.,
                transform_matrix=row['transform_matrix'], grid_resolution_override=64)
    try:
        if mode == 'normal':
            assert call() is expected and events[-1]['native_sparse_training_consumer_reused']
            assert torch.equal(rng, torch.get_rng_state())
        else:
            with pytest.raises(RuntimeError if mode == 'encoder-error' else ValueError): call()
        assert pipeline.get_proj_cond_shape is original and moved == ['GPU', 'CPU']
    finally:
        torch.set_rng_state(rng)
