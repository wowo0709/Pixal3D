from copy import deepcopy

import numpy as np
import pytest

from experiments.deformnext_retargeting.probe_sparse_pbr_source_geometry import validate_declared_input


def fixture():
    row = dict(split='train', target_anchor_kind='rgb', conditioning_camera_ids=[8, 9, 10, 11],
        latent_geometry_schema='deformnext-m1-render-world-v4', pbr_tokens=5,
        mesh_scale=.75, conditioning_input_sha256='a' * 64)
    latent = dict(schema=np.array(row['latent_geometry_schema']),
        coords=np.zeros((5, 3), dtype=np.uint8), mesh_scale=np.array(.75))
    transport = dict(schema=np.array('DEFORMNEXT_PIXAL_VALUES_PBR1024_V1'),
        source_values=np.zeros((3, 1024, 2048), dtype=np.float16),
        target_query=np.zeros((1024, 1024), dtype=np.float16),
        patch_ids=np.arange(5, dtype=np.int64), conditioning_input_sha256=np.array('a' * 64))
    return row, latent, transport


def test_real_input_contract_accepts_PBR_and_rejects_SS_or_different_RGB_identity():
    row, latent, transport = fixture()
    validate_declared_input(row, latent, transport)
    for update in [dict(source_values=transport['source_values'][..., :1024]),
                   dict(conditioning_input_sha256=np.array('b' * 64)),
                   dict(schema=np.array('other-cache'))]:
        with pytest.raises(ValueError):
            validate_declared_input(row, latent, dict(transport, **update))


def test_native_support_scale_camera_order_and_patch_addresses_are_mandatory():
    row, latent, transport = fixture()
    for update in [dict(mesh_scale=.8), dict(pbr_tokens=6),
                   dict(conditioning_camera_ids=[8, 9, 11, 10]), dict(split='validation')]:
        with pytest.raises(ValueError):
            validate_declared_input(dict(row, **update), latent, transport)
    bad = deepcopy(transport)
    bad['patch_ids'][0] = -1
    with pytest.raises(ValueError):
        validate_declared_input(row, latent, bad)
