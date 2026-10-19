import numpy as np
import pytest
import torch

from pixal3d.datasets.deformnext_feature import DeformNeXtFeaturePBR1024
from pixal3d.datasets.deformnext_deformation_query_pbr import DeformNeXtDeformationQueryPBR1024


@pytest.mark.parametrize('corruption', [None, 'RGB-hash', 'FRAME', 'legacy-deformation'])
def test_physical_context_keeps_native_PBR_pack_and_rejects_wrong_input(tmp_path, monkeypatch, corruption):
    target = np.zeros((1024,2048), dtype=np.float16)
    source = np.zeros((3,1024,2048), dtype=np.float16)
    target[:,1024:] = 1
    source[:,:,1024:] = 2
    path = tmp_path / 'shared-omega.npz'
    np.savez_compressed(path, target_keys=target, source_keys=source,
                        conditioning_input_sha256='wrong' if corruption == 'RGB-hash' else 'RGB')
    texture, shape, mapping = object(), object(), torch.tensor([1,4,8])
    pack = dict(pbr_feats=texture, shape_feats=shape, transport_patch_ids=mapping,
                transport_target_keys=torch.from_numpy(target[:,:1024].copy()),
                transport_source_keys=torch.from_numpy(source[:,:,:1024].copy()))
    if corruption == 'FRAME':
        pack['transport_target_keys'][0,0] = 3
    if corruption == 'legacy-deformation':
        pack['transport_deformation'] = torch.zeros(1024,2048)
    monkeypatch.setattr(DeformNeXtFeaturePBR1024, '__getitem__', lambda self,index: dict(pack))
    dataset = object.__new__(DeformNeXtDeformationQueryPBR1024)
    dataset.rows = [dict(transport_cache=str(path), conditioning_input_sha256='RGB')]
    dataset._resolve = lambda value: value
    if corruption:
        with pytest.raises(ValueError):
            dataset[0]
        return
    actual = dataset[0]
    assert actual['pbr_feats'] is texture and actual['shape_feats'] is shape
    assert actual['transport_patch_ids'] is mapping
    assert actual['transport_target_context'].shape == (1024,2048)
    assert actual['transport_source_context'].shape == (3,1024,2048)
    assert actual['transport_target_context'][:,1024:].count_nonzero() > 0
    assert torch.all(actual['transport_source_context'][:,:,1024:] == 2)
    assert not any(key in actual for key in ('transport_deformation','transport_spatial_indices'))
