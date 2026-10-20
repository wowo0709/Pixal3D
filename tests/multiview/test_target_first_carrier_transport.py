"""The carrier diagnostic must not leak disabled transport into the denoiser."""
import pytest
from experiments.deformnext_retargeting.probe_target_first_carriers_v2 import CarrierOnlyDataset


class Dataset:
    def __init__(self):
        self.item = dict(cond=object(),x_0=object(),transform_matrix=object(),mesh_scale=object(),
                         transport_target_keys=object(),transport_source_values=object(),
                         _sha256='original-input-id')
        self.rows = ['unchanged']
    def __getitem__(self, index):
        return self.item
    def __len__(self):
        return 1


def test_carrier_strips_only_recognized_transport_and_preserves_input_objects():
    original=Dataset();proxy=CarrierOnlyDataset(original,('target_keys','source_values'))
    item=proxy[0]
    assert len(proxy)==1 and proxy.rows is original.rows
    assert set(item)=={'cond','x_0','transform_matrix','mesh_scale','_sha256'}
    assert all(item[key] is original.item[key] for key in item)
    assert set(proxy.removed)=={'transport_target_keys','transport_source_values'}
    assert 'transport_target_keys' in original.item
    # A strict native call now receives no unsupported transport keyword.
    def native(*,cond,x_0,transform_matrix,mesh_scale,_sha256):
        return cond,x_0
    assert native(**item)==(original.item['cond'],original.item['x_0'])
    with pytest.raises(TypeError):native(**original.item)


def test_unknown_transport_is_an_error_instead_of_silently_dropped_input():
    original=Dataset();original.item['transport_unknown']=object()
    with pytest.raises(ValueError,match='unrecognized'):
        CarrierOnlyDataset(original,('target_keys','source_values'))[0]
