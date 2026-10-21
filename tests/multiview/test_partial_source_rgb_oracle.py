import pytest
import torch

from experiments.deformnext_retargeting.partial_source_rgb_oracle import (
    oracle_dataset_class, replace_source_rgb,
)


def test_replaces_only_camera9_10_without_mutating_original():
    original = torch.arange(4.)[:, None, None, None].expand(4, 3, 8, 8).clone() / 4
    before = original.clone()
    out = replace_source_rgb(original, {9: torch.ones_like(original[0]),
                                        10: torch.zeros_like(original[0])})
    assert torch.equal(original, before)
    assert torch.equal(out[0], before[0]) and torch.equal(out[3], before[3])
    assert torch.equal(out[1], torch.ones_like(out[1]))
    assert torch.equal(out[2], torch.zeros_like(out[2]))
    with pytest.raises(ValueError):
        replace_source_rgb(original, {8: original[0], 9: original[1]})
    with pytest.raises(ValueError):
        replace_source_rgb(original, {9: original[0], 10: original[1] + float('nan')})


def test_adapter_preserves_non_rgb_and_rejects_stale_transport():
    class Dataset:
        image_size = 8
        rows = [dict(sample_id='one', conditioning_camera_ids=[8, 9, 10, 11])]

        def __getitem__(self, index):
            return self.item

        def _resolve(self, path):
            return path

    events, captures = {}, {'one': None}
    cls = oracle_dataset_class(Dataset, {'one': {9: 'rgb9', 10: 'rgb10'}},
        lambda path, size: torch.full((3, size, size), .5 if path == 'rgb9' else .75),
        events, captures)
    ds = cls()
    ds.item = dict(cond=torch.zeros(4, 3, 8, 8), camera=torch.eye(4), x_0=torch.ones(1))
    result = ds[0]
    assert result['camera'] is ds.item['camera'] and result['x_0'] is ds.item['x_0']
    assert events['one']['actual_view_SHA256'][1] == events['one']['teacher_view_SHA256']['9']
    assert events['one']['actual_view_SHA256'][2] == events['one']['teacher_view_SHA256']['10']
    assert captures['one'].shape == (4, 3, 8, 8)
    ds.item['transport_source_values'] = torch.zeros(1)
    with pytest.raises(ValueError):
        ds[0]
