from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting.deformation_query_metrics import adapt_ss_metrics


def test_physical_query_dataset_uses_ss_metrics_without_losing_actual_identity():
    config = dict(dataset_name='DeformNeXtDeformationQuerySS64',
                  quick_metrics=dict(enabled=True, panel_size=32))
    dataset = SimpleNamespace(deformation_query_channels=2048)
    received = []

    def native(trainer, actual_dataset, metric_config):
        received.append((actual_dataset, metric_config))
        assert metric_config['dataset_name'] == 'DeformNeXtSS64'
        return {'voxel_iou': .4}, 'panel', ['a']

    result = adapt_ss_metrics(native, None, dataset, config)
    assert result == ({'voxel_iou': .4}, 'panel', ['a'])
    assert received[0][0] is dataset
    assert received[0][1]['quick_metrics'] is config['quick_metrics']
    assert config['dataset_name'] == 'DeformNeXtDeformationQuerySS64'
    with pytest.raises(ValueError):
        adapt_ss_metrics(native, None, SimpleNamespace(), config)
