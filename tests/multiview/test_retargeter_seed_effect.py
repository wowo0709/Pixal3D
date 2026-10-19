import copy
import pytest

from experiments.deformnext_retargeting.summarize_retargeter_seed_effect import validate_pair


def pair():
    base = dict(models={}, training_data={}, data_dir='fixed-manifest', num_gpus=2, num_nodes=1,
        training_seed=101, ckpt='none', load_dir='',
        trainer={'name': 'fixed-trainer', 'args': {'max_steps': 480, 'batch_size_per_gpu': 48,
            'batch_split': 1, 'deform_conditioner': {'arm': 'b2', 'implementation': 'r9-native-v5',
                'freeze_q2c': True, 'score_temperature': .005, 'query_dim': 1024}}},
        dataset={'args': {'transport_arm': 'b2'}}, validation={'dataset_args': {'transport_arm': 'b2'}})
    adapter = copy.deepcopy(base)
    adapter['trainer']['args']['deform_conditioner'].update(arm='c1', candidate_aggregation='top1', score_temperature=.05)
    adapter['dataset']['args']['transport_arm'] = 'c1'
    adapter['validation']['dataset_args']['transport_arm'] = 'c1'
    return base, adapter


def test_dormant_temperature_is_allowed_only_with_matching_top1_consumers():
    a, b = pair(); validate_pair(a, b)
    b['trainer']['args']['deform_conditioner']['candidate_aggregation'] = 'weighted'
    with pytest.raises(ValueError, match='raw-score Top1'): validate_pair(a, b)


def test_different_data_and_optimization_cannot_be_called_adapter_effect():
    a, b = pair(); b['data_dir'] = 'different-manifest'
    with pytest.raises(ValueError, match='common recipe'): validate_pair(a, b)
    a, b = pair(); b['trainer']['args']['optimizer'] = {'lr': 9e-4}
    with pytest.raises(ValueError, match='optimizer'): validate_pair(a, b)


def test_deformation_channels_are_not_part_of_the_adapter_only_comparison():
    a, b = pair(); b['trainer']['args']['deform_conditioner']['deformation_dim'] = 2048
    with pytest.raises(ValueError, match='without deformation'): validate_pair(a, b)
