import pytest
import torch
from torch.utils.data import default_collate

from experiments.deformnext_retargeting.conditioning_controls import TransportControl, transport_diagnostic_context
from pixal3d.modules.matching_v2_conditioner import MatchingV2Conditioner


class CachedSS:
    transport_implementation = 'matching-v2'
    transport_arm = 'b3'

    def __init__(self):
        self.samples = [dict(
            cond=torch.full((4, 3, 2, 2), float(i)),
            x_0=torch.full((1, 2, 2, 2), float(i)),
            transform_matrix=torch.eye(4).repeat(4, 1, 1),
            transport_evidence_top1=torch.arange(8 * 1024).reshape(8, 1024).float(),
            transport_evidence_soft=torch.arange(8 * 1024).reshape(8, 1024).float() + 1,
            transport_p_use=torch.full((8,), .8),
            transport_cached_valid=torch.ones(8, dtype=torch.bool),
            transport_target_valid=torch.ones(8, dtype=torch.bool),
        ) for i in range(2)]

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is not None:
            raise ValueError('dense SS cannot split collation')
        return default_collate(batch)


@pytest.mark.parametrize('mode', ['source-off', 'wrong-corr'])
def test_transport_diagnostic_keeps_rgb_cameras_targets_and_cache(mode):
    dataset = CachedSS()
    original = dataset.collate_fn(dataset.samples)
    wrapped = TransportControl(dataset, mode)
    batch = wrapped.collate_fn([wrapped[i] for i in range(len(wrapped))])
    assert batch['transport_control'] == mode
    assert batch['cond'].shape[1] == 4
    for key, value in original.items():
        torch.testing.assert_close(batch[key], value, rtol=0, atol=0)
    assert 'transport_control' not in dataset[0]
    module = MatchingV2Conditioner(arm='b3')
    output = module(**{k.removeprefix('transport_'): v for k, v in batch.items()
                       if k.startswith('transport_') and k != 'transport_control'},
                    control=batch['transport_control'])
    if mode == 'source-off':
        assert not output.query_valid.any()
        assert torch.count_nonzero(output.usage_scale) == 0
    else:
        torch.testing.assert_close(output.evidence, torch.roll(original['transport_evidence_soft'], 1, 1))


def test_transport_controls_reject_native_legacy_and_ineffective_query_shuffle():
    dataset = CachedSS()
    dataset.transport_arm = 'v1'
    with pytest.raises(ValueError, match='matching-v2'):
        TransportControl(dataset, 'source-off')
    dataset.transport_arm = 'c1'
    dataset.transport_implementation = 'r9-native-v5'
    with pytest.raises(ValueError, match='matching-v2'):
        TransportControl(dataset, 'source-off')
    dataset.transport_implementation = 'matching-v2'
    with pytest.raises(ValueError, match='physical C2'):
        TransportControl(dataset, 'shuffled-deform')
    dataset.transport_arm = 'c2'
    wrapped = TransportControl(dataset, 'shuffled-deform')
    with pytest.raises(ValueError, match='at least two'):
        wrapped.collate_fn([dataset[0]])
    assert wrapped.collate_fn(dataset.samples)['transport_control'] == 'shuffled-deform'
    dataset.transport_arm = 'b3'
    with pytest.raises(ValueError, match='with a retargeter'):
        TransportControl(dataset, 'adapter-off')


def test_adapter_only_removal_preserves_weights_evidence_usage_and_restores_after_error():
    dataset = CachedSS()
    dataset.transport_arm = 'c1'
    batch = TransportControl(dataset, 'adapter-off').collate_fn(dataset.samples)
    assert batch['transport_control'] == 'normal'
    module = MatchingV2Conditioner(arm='c1', reject=True, reject_threshold=.5)
    with torch.no_grad():
        module.retargeter.net[-1].bias.fill_(.4)
        module.fusion_gate.fill_(.2)
    inputs = {k.removeprefix('transport_'): v for k, v in batch.items()
              if k.startswith('transport_') and k != 'transport_control'}
    inputs['target_query'] = torch.zeros_like(inputs['evidence_soft'])
    weights = {k: v.clone() for k, v in module.state_dict().items()}
    normal = module(**inputs)
    assert torch.count_nonzero(normal.retarget_residual) > 0
    with pytest.raises(RuntimeError, match='generation failed'):
        with transport_diagnostic_context(module, 'adapter-off'):
            removed = module(**inputs)
            assert torch.count_nonzero(removed.retarget_residual) == 0
            torch.testing.assert_close(removed.evidence, normal.base_evidence, rtol=0, atol=0)
            torch.testing.assert_close(removed.usage_scale, normal.usage_scale, rtol=0, atol=0)
            torch.testing.assert_close(removed.fusion_scale, normal.fusion_scale, rtol=0, atol=0)
            raise RuntimeError('generation failed')
    torch.testing.assert_close(module(**inputs).evidence, normal.evidence, rtol=0, atol=0)
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='loaded retargeter'):
        with transport_diagnostic_context(MatchingV2Conditioner(arm='b3'), 'adapter-off'):
            pass


@pytest.mark.parametrize('mode', ['normal', 'source-off', 'wrong-corr'])
def test_evaluation_control_reaches_native_inference_dispatch(mode):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin

    class InferenceSink:
        def get_inference_cond(self, cond, **kwargs):
            return {'cond': cond, **kwargs}

    class DispatchProbe(ImageConditionedProjMixin, InferenceSink):
        def __init__(self):
            self.deform_conditioner_config = {'implementation': 'matching-v2', 'arm': 'b3'}
            self.image_attn_mode = 'proj'
            self.conditioner = MatchingV2Conditioner(arm='b3')

        def _encode_deform_views(self, cond, camera_info, transport, control):
            self.received_rgb = cond
            self.received_camera = camera_info
            self.received_control = control
            evidence = {k: v for k, v in transport.items() if k != 'patch_grid'}
            self.output = self.conditioner(**evidence, control=control)
            result = {'global': self.output.evidence}
            return result, {'global': torch.zeros_like(self.output.evidence)}

    dataset = CachedSS()
    if mode == 'normal':
        batch = dataset.collate_fn(dataset.samples)
    else:
        batch = TransportControl(dataset, mode).collate_fn(dataset.samples)
    rgb = batch['cond']
    original_evidence = batch['transport_evidence_soft'].clone()
    cameras = {'transform_matrix': batch['transform_matrix'], 'coords': None}
    batch.pop('x_0')
    batch.pop('transform_matrix')
    batch['camera_info'] = cameras
    batch['transport_patch_grid'] = torch.zeros(2, 2, 2, 2, dtype=torch.long)
    bridge = DispatchProbe()
    result = bridge.get_inference_cond(**batch)
    assert bridge.received_control == mode
    assert bridge.received_rgb is rgb
    assert bridge.received_camera is cameras
    # The diagnostic scalar is consumed before conditioning reaches the sampler.
    assert not any(key.startswith('transport_') for key in result)
    if mode == 'source-off':
        assert torch.count_nonzero(bridge.output.usage_scale) == 0
    else:
        expected = torch.roll(original_evidence, 1, 1) if mode == 'wrong-corr' else original_evidence
        torch.testing.assert_close(result['cond']['global'], expected, rtol=0, atol=0)
