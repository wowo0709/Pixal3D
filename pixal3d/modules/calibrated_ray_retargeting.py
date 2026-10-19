"""Initialize correspondence from Q/K calibration without replacing native R.

Only seven matching tensors transfer. Native value/output projections keep
their own SS/PBR widths and initialization; the output must remain zero.
The actual fit epoch and scientific eligibility belong to the experiment
runtime, rather than this reusable weight loader.
"""
from pathlib import Path
import hashlib

import torch

from .direct_ray_retargeting import DirectRayRetargetingConditioner
from .direct_ray_retargeting_no_query import DirectRayRetargetingNoQueryConditioner


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8388608), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value):
    data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


FEATURE_NAMES = {'value.weight', 'output.weight'}


def load_calibrated_matching(spec):
    required = {'checkpoint', 'checkpoint_SHA256', 'reference_conditioner', 'reference_SHA256'}
    if set(spec) != required:
        raise ValueError('explicit calibrated and cold reference paths/SHA256 required')
    paths = {spec['checkpoint']: spec['checkpoint_SHA256'],
             spec['reference_conditioner']: spec['reference_SHA256']}
    if any(file_sha(path) != digest for path, digest in paths.items()):
        raise ValueError('calibrated matching artifact SHA256 differs')
    raw = torch.load(spec['reference_conditioner'], map_location='cpu', weights_only=True)
    reference = {key[len('ray_readout.'):]: value for key, value in raw.items() if key.startswith('ray_readout.')}
    calibrated = torch.load(spec['checkpoint'], map_location='cpu', weights_only=True)
    if (len(reference) != 9 or set(calibrated) != set(reference)
            or any(not isinstance(value, torch.Tensor) for value in calibrated.values())):
        raise ValueError('pure nine-tensor calibrated readout and native cold reference required')
    for name, value in calibrated.items():
        if value.shape != reference[name].shape or not torch.isfinite(value).all():
            raise ValueError('calibrated tensor shape or finite-value contract differs')
    if torch.count_nonzero(reference['output.weight']) or torch.count_nonzero(calibrated['output.weight']):
        raise ValueError('zero feature output must survive auxiliary matching calibration')
    anchored = {key: value.clone() for key, value in reference.items()}
    anchored['query.weight'].zero_()
    changed = {key for key, value in calibrated.items() if not torch.equal(value, anchored[key])}
    if changed != {'query.weight', 'key.weight'}:
        raise ValueError('only Q/K may differ from the prior-anchored cold readout')
    if any(file_sha(path) != digest for path, digest in paths.items()):
        raise ValueError('matching initializer changed during loading')
    return {key: value.clone() for key, value in calibrated.items() if key not in FEATURE_NAMES}


def install_calibrated_matching(head, matching):
    original = head.state_dict()
    if (set(matching) != set(original)-FEATURE_NAMES
            or torch.count_nonzero(original['output.weight'])
            or any(matching[key].shape != original[key].shape for key in matching)):
        raise ValueError('compatible seven-tensor matching initializer and zero native output required')
    feature = {key: original[key].detach().clone() for key in FEATURE_NAMES}
    head.load_state_dict(dict(original, **matching), strict=True)
    if any(not torch.equal(head.state_dict()[key], value) for key, value in feature.items()):
        raise ValueError('native SS/PBR value or output projection changed')
    return {key: tensor_sha(value) for key, value in head.state_dict().items()}


class _CalibratedMatchingInitializer:
    def __init__(self, *, calibrated_matching, **kwargs):
        super().__init__(**kwargs)
        matching = load_calibrated_matching(calibrated_matching)
        self._calibrated_initial_SHA256 = install_calibrated_matching(self.ray_readout, matching)
        self._calibrated_matching_source = dict(calibrated_matching)

    def assert_calibrated_initializer(self):
        actual = {key: tensor_sha(value) for key, value in self.ray_readout.state_dict().items()}
        if actual != self._calibrated_initial_SHA256:
            raise ValueError('calibrated matching or native feature initialization was overwritten')
        return dict(matching_tensors_loaded=7, auxiliary_fitted_tensors=2,
            native_feature_tensors_preserved=2, output_projection_zero=True,
            state_SHA256=actual, source=dict(self._calibrated_matching_source),
            value_width=self.value_dim, retargeter_input_channels=self.retargeter.input_dim)


class CalibratedDirectRayRetargetingConditioner(_CalibratedMatchingInitializer,
                                               DirectRayRetargetingConditioner):
    """Unchanged Direct+DQ forward, initialized with calibrated correspondence."""


class CalibratedDirectRayNoQueryConditioner(_CalibratedMatchingInitializer,
                                           DirectRayRetargetingNoQueryConditioner):
    """Same calibrated correspondence with deformation channels physically absent."""
