"""Explicit whole-conditioner initialization before native optimizer creation.

This adds no teacher labels or loss to joint SS training. Original19 and
SourceContrast19 can receive their equally trained source-feature prototypes;
their optimizer and denoiser continuation still belong to the native trainer.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch

from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
from pixal3d.modules.source_contrast_member_feature import SourceContrastMemberFeatureConditioner


CLASSES = {
    'original-NoDQ': MemberFeatureSourceRayConditioner,
    'source-contrast-NoDQ': SourceContrastMemberFeatureConditioner,
}
MARKER = 'retargeter.source_contrast_schema'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_teacher_initialization(module, spec):
    """Validate everything before copying, including the frozen matcher.

    The caller separately verifies canonical file4 receipts. Hash bindings
    here prevent an otherwise valid audit/checkpoint from silently changing.
    No optimizer, denoiser, dataset, RNG or global registry is accessed.
    """
    if (set(spec) != {'variant', 'checkpoint', 'checkpoint_SHA256', 'audit', 'audit_SHA256'}
            or spec['variant'] not in CLASSES):
        raise ValueError('explicit nineteen-parameter source teacher binding required')
    for key in ('checkpoint', 'audit'):
        if _sha(spec[key]) != spec[key+'_SHA256']:
            raise ValueError('source teacher '+key+' SHA256 mismatch')
    audit = json.loads(Path(spec['audit']).read_text())
    required = dict(status='PASS_ACTUAL_SOURCE_FEATURE_TEACHER_TRAIN',
        variant=spec['variant'], actual_source_parameters=19, epochs=7,
        optimizer_updates=112, TRAIN_assets=1536, source_clusters=1536,
        validation_source_overlap=0, global_batch=96, microbatch=96,
        accumulation=1, matcher_unchanged=True,
        actual_original_SS_or_PBR_training=False, denoiser_not_loaded_or_updated=True)
    if any(key not in audit or audit[key] != value for key, value in required.items()):
        raise ValueError('complete matched TRAIN1536/112-update source prototype required')
    if (not isinstance(module, CLASSES[spec['variant']])
            or sum(p.requires_grad for p in module.parameters()) != 19
            or any(p.requires_grad for p in module.q2c.parameters())):
        raise ValueError('actual nineteen-parameter variant and frozen matcher required')
    expected = module.state_dict()
    state = torch.load(spec['checkpoint'], map_location='cpu', weights_only=True)
    contrast = spec['variant'] == 'source-contrast-NoDQ'
    if (not isinstance(state, dict) or set(state) != set(expected)
            or (MARKER in expected) != contrast):
        raise ValueError('complete strict variant-specific state and marker required')
    for name, value in expected.items():
        loaded = state[name]
        if (not isinstance(loaded, torch.Tensor) or loaded.shape != value.shape
                or loaded.dtype != value.dtype or not torch.isfinite(loaded).all()):
            raise ValueError('invalid source teacher tensor: '+name)
        if name.startswith('q2c.') and not torch.equal(loaded, value.detach().cpu()):
            raise ValueError('teacher initialization would replace the frozen matcher')
    if contrast and not torch.equal(state[MARKER], torch.tensor([1], dtype=torch.uint8)):
        raise ValueError('source contrast checkpoint marker must equal one')
    module.load_state_dict(state, strict=True)
    if any(not torch.equal(value.detach().cpu(), state[name])
           for name, value in module.state_dict().items()):
        raise ValueError('source teacher initialization failed exact tensor roundtrip')
    return dict(variant=spec['variant'], checkpoint_SHA256=spec['checkpoint_SHA256'],
        audit_SHA256=spec['audit_SHA256'], source_tensors=19,
        additional_TRAIN_M1_source9_10_supervision=True, source_teacher_updates=112,
        frozen_matcher_exact=True, whole_conditioner_loaded_after_utility=True,
        original_denoiser_or_optimizer_loaded=False, cold_native_identity_expected=False)


@contextmanager
def source_input_binding(trainer_module, *, variant, teacher=None):
    """Bind a factory in this process; load teacher AFTER historical utility.

    The original trainer calls ``load_utility`` after constructing its source
    conditioner. Loading the teacher in the constructor would overwrite its
    learned score selector. Both steps here precede native optimizer creation.
    """
    if variant not in CLASSES or (teacher is not None and teacher['variant'] != variant):
        raise ValueError('one declared source-input design and matching initializer required')
    previous = trainer_module.MemberFeatureSourceRayConditioner
    if previous is not MemberFeatureSourceRayConditioner:
        raise ValueError('uncombined original native source trainer binding required')
    instances = []

    class Configured(CLASSES[variant]):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._source_input_initialization = None
            instances.append(self)

        def load_utility(self, state):
            if self._source_input_initialization is not None:
                raise ValueError('source utility/teacher initialization may run only once')
            super().load_utility(state)
            self._source_input_initialization = (
                load_teacher_initialization(self, teacher) if teacher is not None else
                dict(variant=variant, source_tensors=19, additional_TRAIN_M1_source9_10_supervision=False,
                     whole_conditioner_loaded_after_utility=False, cold_native_identity_expected=True))

    try:
        trainer_module.MemberFeatureSourceRayConditioner = Configured
        yield instances
    finally:
        trainer_module.MemberFeatureSourceRayConditioner = previous
