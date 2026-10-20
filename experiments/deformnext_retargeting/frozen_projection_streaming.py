"""Stream frozen per-image projection while retaining the full denoiser batch.

This optional process-local adapter leaves the native extractor untouched on
disk. It bounds the dense NAF1024 intermediates, which need no backward graph.
Camera/image ordering and the native multiview averaging remain unchanged.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import torch


def stream_projection(original, model, image, camera_angle_x=None, distance=None,
                      mesh_scale=None, transform_matrix=None, *, max_assets=1):
    if max_assets < 1 or not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError('explicit positive image chunk and native BCHW tensor required')
    batch = len(image)
    if batch == 0:
        raise ValueError('empty image batch')
    cameras = (camera_angle_x, distance, mesh_scale, transform_matrix)
    for value in cameras:
        if value is not None and (not isinstance(value, torch.Tensor)
                                  or value.ndim == 0 or len(value) != batch):
            raise ValueError('native per-image camera arrays must match the image batch')
    if any(p.requires_grad for p in model.model.parameters()):
        raise ValueError('streaming is only valid for the frozen image encoder')
    if model.naf_model is not None and any(p.requires_grad for p in model.naf_model.parameters()):
        raise ValueError('streaming requires the NAF encoder to be frozen')
    result = None
    with torch.no_grad():
        for start in range(0, batch, max_assets):
            stop = min(batch, start+max_assets)
            args = [None if value is None else value[start:stop] for value in cameras]
            values = original(model, image[start:stop], *args)
            if result is None:
                result = tuple(value.new_empty((batch, *value.shape[1:])) for value in values)
            if len(values) != len(result):
                raise ValueError('native projection tuple changed')
            for target, value in zip(result, values):
                if (value.shape[0] != stop-start or value.shape[1:] != target.shape[1:]
                        or value.dtype != target.dtype or value.device != target.device):
                    raise ValueError('native image projection shape/dtype/device changed across chunks')
                target[start:stop].copy_(value)
            del values
    return result


def install_frozen_projection_streaming(*, max_assets=1, audit_directory=None):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

    cls = DinoV3ProjFeatureExtractor
    original = cls._forward_single_view
    if getattr(original, '_frozen_projection_streaming', False):
        raise ValueError('streaming adapter already installed in this process')
    audited = False

    @functools.wraps(original)
    def forward(self, image, camera_angle_x=None, distance=None,
                mesh_scale=None, transform_matrix=None):
        nonlocal audited
        # First actual pair: compare the untouched native batched extractor
        # with streaming before allowing any optimizer update.
        if not audited and audit_directory is not None and len(image) >= 2:
            cameras = (camera_angle_x, distance, mesh_scale, transform_matrix)
            args = [None if x is None else x[:2] for x in cameras]
            with torch.no_grad():
                reference = original(self, image[:2], *args)
                streamed = stream_projection(original, self, image[:2], *args, max_assets=max_assets)
            rows = []
            for left, right in zip(reference, streamed):
                delta = (left.float()-right.float()).abs()
                row = dict(shape=list(left.shape), dtype=str(left.dtype),
                           exactly_equal=torch.equal(left, right),
                           maximum_absolute_difference=float(delta.max()),
                           mean_absolute_difference=float(delta.mean()),
                           allclose=torch.allclose(left, right, atol=2e-5, rtol=2e-5))
                if not row['allclose']:
                    raise ValueError('actual native image projection streaming parity failed: '+str(row))
                rows.append(row)
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            out = Path(audit_directory)
            out.mkdir(parents=True, exist_ok=True)
            (out/f'projection-streaming-parity.rank{rank}.json').write_text(json.dumps(dict(
                status='PASS_ACTUAL_NATIVE_PROJECTION_BATCH2_STREAMING_PARITY',
                image_chunk=max_assets, actual_native_images=2, outputs=rows,
                denoiser_batch_unchanged=True, gradient_accumulation_added=False,
                GT_or_model_weights_changed=False), indent=2)+'\n')
            audited = True
            del reference, streamed
        return stream_projection(original, self, image, camera_angle_x, distance,
                                 mesh_scale, transform_matrix, max_assets=max_assets)

    forward._frozen_projection_streaming = True
    cls._forward_single_view = forward

