"""Join individual and distribution metrics through actual scored RGB bytes."""
from __future__ import annotations

import hashlib

import numpy as np

from .extract_official_inception import render_uint8


def scored_image_observation(arrays, *, sample_id, camera_id, source_cluster_id, archive, archive_SHA256):
    values = {name: arrays[name] for name in ['prediction', 'GT_shared_renderer', 'GT_alpha']}
    if (any(value.dtype != np.float32 or not np.isfinite(value).all() for value in values.values())
            or values['prediction'].shape != (3, 512, 512)
            or values['GT_shared_renderer'].shape != (3, 512, 512)
            or values['GT_alpha'].shape != (512, 512)):
        raise ValueError('actual native512 FP32 RGB/reference/alpha arrays required')
    digest = lambda value: hashlib.sha256(value.tobytes()).hexdigest()
    return dict(sample_id=sample_id, camera_id=camera_id, source_cluster_id=source_cluster_id,
        archive=str(archive), archive_SHA256=archive_SHA256,
        prediction_float_RGB_SHA256=digest(values['prediction']),
        reference_float_RGB_SHA256=digest(values['GT_shared_renderer']),
        reference_alpha_SHA256=digest(values['GT_alpha']),
        prediction_CHW_uint8_SHA256=digest(render_uint8(values['prediction'])),
        reference_CHW_uint8_SHA256=digest(render_uint8(values['GT_shared_renderer'])))


def validate_metric_image_join(eight, distribution, *, sample_ids, clusters, models):
    """Reject relabelled models, rerenders, different references and conversions."""
    required = ['scored_image_observations', 'input_SHA256']
    if (any(key not in eight for key in required)
            or 'feature_image_observations' not in distribution
            or 'reference_image_observations' not in distribution):
        raise ValueError('actual same-image provenance required for all ten metrics')
    if (tuple(eight['scored_image_observations']) != tuple(models)
            or tuple(distribution['feature_image_observations']) != tuple(models)):
        raise ValueError('same-image provenance must retain every ordered frozen model')
    keys = {(sid, cid) for sid in sample_ids for cid in (8, 9, 10)}

    def index(rows):
        observed = {}
        for row in rows:
            key = row['sample_id'], row['camera_id']
            if (key in observed or key not in keys
                    or row['source_cluster_id'] != clusters[row['sample_id']]):
                raise ValueError('image provenance sample/camera/source differs or is duplicated')
            observed[key] = row
        if set(observed) != keys:
            raise ValueError('complete paired image provenance required; no dropped views')
        return observed

    reference_rows = index(distribution['reference_image_observations'])
    references, reference_archives = {}, {}
    for alias in models:
        scored = index(eight['scored_image_observations'][alias])
        features = index(distribution['feature_image_observations'][alias])
        for key, row in scored.items():
            if eight['input_SHA256'].get(row['archive']) != row['archive_SHA256']:
                raise ValueError('scored image archive lacks the bound eight-metric input digest')
            feature = features[key]
            if (feature['array'] != 'prediction'
                    or feature['archive'] != row['archive']
                    or feature['archive_SHA256'] != row['archive_SHA256']
                    or feature['original_float_RGB_SHA256'] != row['prediction_float_RGB_SHA256']
                    or feature['converted_CHW_uint8_SHA256'] != row['prediction_CHW_uint8_SHA256']):
                raise ValueError('distribution prediction must use this model\'s exact scored RGB and conversion')
            reference = tuple(row[name] for name in ['reference_float_RGB_SHA256',
                'reference_CHW_uint8_SHA256', 'reference_alpha_SHA256'])
            if references.setdefault(key, reference) != reference:
                raise ValueError('every frozen model must use the identical shared reference RGB and alpha')
            reference_archives.setdefault(key, set()).add((row['archive'], row['archive_SHA256']))
    for key, row in reference_rows.items():
        if (row['array'] != 'GT_shared_renderer'
                or (row['archive'], row['archive_SHA256']) not in reference_archives[key]
                or row['original_float_RGB_SHA256'] != references[key][0]
                or row['converted_CHW_uint8_SHA256'] != references[key][1]):
            raise ValueError('distribution reference must use the identical scored shared GT pixels and conversion')
    return dict(status='PASS_SAME_ACTUAL_SCORED_RGB_ACROSS_ALL_TEN_METRICS',
        prediction_images=len(models)*len(keys), reference_images=len(keys),
        native_float_RGB_and_official_fullframe_uint8_identities_verified=True,
        exact_model_sample_camera_source_and_archive_bindings_verified=True,
        shared_reference_RGB_and_alpha_identical_across_models=True)
