"""Join official PBR scores only when their exact scored pixels are paired.

The caller verifies each source file's closure before supplying its record.
Only arm labels change; score values and image audits are copied unchanged.
This is a DEV32 four-PBR join, not a model-selection or TEST reducer.
"""
from copy import deepcopy
from pathlib import Path
import re

from .common_geometry_pbr_contract import MODES


def require(value, message):
    if not value:
        raise ValueError(message)


def _key(row):
    require(type(row.get('camera_id')) is int and row['camera_id'] in (8, 9, 10),
            'integer target8/source9/source10 camera required')
    return row['arm'], row['sample_id'], row['camera_id']


def _official(record, registry):
    require(record.get('status') == 'PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION'
            and record.get('metric_registry_SHA256') == registry
            and record.get('model_precision') == 'float32,TF32off,noAMP'
            and record.get('full_panel') is True and record.get('no_model_training') is True
            and record.get('global_environment_changed') is False
            and record.get('GPU_sanity', {}).get('status') == 'PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY'
            and record['GPU_sanity'].get('CPU_model_forward_or_training_updates') == 0
            and bool(record.get('model_weights')),
            'original full-panel official GPU metric protocol and model weights required')


def merge_official_reports(new_report, reused_reports, *, expected_inputs,
                           metric_registry_SHA256):
    """Merge fresh rows with explicitly bound old arms, rejecting pixel drift.

    ``reused_reports`` entries contain ``mode``, ``source_arm``, ``record``, and
    ``spec`` (the absolute path and SHA256 checked by the caller). Expected
    inputs must include the actual shared-NPZ SHA256 for every case and camera.
    """
    _official(new_report, metric_registry_SHA256)
    expected = {}
    clusters = {}
    for row in expected_inputs:
        key = _key(row)
        require(key not in expected and key[0] in MODES,
                'unique four-PBR expected image identities required')
        require(isinstance(row.get('shared_npz_SHA256'), str)
                and re.fullmatch('[0-9a-f]{64}', row['shared_npz_SHA256']) is not None,
                'actual expected scored-pixel SHA256 required')
        require(clusters.setdefault(key[1], row['source_cluster_id']) == row['source_cluster_id'],
                'one source cluster per asset across PBR modes required')
        expected[key] = row
    require(len(clusters) == 32 and set(expected) == {
        (mode, sid, camera) for mode in MODES for sid in clusters for camera in (8, 9, 10)
    }, 'complete four-mode DEV32/384-view expected coverage required')

    scores = {}

    def add(row, mode=None):
        row = deepcopy(row)
        if mode is not None:
            row['arm'] = mode
        key = _key(row)
        require(key in expected and key not in scores,
                'unexpected or duplicate actual official metric image')
        want = expected[key]
        require(row.get('source_cluster_id') == want['source_cluster_id']
                and row.get('shared_npz_SHA256') == want['shared_npz_SHA256'],
                'official score belongs to another source or scored pixel file')
        scores[key] = row

    for row in new_report['per_view']:
        add(row)
    new_modes = {row['arm'] for row in new_report['per_view']}
    require(bool(new_modes) and len(scores) == 96 * len(new_modes),
            'each freshly scored PBR mode must cover all32 assets and three cameras')
    reuse_modes, provenance = set(), []
    for item in reused_reports:
        mode, record, spec = item['mode'], item['record'], item['spec']
        require(mode in MODES and mode not in new_modes and mode not in reuse_modes,
                'one disjoint official source per reused PBR mode required')
        require(isinstance(spec, dict) and set(spec) == {'path', 'SHA256'}
                and isinstance(spec['path'], str) and Path(spec['path']).is_absolute()
                and isinstance(spec['SHA256'], str)
                and re.fullmatch('[0-9a-f]{64}', spec['SHA256']) is not None,
                'explicit closed official source path and SHA256 required')
        _official(record, metric_registry_SHA256)
        require(record['model_weights'] == new_report['model_weights'],
                'official model weights differ across reused and fresh scores')
        selected = [row for row in record['per_view'] if row['arm'] == item['source_arm']]
        require(len(selected) == 96, 'reused source arm must have exactly96 actual views')
        for row in selected:
            add(row, mode)
        reuse_modes.add(mode)
        provenance.append(dict(mode=mode, source_arm=item['source_arm'], **deepcopy(spec)))
    require(set(scores) == set(expected), 'complete actual official384 coverage required')
    result = deepcopy(new_report)
    result.update(per_view=[scores[_key(row)] for row in expected_inputs],
                  actual_current_GPU_views=len(new_report['per_view']),
                  reused_closed_GPU_views=96 * len(reuse_modes),
                  reuse_provenance=provenance,
                  all384_rows_recomputed_this_job=not reuse_modes,
                  scores_and_image_audits_numerically_unchanged=True)
    return result
