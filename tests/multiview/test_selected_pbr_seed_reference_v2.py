"""Reference-scope regression tests; all training records here are synthetic."""
from copy import deepcopy
import hashlib

import pytest
from PIL import Image

from experiments.deformnext_retargeting.collect_selected_pbr_seed_confirmation_v2 import actual_GT_pixels
from experiments.deformnext_retargeting.pbr_initial_render_identity import POLICY, RENDER_ABSOLUTE_LIMITS
from experiments.deformnext_retargeting.selected_pbr_seed_confirmation import summarize as legacy_summary
from experiments.deformnext_retargeting.selected_pbr_seed_confirmation_v2 import (
    REFERENCE_SCOPE, SUMMARY_SCHEMA, summarize,
)
from test_selected_pipeline_freeze_v2 import pbr_inputs


def different_seed_references():
    records, configs, hashes, kwargs = pbr_inputs('top1-only')
    for record in records[202].values():
        for pixels in record['GT_PNG_pixel_SHA256'].values():
            pixels['0'] = 'f' * 64
        record['initial_render_identity'] = dict(
            policy=POLICY, actual_inputs_noise_GT_and_PBR_latent_exact=True,
            render_metrics_within_declared_absolute_bounds=True,
            render_absolute_limits=deepcopy(RENDER_ABSOLUTE_LIMITS),
            unlisted_metrics_require_exact_equality=True,
        )
    return records, configs, hashes, kwargs


def test_independent_render_references_preserve_metrics_and_strict_within_seed_scope():
    r, c, h, kw = different_seed_references()
    before = deepcopy((r, c, h, kw))
    result = summarize(r, c, h, **kw)
    assert result['summary_schema'] == SUMMARY_SCHEMA
    assert result['reference_scope'] == REFERENCE_SCOPE
    assert result['actual_GT_pixels_exact_within_training_seed_models_and_epochs']
    assert not result['actual_GT_pixels_exact_between_training_seeds']
    assert result['cross_seed_quality_pool_or_common_pixel_reference_not_claimed']
    assert result['metric_values_unchanged']
    assert not result['source_preservation_measured']
    assert not result['generated_geometry_measured']
    assert not result['automatic_model_adoption']
    assert (r, c, h, kw) == before
    assert result['actual_GT_PNG_pixel_SHA256_by_seed_mode'][202]['vanilla'] == r[202]['vanilla']['GT_PNG_pixel_SHA256']
    with pytest.raises(ValueError, match='GT'):
        legacy_summary(r, c, h, **kw)


def test_bit_identical_reference_case_remains_supported():
    r, c, h, kw = pbr_inputs('top1-only')
    result = summarize(r, c, h, **kw)
    assert result['actual_GT_pixels_exact_between_training_seeds']


@pytest.mark.parametrize('damage', [
    'within-seed-model', 'within-seed-epoch', 'missing-epoch', 'missing-producer',
    'wrong-policy', 'unbounded-producer', 'changed-bound', 'changed-input',
    'changed-noise', 'changed-GT-latent', 'changed-camera', 'changed-cold-latent',
    'excess-render-change', 'incomplete-updates', 'changed-stream', 'changed-frozen-base',
])
def test_cross_seed_render_policy_does_not_relax_training_or_within_seed_identity(damage):
    r, c, h, kw = different_seed_references()
    record = r[202]['top1-only']
    first = record['validation_rows'][0]['quick_per_asset'][0]
    if damage == 'within-seed-model':
        for pixels in record['GT_PNG_pixel_SHA256'].values():
            pixels['0'] = 'e' * 64
    elif damage == 'within-seed-epoch':
        record['GT_PNG_pixel_SHA256']['1900']['0'] = 'e' * 64
    elif damage == 'missing-epoch':
        del record['GT_PNG_pixel_SHA256']['1900']
    elif damage == 'missing-producer':
        del record['initial_render_identity']
    elif damage == 'wrong-policy':
        record['initial_render_identity']['policy'] = 'relaxed'
    elif damage == 'unbounded-producer':
        record['initial_render_identity']['render_metrics_within_declared_absolute_bounds'] = False
    elif damage == 'changed-bound':
        record['initial_render_identity']['render_absolute_limits']['psnr'] = 1.
    elif damage.startswith('changed-') and damage not in ('changed-stream', 'changed-frozen-base'):
        key = {'changed-input':'actual_RGB_input_SHA256', 'changed-noise':'actual_noise_SHA256',
               'changed-GT-latent':'actual_GT_PBR_latent_SHA256', 'changed-camera':'actual_camera_SHA256',
               'changed-cold-latent':'actual_prediction_PBR_latent_SHA256'}[damage]
        first[key] = 'changed'
    elif damage == 'excess-render-change':
        first['metrics']['render_psnr'] += 1.
    elif damage == 'incomplete-updates':
        record['completion_proof']['ranks'][0]['actual_optimizer_updates'] = 380
    elif damage == 'changed-stream':
        record['training_trace_digests']['0'] = 'e' * 64
    else:
        kw['frozen_base_proofs'][202]['top1-only']['actual_original_frozen_NN_base_tensors_exact'] = False
    with pytest.raises(ValueError):
        summarize(r, c, h, **kw)


class ClosedFixture:
    def __init__(self):
        self.visited = []

    def check(self, path, receipt):
        assert path.is_file()
        assert receipt == path.parents[2] / 'file4-closure.json'
        self.visited.append(path)


def test_actual_png_reader_hashes_gt_column_and_all_epochs(tmp_path):
    rows = []
    check = ClosedFixture()
    gt = Image.new('RGB', (8, 8), (10, 20, 30))
    expected = hashlib.sha256(gt.tobytes()).hexdigest()
    for step in (0, 380):
        root = tmp_path / 'validation-panels' / f'step{step:07d}'
        root.mkdir(parents=True)
        panel = Image.new('RGB', (24, 8), (step % 255, 250, 0))
        panel.paste(gt, (0, 0))
        panel.save(root / 'case000.png')
        rows.append(dict(step=step, quick_per_asset=[dict(sample_id='asset')]))
    result = actual_GT_pixels(tmp_path, rows, check)
    assert result == {'0': {'asset': expected}, '380': {'asset': expected}}
    assert len(check.visited) == 2
    changed = tmp_path / 'validation-panels/step0000380/case000.png'
    with Image.open(changed) as image:
        image.putpixel((0, 0), (11, 20, 30))
        image.save(changed)
    assert actual_GT_pixels(tmp_path, rows, check) != result


def test_actual_png_reader_rejects_non_native_layout(tmp_path):
    root = tmp_path / 'validation-panels/step0000000'
    root.mkdir(parents=True)
    Image.new('RGB', (16, 8)).save(root / 'case000.png')
    with pytest.raises(ValueError, match='panel'):
        actual_GT_pixels(tmp_path, [dict(step=0, quick_per_asset=[dict(sample_id='asset')])], ClosedFixture())
