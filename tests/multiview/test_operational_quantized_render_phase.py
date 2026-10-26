"""The final-render admission uses a uniform whole-pixel one-percent limit."""
import pytest
import torch

from experiments.deformnext_retargeting.operational_quantized_render_phase import quantized_mask_agreement
from experiments.deformnext_retargeting.topology_labels import RasterAgreementError, mask_agreement


def test_campaign_case_wrapper_uses_unpatched_original_spawn():
    from experiments.deformnext_retargeting import run_selected_final_campaign as frozen
    from experiments.deformnext_retargeting.operational_three_campaign_entry import (
        MultiSourceCampaign, OriginalCampaign, operational_runtime,
    )
    with operational_runtime():
        assert frozen.Campaign is MultiSourceCampaign
        assert OriginalCampaign is not frozen.Campaign
        assert OriginalCampaign.spawn is not MultiSourceCampaign.spawn


def test_case_validator_replaces_only_the_gt_raster_gate(tmp_path):
    from experiments.deformnext_retargeting.operational_quantized_case_integrity import adapt_case_validator
    from experiments.deformnext_retargeting.selected_final_case_integrity import validate_case_reports
    adjusted, proof = adapt_case_validator(validate_case_reports, tmp_path / 'gate.json')
    assert adjusted is not validate_case_reports
    assert proof['changed_GT_raster_admission_expressions'] == 1
    assert proof['all_other_original_case_checks_unchanged'] is True


def test_thin_alpha_integer_pixel_boundary():
    raster = torch.zeros((64, 64), dtype=torch.bool)
    raster[16:46, 16:46] = True
    alpha = torch.zeros_like(raster)
    alpha[16:46, 16:46] = True
    # Use exactly 641 interior pixels and seven pixels beyond dilation.
    points = raster.nonzero()[:641]
    alpha.zero_()
    alpha[points[:, 0], points[:, 1]] = True
    alpha[0, :7] = True
    with pytest.raises(RasterAgreementError):
        mask_agreement(raster, alpha, alpha_cutouts=True)
    iou, containment = quantized_mask_agreement(raster, alpha, alpha_cutouts=True)
    assert 0 < iou < 1 and containment == pytest.approx(641 / 648)
    alpha[0, 7] = True
    with pytest.raises(RasterAgreementError):
        quantized_mask_agreement(raster, alpha, alpha_cutouts=True)


def test_final_render_gate_requires_nonempty_gt_alpha():
    empty = torch.zeros((8, 8), dtype=torch.bool)
    with pytest.raises(ValueError):
        quantized_mask_agreement(empty, empty, alpha_cutouts=True)
