import math

import pytest

from experiments.deformnext_retargeting.evaluate_pbr_streaming import aggregate_metrics


def test_pools_pixel_errors_before_log_and_weights_foreground():
    rows = [dict(render_psnr=10., base_color_psnr=10., foreground_psnr=10.,
                 foreground_base_color_psnr=10., foreground_fraction=.1, crop_lpips=.2),
            dict(render_psnr=20., base_color_psnr=20., foreground_psnr=20.,
                 foreground_base_color_psnr=20., foreground_fraction=.9, crop_lpips=.4)]
    result = aggregate_metrics(rows)
    assert result['render_psnr'] == pytest.approx(-10*math.log10(.055))
    assert result['foreground_psnr'] == pytest.approx(-10*math.log10(.019))
    assert result['crop_lpips'] == pytest.approx(.3)
    assert result['foreground_fraction'] == pytest.approx(.5)
