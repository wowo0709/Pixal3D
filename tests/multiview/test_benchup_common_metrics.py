import json

import numpy as np
from scipy.linalg import sqrtm

from experiments.benchup.aggregate_common_metrics import fid_lowrank
from experiments.benchup.write_common_table import main as write_table


def test_fid_lowrank_matches_covariance_definition():
    rng = np.random.default_rng(114)
    x = rng.normal(size=(30, 6))
    y = rng.normal(loc=0.2, size=(35, 6))
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    reference = np.square(x.mean(0) - y.mean(0)).sum() + np.trace(cx + cy - 2 * sqrtm(cx @ cy).real)
    assert np.isclose(fid_lowrank(x, y), reference, rtol=1e-7, atol=1e-7)


def test_fid_lowrank_zero_for_same_features():
    x = np.random.default_rng(8).normal(size=(5, 15))
    assert fid_lowrank(x, x) < 1e-7


def test_common_table_ranks_separately_for_each_alignment(tmp_path, monkeypatch):
    def values(psnr):
        return {'PSNR': psnr, 'SSIM': .9, 'M_IoU': .8, 'source_LPIPS': .1,
                'source_DreamSim': .2, 'source_CLIP': .9, 'FID': None, 'KID': None}
    data = {'status': 'PROVISIONAL_COMMON_SUCCESS_INPUT_IMAGE_METRICS', 'common_count': 4,
            'available_counts': {'A': 4, 'B': 4},
            'methods': {'A': {'raw': values(20), 'source_m0': values(10)},
                        'B': {'raw': values(10), 'source_m0': values(20)}}}
    source = tmp_path / 'metrics.json'; output = tmp_path / 'table.md'
    source.write_text(json.dumps(data))
    monkeypatch.setattr('sys.argv', ['write_common_table', '--input', str(source), '--output', str(output)])
    write_table()
    text = output.read_text()
    assert '| A | raw | 4 | **20.00000**' in text
    assert '| B | source_m0 | 4 | **20.00000**' in text
    assert '| B | raw | 4 | <u>10.00000</u>' in text
