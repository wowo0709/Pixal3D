"""Protect the query-domain boundary between incompatible ray experiments."""
import numpy as np
import pytest

from experiments.deformnext_retargeting.anchor_patch_head_training import validate_query_domain


def declared_rows():
    return [dict(role='train' if i < 512 else 'calibration', source_cluster_id=str(i),
                 query_unit='edited-target-patch-ray', inputs_relative_path='input.npz',
                 queries=1, usable_queries=1) for i in range(544)]


def test_rejects_pooled_source_ray_metadata(tmp_path):
    rows = declared_rows()
    rows[0]['query_unit'] = 'source-camera-pooled-ray'
    with pytest.raises(ValueError, match='edited-target-patch ray'):
        validate_query_domain(tmp_path, rows)


def test_rejects_source_overlap(tmp_path):
    rows = declared_rows()
    rows[-1]['source_cluster_id'] = rows[0]['source_cluster_id']
    with pytest.raises(ValueError, match='source clusters overlap'):
        validate_query_domain(tmp_path, rows)


def test_rejects_old_query_archive_under_new_metadata(tmp_path):
    np.savez(tmp_path / 'input.npz', query_unit='source-camera-pooled-ray',
             query_patch_indices=np.array([42], dtype=np.int32))
    with pytest.raises(ValueError, match='archive query domain differs'):
        validate_query_domain(tmp_path, declared_rows())
