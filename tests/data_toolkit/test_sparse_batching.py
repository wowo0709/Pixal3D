import pytest
import torch

from data_toolkit.pipeline.sparse_batching import (
    batch_sparse_tensors,
    micro_batches,
    split_sparse_tensor,
)
from pixal3d.modules.sparse.basic import SparseTensor


def sparse(feats, coords):
    return SparseTensor(
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(coords, dtype=torch.int32),
    )


def test_sparse_batch_round_trip_preserves_order_and_values():
    first = sparse(
        [[1.0], [2.0]],
        [[0, 1, 2, 3], [0, 4, 5, 6]],
    )
    second = sparse([[3.0]], [[0, 7, 8, 9]])

    combined = batch_sparse_tensors([first, second])
    restored = split_sparse_tensor(combined)

    assert len(combined) == 2
    assert len(restored) == 2
    assert torch.equal(restored[0].feats, first.feats)
    assert torch.equal(restored[0].coords, first.coords)
    assert torch.equal(restored[1].feats, second.feats)
    assert torch.equal(restored[1].coords, second.coords)


def test_micro_batches_are_bounded_and_ordered():
    assert micro_batches(list(range(10)), 4) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]


@pytest.mark.parametrize("size", [True, 0, -1, 1.5])
def test_micro_batches_reject_invalid_size(size):
    with pytest.raises(ValueError, match="positive integer"):
        micro_batches([1, 2], size)


def test_batch_rejects_empty_or_mixed_sparse_types():
    with pytest.raises(ValueError, match="zero"):
        batch_sparse_tensors([])

    class OtherSparse(SparseTensor):
        pass

    first = sparse([[1.0]], [[0, 1, 2, 3]])
    second = OtherSparse(
        torch.tensor([[2.0]]),
        torch.tensor([[0, 4, 5, 6]], dtype=torch.int32),
    )
    with pytest.raises(TypeError, match="same sparse tensor type"):
        batch_sparse_tensors([first, second])


def test_batch_rejects_nonfinite_features_and_nonintegral_coordinates():
    nonfinite = sparse([[float("nan")]], [[0, 1, 2, 3]])
    with pytest.raises(ValueError, match="finite"):
        batch_sparse_tensors([nonfinite])

    invalid_coords = SparseTensor(
        torch.tensor([[1.0]]),
        torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
    )
    with pytest.raises(ValueError, match="integral"):
        batch_sparse_tensors([invalid_coords])


def test_split_rejects_noncontiguous_or_incomplete_layout():
    tensor = sparse(
        [[1.0], [2.0]],
        [[0, 1, 2, 3], [0, 4, 5, 6]],
    )
    tensor.register_spatial_cache("layout", [slice(1, 2)])

    with pytest.raises(ValueError, match="contiguous"):
        split_sparse_tensor(tensor)
