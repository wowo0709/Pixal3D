from __future__ import annotations

from collections.abc import Sequence

import torch


INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def micro_batches(tasks, size: int) -> list[list]:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("micro-batch size must be a positive integer")
    values = list(tasks)
    return [
        values[index : index + size]
        for index in range(0, len(values), size)
    ]


def _validate_layout(tensor) -> tuple[slice, ...]:
    feats = getattr(tensor, "feats", None)
    coords = getattr(tensor, "coords", None)
    if not isinstance(feats, torch.Tensor) or not isinstance(
        coords, torch.Tensor
    ):
        raise TypeError("sparse tensors must expose tensor feats and coords")
    if feats.ndim < 1 or coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("invalid sparse tensor dimensions")
    if feats.shape[0] != coords.shape[0] or feats.shape[0] == 0:
        raise ValueError("sparse features and coordinates must be nonempty")
    if not bool(torch.isfinite(feats).all()):
        raise ValueError("sparse features must be finite")
    if coords.dtype not in INTEGER_DTYPES:
        raise ValueError("sparse coordinates must use an integral dtype")

    layout = tuple(tensor.layout)
    if not layout:
        raise ValueError("sparse layout cannot be empty")
    offset = 0
    for batch_index, item in enumerate(layout):
        if (
            not isinstance(item, slice)
            or item.step not in (None, 1)
            or item.start != offset
            or item.stop is None
            or item.stop <= item.start
        ):
            raise ValueError("sparse layout must be contiguous and nonempty")
        if not bool(torch.all(coords[item, 0] == batch_index)):
            raise ValueError("sparse layout batch coordinates are not contiguous")
        offset = item.stop
    if offset != feats.shape[0]:
        raise ValueError("sparse layout must cover every coordinate")
    return layout


def batch_sparse_tensors(tensors: Sequence) -> object:
    values = list(tensors)
    if not values:
        raise ValueError("cannot batch zero sparse tensors")
    tensor_type = type(values[0])
    if any(type(value) is not tensor_type for value in values):
        raise TypeError("all values must use the same sparse tensor type")

    reference_feats = values[0].feats
    reference_coords = values[0].coords
    for value in values:
        layout = _validate_layout(value)
        if len(layout) != 1:
            raise ValueError("batch inputs must each contain one sparse sample")
        if (
            value.feats.dtype != reference_feats.dtype
            or value.feats.device != reference_feats.device
            or value.feats.shape[1:] != reference_feats.shape[1:]
        ):
            raise ValueError("sparse feature contracts must match")
        if (
            value.coords.dtype != reference_coords.dtype
            or value.coords.device != reference_coords.device
            or value.coords.shape[1:] != reference_coords.shape[1:]
        ):
            raise ValueError("sparse coordinate contracts must match")

    combined = tensor_type.from_tensor_list(
        [value.feats for value in values],
        [value.coords for value in values],
    )
    if type(combined) is not tensor_type or len(_validate_layout(combined)) != len(
        values
    ):
        raise ValueError("sparse batch output count or type is invalid")
    return combined


def split_sparse_tensor(tensor) -> list:
    layout = _validate_layout(tensor)
    outputs = []
    for item in layout:
        coords = tensor.coords[item].clone()
        coords[:, 0] = 0
        output = type(tensor)(tensor.feats[item], coords)
        if len(_validate_layout(output)) != 1:
            raise ValueError("split sparse output must contain one sample")
        outputs.append(output)
    if len(outputs) != len(layout):
        raise ValueError("split sparse output count is invalid")
    return outputs
