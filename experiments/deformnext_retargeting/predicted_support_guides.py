"""A declared texture-decoding pyramid from an already predicted voxel support.

This policy preserves archived geometry. It does not recover the unrecorded
historical Shape decoder hierarchy: intermediate branches without final leaves
are omitted. Every compared texture must use this same explicit policy.
"""
import torch

POLICY = 'archived-predicted-support-pyramid4-v1'


def _packed(x):
    x = x.to(torch.int64)
    return (x[:, 0] * 1024 + x[:, 1]) * 1024 + x[:, 2]


def _unpacked(x):
    return torch.stack([x // 1024**2, (x // 1024) % 1024, x % 1024], dim=1)


def predicted_support_guides(root_coords, leaf_coords):
    """Return four (parent coordinates, eight positive-child bits) pairs.

    Roots retain the original latent row order, including roots with no final
    descendants. Children use native Channel2Spatial order: x + 2*y + 4*z.
    No feature values, ground truth, nearest neighbours, or interpolation enter.
    """
    for x, width in [(root_coords, 4), (leaf_coords, 3)]:
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != width
                or not len(x) or x.dtype not in [torch.int32, torch.int64]):
            raise ValueError('nonempty integer root[N,4] and leaf[M,3] coordinates required')
    if root_coords.device != leaf_coords.device:
        raise ValueError('root and predicted leaf support must share a device')
    if (torch.any(root_coords[:, 0] != 0) or torch.any(root_coords[:, 1:] < 0)
            or torch.any(root_coords[:, 1:] >= 64) or torch.any(leaf_coords < 0)
            or torch.any(leaf_coords >= 1024)):
        raise ValueError('one predicted Shape64 latent and voxel1024 support required')
    for x in [root_coords[:, 1:], leaf_coords]:
        if len(torch.unique(_packed(x))) != len(x):
            raise ValueError('duplicate predicted support coordinates are invalid')
    parents = root_coords[:, 1:].to(torch.int64)
    result = []
    for stage in range(4):
        children = _unpacked(torch.unique(_packed(leaf_coords // 2**(3-stage)), sorted=True))
        codes, order = torch.sort(_packed(parents))
        target = _packed(children // 2)
        positions = torch.searchsorted(codes, target)
        if torch.any(positions == len(codes)):
            raise ValueError('predicted leaf support has no original Shape latent ancestor')
        if not torch.equal(codes[positions], target):
            raise ValueError('predicted leaf support has no original Shape latent ancestor')
        parent_index = order[positions]
        child_bit = (children[:, 0] % 2) + 2*(children[:, 1] % 2) + 4*(children[:, 2] % 2)
        mask = torch.zeros((len(parents), 8), device=parents.device, dtype=torch.bool)
        mask[parent_index, child_bit] = True
        coords = torch.cat([torch.zeros_like(parents[:, :1]), parents], dim=1).to(root_coords.dtype)
        result.append((coords, mask))
        row, bit = mask.nonzero(as_tuple=True)
        offset = torch.stack([bit % 2, (bit // 2) % 2, (bit // 4) % 2], dim=1)
        parents = parents[row] * 2 + offset
    if not torch.equal(torch.sort(_packed(parents)).values, torch.sort(_packed(leaf_coords)).values):
        raise ValueError('guided decoder output support must exactly equal archived predicted support')
    return result
