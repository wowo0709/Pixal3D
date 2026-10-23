"""Archive the actual Shape decoder guides, including dead branches and logits."""
from fractions import Fraction
import hashlib
from pathlib import Path

import torch

POLICY = 'captured-native-shape-subdivision-guides-v1'


def tensor_sha(value):
    value = value.detach().contiguous().cpu()
    header = str((str(value.dtype), tuple(value.shape))).encode()
    return hashlib.sha256(header + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def save_guides(path, shape, guides):
    """Copy four actual native guides without rebuilding or thresholding them."""
    if len(guides) != 4:
        raise ValueError('four actual native1024 subdivision guides required')
    payload = dict(policy=POLICY, shape_coords=shape.coords.detach().cpu().clone(),
                   shape_feats=shape.feats.detach().cpu().clone(), guides=[])
    for guide in guides:
        if guide.coords.ndim != 2 or guide.coords.shape[1] != 4 or guide.feats.shape != (len(guide.coords), 8):
            raise ValueError('native ordered parent coordinates and eight subdivision logits required')
        scale = [Fraction(x) for x in guide._scale]
        if len(scale) != 3 or any(x <= 0 for x in scale):
            raise ValueError('three exact positive rational sparse scales required')
        payload['guides'].append(dict(coords=guide.coords.detach().cpu().clone(),
            feats=guide.feats.detach().cpu().clone(), scale=[[x.numerator, x.denominator] for x in scale]))
    payload['tensor_SHA256'] = _hashes(payload)
    with Path(path).open('xb') as stream:
        torch.save(payload, stream)
    return payload['tensor_SHA256']


def _hashes(payload):
    return dict(shape_coords=tensor_sha(payload['shape_coords']), shape_feats=tensor_sha(payload['shape_feats']),
                guides=[dict(coords=tensor_sha(g['coords']), feats=tensor_sha(g['feats']), scale=g['scale'])
                        for g in payload['guides']])


def load_guides(path, shape, sparse_type):
    """Restore original guide values/order/scale for the exact generated latent."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['policy'] != POLICY or len(payload['guides']) != 4 or payload['tensor_SHA256'] != _hashes(payload):
        raise ValueError('unchanged actual captured native guides required')
    if (tensor_sha(shape.coords) != payload['tensor_SHA256']['shape_coords']
            or tensor_sha(shape.feats) != payload['tensor_SHA256']['shape_feats']):
        raise ValueError('exact original generated Shape latent values and row order required')
    return [sparse_type(coords=g['coords'].to(shape.coords.device), feats=g['feats'].to(shape.feats.device),
                        scale=tuple(Fraction(a, b) for a, b in g['scale'])) for g in payload['guides']]
