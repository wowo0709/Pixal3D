"""Exact historical G2 condition, native scoring and inference-noise math."""
from __future__ import annotations
from hashlib import sha256
import numpy as np
import torch
from torch.nn import functional as F
from .historical_reproduction import locked_endpoint_loss
CAMPAIGN = "SS-TTO-ADAPTER0"
from functools import lru_cache


def _condition(tree: dict, device) -> dict:
    return {
        branch: {name: tensor.to(device) for name, tensor in values.items()}
        for branch, values in tree.items()
    }

def _sample_native(*, sampler, flow, noise, condition: dict, params: dict):
    sequence = np.linspace(1.0, 0.0, int(params["steps"]) + 1)
    rescale_t = float(params.get("rescale_t", 1.0))
    sequence = rescale_t * sequence / (1.0 + (rescale_t - 1.0) * sequence)
    inference = dict(params)
    inference.pop("steps")
    inference.pop("rescale_t", None)
    inference.pop("verbose", None)
    sample = noise.clone()
    for time_value, previous in zip(sequence[:-1], sequence[1:]):
        velocity = sampler._inference_model(
            flow, sample, float(time_value), **condition, **inference
        ).float()
        sample = sample - float(time_value - previous) * velocity
    return sample

def _decode_logits(decoder, latent):
    logits = decoder(latent)
    if tuple(logits.shape) != (1, 1, 64, 64, 64):
        raise RuntimeError(f"SS decoder did not return raw native logits: {tuple(logits.shape)}")
    return logits

def _noise(pair_id: str, *, shape: tuple[int, ...], device):
    """Reproduce the immutable base SS experiment's pair-specific noise."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(
        int.from_bytes(sha256(pair_id.encode()).digest()[:4], "big")
    )
    return torch.randn(shape, generator=generator).to(device)

def _seeded_noise(pair_id: str, seed_index: int, *, shape: tuple[int, ...], device):
    import torch

    if int(seed_index) == 0:
        return _noise(pair_id, shape=shape, device=device)
    payload = f"{CAMPAIGN}|{pair_id}|{int(seed_index)}".encode()
    seed = int.from_bytes(sha256(payload).digest()[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator).to(device)

def _f1(pred, target):
    tp = int((pred & target).sum())
    denominator = int(pred.sum()) + int(target.sum())
    return 2 * tp / denominator if denominator else 1.0

def _score(pipeline, condition, noise, regions, params, targets):
    with torch.inference_mode():
        flow = pipeline.models["sparse_structure_flow_model"]
        latent = _sample_native(sampler=pipeline.sparse_structure_sampler, flow=flow,
                                noise=noise.clone(), condition=condition, params=params)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits64 = _decode_logits(pipeline.models["sparse_structure_decoder"], latent).float()
        support64 = logits64 > 0
        support32 = F.max_pool3d(logits64, 2, 2) > 0
        loss, terms = locked_endpoint_loss(logits64, regions)
    metrics = {"regional_bce": float(loss),
               "regional_bce_terms": {
                   str(resolution): {name: float(value) for name, value in terms[resolution].items()}
                   for resolution in (64, 32)
               }}
    for resolution, support in ((64, support64), (32, support32)):
        masks = regions[resolution]
        positive = support[masks["visible_positive"]].float().mean()
        reject = (~support[masks["visible_negative"]]).float().mean()
        metrics[f"source_recall{resolution}"] = float(support[masks["source_positive"]].float().mean())
        metrics[f"stale_leak{resolution}"] = float(support[masks["stale"]].float().mean())
        metrics[f"visible_state_acc{resolution}"] = float((positive + reject) / 2)
        metrics[f"global_f1_{resolution}"] = _f1(support, targets[resolution])
    return metrics

def adapted_condition(pair, adapter, geometry, mode, device, *, source_off=False):
    clean = _condition(pair["clean_target_condition"], device)
    mismatch = _condition(pair["native_mismatch_condition"], device)
    source_proj = clean["cond"]["proj"] if source_off else mismatch["cond"]["proj"]
    projected, trace = adapter(clean["cond"]["proj"], source_proj,
                               geometry.to(device), mode=mode)
    condition = {"cond": {"global": mismatch["cond"]["global"], "proj": projected},
                 "neg_cond": mismatch["neg_cond"]}
    if not torch.equal(condition["cond"]["global"], mismatch["cond"]["global"]):
        raise RuntimeError("factorial adapter changed frozen global condition")
    return condition, clean, mismatch, trace

def _seed(*parts):
    value = sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(value[:8], "big") % (2**63 - 1)

@lru_cache(maxsize=48)
def geometry_derangement(pair_id: str) -> torch.Tensor:
    seed = int.from_bytes(sha256(f"{pair_id}|geometry-tuple-derangement".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    source = np.arange(4096)
    for _ in range(20):
        order = rng.permutation(4096)
        if not np.any(order == source):
            return torch.from_numpy(order.copy()).long()
    # The deterministic nonzero cyclic shift preserves every geometry tuple
    # exactly even if random derangement rejection was unexpectedly unlucky.
    return torch.from_numpy(np.roll(source, 2048).copy()).long()
