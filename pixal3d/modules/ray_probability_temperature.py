"""Calibrate pooled correspondence confidence while preserving its ranking.

Native Q/K and utility logits are unchanged. Temperature acts after pooling
the four probability heads, on the weights consumed by geometry retargeting.
This is neither a uniform-score nor a matched-Top1 feature ablation.
"""
import math

import torch
from torch.nn import functional as F


def matching_logits(head, native, context, keys, values, valid, prior):
    m, k = valid.shape
    if (native.shape != (m, 1024) or context.shape != native.shape
            or keys.shape != (m, k, 1024) or values.shape != (m, k, head.value_width)
            or prior.shape != valid.shape or valid.dtype != torch.bool):
        raise ValueError('aligned inference ray/candidate observations required')
    keys = torch.where(valid[..., None], keys, torch.zeros_like(keys))
    values = torch.where(valid[..., None], values, torch.zeros_like(values))
    score = head.score_selector(native, context, values[..., :1024], valid, prior)[2]
    query_input = torch.cat([F.layer_norm(native, (1024,)),
                             F.layer_norm(context, (1024,))], -1)
    q = head.query(query_input).reshape(m, head.heads, head.head_width)
    kk = head.key(F.layer_norm(keys, (1024,))).reshape(
        m, k, head.heads, head.head_width).transpose(1, 2)
    with torch.autocast(device_type=native.device.type, enabled=False):
        logits = (q.float()[:, :, None] * kk.float()).sum(-1) / math.sqrt(head.head_width)
        return (logits + score[:, None].float()).masked_fill(~valid[:, None], -torch.inf)


def native_probability(logits, valid):
    if logits.ndim != 3 or valid.shape != (logits.shape[0], logits.shape[2]) or valid.dtype != torch.bool:
        raise ValueError('per-head logits and candidate mask required')
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float().masked_fill(~valid[:, None], -torch.inf)
        maximum = logits.amax(-1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        probability = torch.exp(logits-maximum).masked_fill(~valid[:, None], 0.)
        return probability/probability.sum(-1, keepdim=True).clamp_min(1e-8)


def pooled_temperature(logits, valid, temperature):
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('finite positive fixed temperature required')
    probability = native_probability(logits, valid)
    if temperature == 1.:
        return probability.mean(1)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        # Pool in log space so valid low probabilities can recover when T>1,
        # including probabilities that underflow at the original temperature.
        active = valid.any(-1)
        masked = logits.float().masked_fill(~valid[:, None], -torch.inf)
        safe = torch.where(active[:, None, None], masked, torch.zeros_like(masked))
        logp = safe-torch.logsumexp(safe, -1, keepdim=True)
        pooled = torch.logsumexp(logp, 1)-math.log(logits.shape[1])
        scaled = (pooled/temperature).masked_fill(~valid, -torch.inf)
        maximum = scaled.amax(-1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        output = torch.exp(scaled-maximum).masked_fill(~valid, 0.)
        return output/output.sum(-1, keepdim=True).clamp_min(1e-8)
