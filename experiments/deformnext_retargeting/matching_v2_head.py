"""Variable-budget R9-style scorer with separate usefulness/visibility outputs.

This is a new version: original R9's 64-candidate contract is left unchanged.
NULL is represented by p_use; it never turns a proposal miss into invisibility.
"""
import torch
from torch import nn
from torch.nn import functional as F

from pixal3d.modules.r9_matching import _ContextBlock, _ScalarHead

SCHEMA = 'deformnext-matching-v2-head-v1'


class MatchingV2Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = _ContextBlock(query_only=True)
        self.scorer = _ScalarHead(1024, 1)
        # Selected-match correctness, physical visibility, candidate hit are
        # distinct labels. Candidate hit alone is not a trustworthy p_use.
        self.confidence = nn.Sequential(nn.LayerNorm(1028), nn.Linear(1028, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, query, candidates, cosine, valid):
        if query.ndim != 2 or candidates.shape[:2] != cosine.shape or valid.shape != cosine.shape:
            raise ValueError('matching-v2 query/candidate shapes differ')
        if candidates.shape[0] != len(query) or query.shape[-1] != 1024 or candidates.shape[-1] != 1024:
            raise ValueError('matching-v2 requires FRAME 1024-D descriptors')
        block = self.context
        def heads(value):
            return value.reshape(value.shape[0], value.shape[1], 4, 32).transpose(1, 2)
        qn, kn = block.norm_attention(query[:, None]), block.norm_attention(candidates)
        attention = F.scaled_dot_product_attention(
            heads(block.query_projection(qn)), heads(block.key_projection(kn)), heads(block.value_projection(kn)),
            attn_mask=valid[:, None, None], dropout_p=0.)
        attended = attention.transpose(1, 2).reshape(len(query), 1, 128)
        contextual = query[:, None] + block.attention_gate * block.attention_output(attended)
        contextual = contextual + block.ffn_gate * block.ffn(block.norm_ffn(contextual))
        q, k = F.normalize(contextual[:, 0], dim=-1), F.normalize(candidates, dim=-1)
        raw = (F.normalize(query, dim=-1)[:, None] * k).sum(-1)
        adapted = (q[:, None] * k).sum(-1)
        base = cosine.nan_to_num(neginf=0.) + adapted - raw
        logits = self.scorer(q, k, base[..., None], base).masked_fill(~valid, -torch.inf)
        # Confidence statistics use a fixed scale, independent of later feature
        # aggregation temperature. Ranking gradients come only from ranking/task losses.
        safe = logits.detach().masked_fill(~valid, -1e4)
        probability = (safe / .05).softmax(-1) * valid
        maximum = safe.max(-1).values
        top = safe.topk(min(2, safe.shape[-1]), -1).values
        margin = top[:, 0] - top[:, -1]
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
        summary = torch.stack((maximum, margin.clamp(max=2.), entropy, valid.float().mean(-1)), -1)
        confidence = self.confidence(torch.cat((q.detach(), summary), -1))
        return logits, confidence


def masked_probabilities(logits, valid, temperature):
    maximum = logits.masked_fill(~valid, -torch.inf).max(-1, keepdim=True).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    weights = torch.exp((logits - maximum) / temperature).masked_fill(~valid, 0.)
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)


def ranking_loss(logits, positive, eligible):
    if not eligible.any():
        return logits.masked_fill(~torch.isfinite(logits), 0.).sum() * 0
    selected = logits[eligible]
    return (selected.logsumexp(-1) - selected.masked_fill(~positive[eligible], -torch.inf).logsumexp(-1)).mean()
