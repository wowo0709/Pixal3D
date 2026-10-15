"""Selected R9 scorer definitions, copied verbatim from the promoted implementation.

Authority: anchor-hybrid-prestudy/pixal3d/experiments/exg_app_cue_b_r9_ladder.py.
Only the selected P0-RAW/M0-L12/D1-Q2C tuple is exposed to training. The
original controlled RGB task and the new deformation endpoint remain distinct.
No runtime dependency on the historical worktree.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

AUTHORITY_SHA256 = 'e7be96c47803d16a769fcb9e9cf0026b314c4983c6b40f9d3c21604b15d973d2'
VERBATIM_AST_SHA256 = {'PROJECTOR_ARMS': '51241363ca00c5a434a0e4e153df5c22481e5528ce04cfba65375f7c50987a59', 'MULTI_ARMS': '12f981d16c8f50e4b615bcbd04d5f459f459cc1f712995aeab1623e3490fd9f3', 'DECODER_ARMS': 'eef7c8143d27597dd705a7e4fca825f1eca7c81a97d24d2536dc07ad977b0e76', 'OmegaProjector': 'c0bcfefcdd9af6efafb9cd614e0efdf8bb63495a24d66863170ec67aaa0be2f8', 'canonical_multi_arm': 'c7959a6184c13ce425746e4fd47f3495643dab7a052cc8271f91afd544794194', 'required_multi_layers': 'fb4a65525f7ae9b7cc417712d1d01a657cde29f19a84a8ffe1b4a328bcb5005f', 'multilayer_scores': 'da568b438232e92538d11387cc13bb843f97a2747289e5d85ad5610ae17f8926', '_ContextBlock': 'ba28ad3abc0acf1b884955f1eaf4df389ad47510a2d2ef699bf7b8a756005b88', 'RegionDecoder': '9455ae65c01ac671679af08793f842810199eb6fb824079afa460a37212c0916', '_ScalarHead': 'f227fb058803a365cf5d62631b1510255b8afccf001c4518854aadf236ba5361', 'R9Scorer': '0b3a96fbbbd6b3f9c27d2e828d1a472d7e9612e799e327857211c0b89731835a'}

PROJECTOR_ARMS = ("P0-RAW", "P1-PROJ128", "P2-RES1024", "P3-RES1024-PRESERVE")


MULTI_ARMS = ("M0-L12", "M1-EARLY", "M2-ALL", "M3-EARLY+LATE-DIAG", "M2-DROP24", "M2-DROP18-24")


DECODER_ARMS = ("D0-PAIRWISE", "D1-Q2C", "D2-BIDIR", "D3-BIDIR-2L")


class OmegaProjector(nn.Module):
    """Preregistered Q/K projection or shared gated residual on frozen L12."""

    def __init__(self, arm: str = "P0-RAW") -> None:
        super().__init__()
        if arm not in PROJECTOR_ARMS:
            raise ValueError("R9 projector arm differs")
        self.arm = arm
        self.output_dim = 128 if arm == "P1-PROJ128" else 1024
        if arm == "P1-PROJ128":
            self.query_projection = nn.Linear(1024, 128, bias=False)
            self.key_projection = nn.Linear(1024, 128, bias=False)
            nn.init.xavier_uniform_(self.query_projection.weight)
            nn.init.xavier_uniform_(self.key_projection.weight)
        elif arm in PROJECTOR_ARMS[2:]:
            self.delta = nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 1024))
            self.beta = nn.Parameter(torch.zeros(()))

    @property
    def alpha(self) -> torch.Tensor:
        return .25 * torch.tanh(self.beta)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape[-1] != 1024 or keys.shape[-1] != 1024:
            raise ValueError("R9 projector requires 1024-D L12")
        if self.arm == "P1-PROJ128":
            query, keys = self.query_projection(query), self.key_projection(keys)
        elif self.arm in PROJECTOR_ARMS[2:]:
            query, keys = query + self.alpha * self.delta(query), keys + self.alpha * self.delta(keys)
        return F.normalize(query, dim=-1), F.normalize(keys, dim=-1)


def canonical_multi_arm(arm: str) -> str:
    if arm not in MULTI_ARMS:
        raise ValueError("R9 multi-layer arm differs")
    return "M1-EARLY" if arm == "M2-DROP18-24" else arm


def required_multi_layers(arm: str) -> tuple[int, ...]:
    """Frozen descriptor layers that an arm actually consumes.

    M3 keeps the early mean as its proposal/base score but supplies the late
    mean to the scalar head, so it still requires all four descriptor layers.
    The DROP18-24 diagnostic is a replay alias of M1 and must not load or train
    an otherwise identical duplicate representation.
    """
    arm = canonical_multi_arm(arm)
    return {
        "M0-L12": (12,),
        "M1-EARLY": (5, 12),
        "M2-ALL": (5, 12, 18, 24),
        "M3-EARLY+LATE-DIAG": (5, 12, 18, 24),
        "M2-DROP24": (5, 12, 18),
    }[arm]


def multilayer_scores(arm: str, scores: dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed, unparameterized mean and scalar features in the specified order.

    Replacing scores[12] with projected L12 implements the adjudication 2x2.
    The other similarities always remain frozen RAW layer cosines.
    """
    arm = canonical_multi_arm(arm)
    layers = (5, 12) if arm == "M3-EARLY+LATE-DIAG" else required_multi_layers(arm)
    values = torch.stack([scores[layer] for layer in layers], dim=-1)
    base = values.mean(dim=-1)
    if arm == "M3-EARLY+LATE-DIAG":
        values = torch.stack((base, (scores[18] + scores[24]) / 2), dim=-1)
    return base, values


class _ContextBlock(nn.Module):
    def __init__(self, *, query_only: bool) -> None:
        super().__init__()
        self.query_only = query_only
        self.norm_attention = nn.LayerNorm(1024)
        self.query_projection = nn.Linear(1024, 128)
        self.key_projection = nn.Linear(1024, 128)
        self.value_projection = nn.Linear(1024, 128)
        self.attention_output = nn.Linear(128, 1024)
        self.attention_gate = nn.Parameter(torch.zeros(()))
        self.norm_ffn = nn.LayerNorm(1024)
        self.ffn = nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 1024))
        self.ffn_gate = nn.Parameter(torch.zeros(()))

    def forward(self, query: torch.Tensor, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = query[:, None] if self.query_only else torch.cat((query[:, None], candidates), dim=1)
        context = candidates if self.query_only else tokens
        normalized = self.norm_attention(tokens)
        normalized_context = self.norm_attention(context)
        def heads(values: torch.Tensor) -> torch.Tensor:
            return values.reshape(values.shape[0], values.shape[1], 4, 32).transpose(1, 2)
        q = heads(self.query_projection(normalized))
        k = heads(self.key_projection(normalized_context))
        v = heads(self.value_projection(normalized_context))
        attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        attention = attention.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[1], 128)
        tokens = tokens + self.attention_gate * self.attention_output(attention)
        tokens = tokens + self.ffn_gate * self.ffn(self.norm_ffn(tokens))
        return tokens[:, 0], candidates if self.query_only else tokens[:, 1:]


class RegionDecoder(nn.Module):
    """Contextualize 1024-D L12 descriptors, without candidate or value changes."""

    def __init__(self, arm: str = "D0-PAIRWISE") -> None:
        super().__init__()
        if arm not in DECODER_ARMS:
            raise ValueError("R9 decoder arm differs")
        depth = 0 if arm == "D0-PAIRWISE" else 2 if arm == "D3-BIDIR-2L" else 1
        self.blocks = nn.ModuleList([_ContextBlock(query_only=arm == "D1-Q2C") for _ in range(depth)])

    def forward(self, query: torch.Tensor, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 2 or query.shape[-1] != 1024 or candidates.shape != (len(query), 64, 1024):
            raise ValueError("R9 decoder requires a query and fixed 64 1024-D candidate tokens")
        for block in self.blocks:
            query, candidates = block(query, candidates)
        return query, candidates


class _ScalarHead(nn.Module):
    def __init__(self, descriptor_dim: int, scalar_count: int) -> None:
        super().__init__()
        width = 2 * descriptor_dim + scalar_count
        self.norm = nn.LayerNorm(width)
        self.hidden = nn.Linear(width, 64)
        self.output = nn.Linear(64, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, scalars: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        query = query[:, None].expand_as(keys)
        features = torch.cat((query * keys, (query - keys).abs(), scalars), dim=-1)
        residual = self.output(F.gelu(self.hidden(self.norm(features)))).squeeze(-1)
        return base + .25 * torch.tanh(residual)


class R9Scorer(nn.Module):
    """Unified cue-free scorer. IDs gather frozen descriptors; never become features.

    Exp4 ordering is context on 1024-D L12, then the adjudicated projector.
    This also permits the P1 128-D representation without an extra adapter.
    """
    context_order = "context_1024_then_adjudicated_projector"

    def __init__(self, *, projector_arm: str = "P0-RAW", multi_arm: str = "M0-L12",
                 decoder_arm: str = "D0-PAIRWISE") -> None:
        super().__init__()
        self.projector = OmegaProjector(projector_arm)
        self.multi_arm = canonical_multi_arm(multi_arm)
        self.decoder_arm = decoder_arm
        scalar_count = {"M0-L12": 1, "M1-EARLY": 2, "M2-ALL": 4,
                        "M3-EARLY+LATE-DIAG": 2, "M2-DROP24": 3}[self.multi_arm]
        # Head initialization remains independent of decoder depth.
        self.fusion_head = _ScalarHead(self.projector.output_dim, scalar_count)
        self.decoder = RegionDecoder(decoder_arm)

    def forward(self, query: dict[int, torch.Tensor], source: dict[int, torch.Tensor],
                candidates: torch.Tensor, *, raw_scores: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if (not isinstance(query, dict) or not isinstance(source, dict) or 12 not in query or set(query) != set(source)
                or not set(query) <= {5, 12, 18, 24}):
            raise ValueError("R9 descriptor-only model fields differ")
        if (candidates.ndim != 2 or candidates.shape != (len(query[12]), 64)
                or candidates.dtype not in (torch.int32, torch.int64) or candidates.numel() == 0
                or candidates.min() < 0 or candidates.max() >= len(source[12])
                or (candidates.sort(dim=-1).values[:, 1:] == candidates.sort(dim=-1).values[:, :-1]).any()):
            raise ValueError("R9 requires exactly 64 distinct fixed candidate IDs")
        for layer in query:
            if (query[layer].shape != (len(query[12]), 1024) or source[layer].shape != (len(source[12]), 1024)
                    or not torch.isfinite(query[layer]).all() or not torch.isfinite(source[layer]).all()):
                raise ValueError("R9 frozen descriptor shape or finite values differ")
        q12, k12 = query[12].detach(), source[12].detach()[candidates]
        raw = (F.normalize(q12, dim=-1)[:, None] * F.normalize(k12, dim=-1)).sum(-1)
        if raw_scores is not None and (raw_scores.shape != raw.shape or not torch.isfinite(raw_scores).all()):
            raise ValueError("R9 fixed RAW scores differ")
        contextual_q, contextual_k = self.decoder(q12, k12)
        # Canonical strides also preserve byte-exact zero-start through a
        # learned upstream projector (strided GEMM can round differently).
        q, k = self.projector(contextual_q.contiguous(), contextual_k.contiguous())
        adapted = (q[:, None] * k).sum(-1)
        # Preserve the exact stored RAW metric, including its float rounding.
        l12_score = adapted if raw_scores is None else raw_scores + (adapted - raw)
        layer_scores = {12: l12_score}
        for layer in query.keys() - {12}:
            layer_scores[layer] = (F.normalize(query[layer].detach(), dim=-1)[:, None]
                                   * F.normalize(source[layer].detach()[candidates], dim=-1)).sum(-1)
        base, scalars = multilayer_scores(self.multi_arm, layer_scores)
        scores = self.fusion_head(q, k, scalars, base)
        preservation = (adapted - raw).square().mean()
        return {"scores": scores, "candidate_indices": candidates, "raw_scores": raw,
                "representation_scores": adapted, "base_scores": base,
                "preservation_loss": preservation if self.projector.arm == "P3-RES1024-PRESERVE" else preservation * 0}


MATCHER_CONTRACT = {
    "architecture": "P0-RAW/M0-L12/D1-Q2C",
    "descriptor": "OMEGA_L12_FRAME_RAW_1024",
    "candidate_budget": 64,
    "authority_sha256": AUTHORITY_SHA256,
}
CHECKPOINT_SCHEMA = "deformnext-r9-d1-v5"


def frame_raw(descriptors: torch.Tensor) -> torch.Tensor:
    """Select FRAME from the verified aggregator [FRAME, GLOBAL] cache layout."""
    if descriptors.shape[-1] != 2048:
        raise ValueError("FRAME extraction requires the verified 2048-D FRAME+GLOBAL cache")
    return descriptors[..., :1024].contiguous()


@torch.no_grad()
def frame_topk(query, source, valid, topk=64):
    """Shared FP32 RAW proposal math for offline, training and inference."""
    if query.ndim != 3 or source.ndim != 4 or query.shape[-1] != 1024 or source.shape[-1] != 1024:
        raise ValueError('FRAME proposal requires [B,Q,1024] and [B,V,S,1024]')
    if valid.shape != source.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError('FRAME proposal validity shape differs')
    with torch.autocast(device_type=query.device.type, enabled=False):
        flat = source.flatten(1, 2).float()
        # Use the same GEMM shape independent of loader batch size/AMP.
        scores = torch.stack([F.normalize(q.float(), dim=-1) @ F.normalize(k, dim=-1).T
                              for q, k in zip(query, flat)])
        usable = valid.flatten(1)
        complete = usable.sum(-1) >= 64
        return scores.masked_fill(~usable[:, None] | ~complete[:, None, None], -torch.inf).topk(topk, dim=-1)


class PromotedR9Matcher(R9Scorer):
    """Fixed promoted tuple with bounded-memory batched candidate scoring."""

    def __init__(self):
        super().__init__(projector_arm="P0-RAW", multi_arm="M0-L12", decoder_arm="D1-Q2C")

    def score_candidates(self, query, keys, raw_scores, *, chunk_size=32):
        if query.shape[-1] != 1024 or keys.shape != (*query.shape[:-1], 64, 1024):
            raise ValueError("D1 requires FRAME 1024-D queries and 64 fixed candidates")
        if raw_scores.shape != keys.shape[:-1] or chunk_size < 1:
            raise ValueError("D1 RAW scores or query chunk size differ")
        shape = query.shape[:-1]
        q, k, raw = query.reshape(-1, 1024), keys.reshape(-1, 64, 1024), raw_scores.reshape(-1, 64)
        scores = []
        for start in range(0, len(q), chunk_size):
            qs, ks, rs = q[start:start+chunk_size], k[start:start+chunk_size], raw[start:start+chunk_size]
            # Unique local IDs select the supplied immutable candidate tokens.
            ids = torch.arange(ks.shape[0] * 64, device=ks.device).reshape(-1, 64)
            result = self({12: qs}, {12: ks.reshape(-1, 1024)}, ids, raw_scores=rs)
            scores.append(result["scores"])
        return torch.cat(scores).reshape(*shape, 64)

    def load_checkpoint(self, checkpoint):
        if (not isinstance(checkpoint, dict)
                or checkpoint.get("schema") != CHECKPOINT_SCHEMA
                or checkpoint.get("matcher_contract") != MATCHER_CONTRACT):
            raise ValueError("checkpoint is not the verified R9 D1 FRAME contract; legacy pairwise weights are incompatible")
        if checkpoint.get("training_objective") not in {"topology-visible-listwise-v1", "controlled-rgb-r9"}:
            raise ValueError("R9 checkpoint requires explicit, audited supervision provenance")
        self.load_state_dict(checkpoint["q2c"], strict=True)
