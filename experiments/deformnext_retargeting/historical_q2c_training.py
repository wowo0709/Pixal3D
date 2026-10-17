"""Pinned R9 RGB-task training math; I/O and current SS experiments stay separate."""
from __future__ import annotations
import copy
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
import numpy as np
import torch
from pixal3d.modules.r9_matching import R9Scorer, PROJECTOR_ARMS, MULTI_ARMS, DECODER_ARMS
ROOT = Path(__file__).resolve().parents[2]
UPDATES = 1200
QUERY_COUNT = 256
SEEDS = (42017, 42018, 42019)
_FORBIDDEN = re.compile(r"evaluator|(^|[_/.-])(dev|eval|oracle)([_/.-]|$)", re.I)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()

def _firewall(value: object) -> None:
    """Reject semantic fields/values; reference authorization belongs to lineage."""
    if isinstance(value, dict):
        # Frozen cache-bank names (including R6 descriptor_cache/dev) are not
        # TRAIN/DEV semantics. Exact scalar references remain opaque here;
        # consumers must validate schema, bytes and parent authority before use.
        if (set(value) == {"path", "sha256"} and isinstance(value["path"], (str, Path))
                and isinstance(value["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"])):
            return
        for key, item in value.items():
            if isinstance(key, str) and _FORBIDDEN.search(key):
                raise ValueError("R9 evaluator/DEV/oracle field is forbidden in training")
            _firewall(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _firewall(item)
    elif isinstance(value, (str, Path)) and _FORBIDDEN.search(str(value)):
        # A repository directory named /root/dev is not a DEV evidence path.
        path = str(value).replace(str(ROOT), "REPOSITORY")
        if _FORBIDDEN.search(path):
            raise ValueError("R9 evaluator/DEV/oracle path is forbidden in training")

@dataclass(frozen=True)
class TrainingTuple:
    wave: str = "exp1"
    arm: str = "REG-GLOBAL64"
    region: str = "REG-GLOBAL64"
    projector_arm: str = "P0-RAW"
    multi_arm: str = "M0-L12"
    decoder_arm: str = "D0-PAIRWISE"
    fold: int = 0
    seed: int = 42017
    backbone: str = "OMEGA_L12_FRAME_RAW"
    train_size: str = "all"

    def __post_init__(self) -> None:
        if (self.wave not in ("exp1", "exp2", "exp2_scaling", "exp2_proposal", "exp3", "adjudication", "exp4")
                or self.region not in ("REG-GLOBAL64", "REG-LOCAL1", "REG-MULTI64")
                or self.projector_arm not in PROJECTOR_ARMS or self.multi_arm not in MULTI_ARMS
                or self.decoder_arm not in DECODER_ARMS or self.seed not in SEEDS
                or type(self.fold) is not int or self.fold not in range(4)
                or self.backbone not in ("OMEGA_L12_FRAME_RAW", "DINOV3_L24_RAW")
                or self.train_size not in ("16", "32", "all")):
            raise ValueError("R9 immutable training tuple differs")
        if self.multi_arm == "M2-DROP18-24":
            raise ValueError("R9 DROP18-24 is an M1 alias, never a duplicate training run")
        if self.wave == "exp1":
            valid = self.arm == self.region and (self.projector_arm, self.multi_arm, self.decoder_arm) == ("P0-RAW", "M0-L12", "D0-PAIRWISE")
        elif self.wave in ("exp2", "exp2_scaling", "exp2_proposal"):
            valid = (self.arm == self.projector_arm or self.wave == "exp2_proposal" and self.arm == "P*-PROJECTED-PROPOSAL") and self.multi_arm == "M0-L12" and self.decoder_arm == "D0-PAIRWISE"
            if self.wave == "exp2_scaling":
                valid = valid and self.projector_arm == "P2-RES1024"
            if self.wave == "exp2_proposal":
                valid = valid and self.arm == "P*-PROJECTED-PROPOSAL" and self.projector_arm != "P0-RAW"
        elif self.wave == "exp3":
            valid = self.arm == self.multi_arm and self.projector_arm == "P0-RAW" and self.decoder_arm == "D0-PAIRWISE"
        elif self.wave == "exp4":
            valid = self.arm == self.decoder_arm
        else:
            valid = self.arm == "2X2" and self.decoder_arm == "D0-PAIRWISE"
        if (not valid or self.backbone == "DINOV3_L24_RAW" and self.wave != "exp1"
                or self.wave != "exp2_scaling" and self.train_size != "all"
                or self.wave != "exp1" and self.region == "REG-LOCAL1"):
            raise ValueError("R9 wave/arm/representation tuple differs")

@dataclass(frozen=True)
class TrainingBatch:
    pair_id: str
    query: dict[int, torch.Tensor]
    source: dict[int, torch.Tensor]
    candidates: torch.Tensor
    label_rgb: torch.Tensor
    target_rgb: torch.Tensor
    raw_scores: torch.Tensor | None
    sampled_query_sha256: str
    replacement: bool

def _normalize_records(records: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    _firewall(records)
    if not isinstance(records, dict) or not records:
        raise ValueError("R9 TRAIN records are required")
    normalized = {}
    required = {"query", "source", "candidates", "label_rgb", "query_vertex_indices"}
    for pair_id, record in records.items():
        if not isinstance(pair_id, str) or not isinstance(record, dict) or not required <= record.keys() or record.keys() - required - {"raw_scores", "target_rgb"}:
            raise ValueError("R9 TRAIN record fields differ")
        query, source = record["query"], record["source"]
        if (not isinstance(query, dict) or not isinstance(source, dict) or set(query) != set(source)
                or 12 not in query or not set(query) <= {5, 12, 18, 24}):
            raise ValueError("R9 descriptor fields differ")
        def array(value: object) -> np.ndarray:
            result = np.array(value, dtype=np.float32, copy=True)
            if not np.isfinite(result).all():
                raise ValueError("R9 nonfinite TRAIN record")
            result.setflags(write=False)
            return result
        q = {layer: array(value) for layer, value in query.items()}
        s = {layer: array(value) for layer, value in source.items()}
        nq, ns = len(q[12]), len(s[12])
        ids = np.asarray(record["candidates"])
        if (ids.dtype.kind not in "iu" or ids.shape != (nq, 64) or nq == 0 or ns < 64
                or ids.min() < 0 or ids.max() >= ns or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0)
                or any(value.shape != (nq, 1024) for value in q.values())
                or any(value.shape != (ns, 1024) for value in s.values())):
            raise ValueError("R9 fixed 64 candidates or descriptor shape differs")
        ids = np.array(ids, dtype=np.int64, copy=True); ids.setflags(write=False)
        label = array(record["label_rgb"])
        query_ids = np.asarray(record["query_vertex_indices"])
        if (query_ids.dtype.kind not in "iu" or query_ids.shape != (nq,) or query_ids.min() < 0 or query_ids.max() >= ns
                or label.shape != (16, ns, 3) or label.min() < 0 or label.max() > 1):
            raise ValueError("R9 label RGB shape/range differs")
        query_ids = np.array(query_ids, dtype=np.int64, copy=True); query_ids.setflags(write=False)
        if "target_rgb" in record and not np.array_equal(np.asarray(record["target_rgb"]), label[:, query_ids]):
            raise ValueError("R9 target RGB differs from original label at authenticated query indices")
        normalized[pair_id] = {"query": q, "source": s, "candidates": ids, "label_rgb": label, "query_vertex_indices": query_ids}
        if "raw_scores" in record:
            raw = array(record["raw_scores"])
            if raw.shape != ids.shape:
                raise ValueError("R9 RAW score shape differs")
            normalized[pair_id]["raw_scores"] = raw
    return normalized

def _record_hashes(records: dict[str, dict[str, object]]) -> dict[str, str]:
    def update(digest, item):
        if isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode()); update(digest, item[key])
        else:
            digest.update(str(item.dtype).encode()); digest.update(_json(item.shape)); digest.update(item.tobytes())
    result = {}
    for pair_id, record in records.items():
        digest = sha256(); update(digest, record); result[pair_id] = digest.hexdigest()
    return result

def _parameter_hash(state: dict[str, torch.Tensor]) -> str:
    digest = sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(_json(array.shape)); digest.update(array.tobytes())
    return digest.hexdigest()

def fuse_label_values(scores: torch.Tensor, candidate_rgb: torch.Tensor, *, temperature: float) -> torch.Tensor:
    """Original RGB values are read only after scores, solely for simplex fusion."""
    if (not math.isfinite(temperature) or temperature <= 0 or scores.ndim != 2 or scores.shape[1] != 64
            or candidate_rgb.ndim != 4 or candidate_rgb.shape[1:] != (*scores.shape, 3)
            or not torch.isfinite(scores).all() or not torch.isfinite(candidate_rgb).all()):
        raise ValueError("R9 label fusion values or temperature differ")
    return torch.einsum("qk,cqkd->cqd", torch.softmax(scores / temperature, dim=-1), candidate_rgb)

def wrong_indices(visible: np.ndarray, *, cue_seed: int, lock_seed: int) -> np.ndarray:
    """Exact R8 visibility matched cyclic derangement, now within the R9 source lock."""
    mask = np.asarray(visible, bool); output = np.arange(len(mask), dtype=np.int64)
    generator = np.random.default_rng(int(lock_seed) ^ int(cue_seed) ^ 0xC0FFEE)
    for group in (np.flatnonzero(mask), np.flatnonzero(~mask)):
        if len(group) > 1:
            output[group] = np.roll(group, int(generator.integers(1, len(group))))
    return output

def component_rows(*, metadata: dict, scores: np.ndarray, candidates: np.ndarray, query_ids: np.ndarray,
                   cue_rgb: np.ndarray, eligible: np.ndarray, wrong_maps: np.ndarray, temperature: float,
                   diagnostics: dict | None = None) -> list[dict]:
    """Reduce query slots inside each original cue; serialize component means only."""
    if (scores.shape != candidates.shape or candidates.shape != (len(query_ids), 64)
            or cue_rgb.ndim != 3 or cue_rgb.shape[-1] != 3 or eligible.shape != (len(cue_rgb),)
            or eligible.dtype != np.bool_ or not eligible.any() or wrong_maps.shape != (int(eligible.sum()), cue_rgb.shape[1])
            or not np.isfinite(scores).all() or not np.isfinite(cue_rgb).all() or not np.isfinite(temperature) or temperature <= 0
            or candidates.min() < 0 or candidates.max() >= cue_rgb.shape[1]
            or query_ids.min() < 0 or query_ids.max() >= cue_rgb.shape[1]
            or (np.diff(np.sort(candidates, axis=1), axis=1) == 0).any()):
        raise ValueError("R9 component input/eligible/K64 authority differs")
    weights = torch.softmax(torch.as_tensor(scores, dtype=torch.float64) / temperature, -1).numpy()
    result = []
    for active, cue_id in enumerate(np.flatnonzero(eligible)):
        values = cue_rgb[cue_id]; target = values[query_ids]
        prediction = np.einsum("qk,qkd->qd", weights, values[candidates])
        result.append({**metadata, "original_cue_id": int(cue_id), "query_slot": 0, "query_id": -1,
                       "query_count": len(query_ids), "unique_query_count": len(np.unique(query_ids)),
                       "task_rgb_error": float(np.abs(prediction - target).mean()),
                       "wrong_rgb_error": float(np.abs(values[wrong_maps[active, query_ids]] - target).mean()),
                       "gt_rgb_error": 0., **({} if diagnostics is None else diagnostics[int(cue_id)])})
    return result

class R9TrainingRunner:
    """Exact historical initializer, pair sampler and RGB optimizer step."""

    def __init__(self, records, *, temperature: float, device: str, tuple_info: TrainingTuple, authority=None,
                 upstream_state: dict[str, torch.Tensor] | None = None):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("R9 fixed TRAIN temperature differs")
        if (tuple_info.wave == "exp4") != (upstream_state is not None):
            raise ValueError("R9 Exp4 requires the exact frozen upstream representation state")
        _firewall(authority)
        self.records = records
        self.tuple_info = tuple_info
        self.updates = UPDATES
        self.temperature = float(temperature)
        self.device = torch.device(device)
        self.authority = copy.deepcopy(authority or {})
        self.authority.update({"tuple": asdict(tuple_info), "temperature": self.temperature,
                               "updates": UPDATES, "queries_per_update": QUERY_COUNT,
                               "record_sha256": _record_hashes(records), "device": str(self.device),
                               "context_order": R9Scorer.context_order,
                               "optimizer_config": {"name": "AdamW", "fusion_lr": .001, "added_lr": .0001, "weight_decay": .0001}})
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.rng = np.random.default_rng(tuple_info.seed)
        random.seed(tuple_info.seed); torch.manual_seed(tuple_info.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(tuple_info.seed)
        self.model = R9Scorer(projector_arm=tuple_info.projector_arm, multi_arm=tuple_info.multi_arm,
                              decoder_arm=tuple_info.decoder_arm).to(self.device)
        self.upstream_projector_sha256 = None
        if upstream_state is not None:
            projector_state = {key.removeprefix("projector."): value for key, value in upstream_state.items() if key.startswith("projector.")}
            self.model.projector.load_state_dict(projector_state, strict=True)
            self.model.projector.requires_grad_(False)
            self.upstream_projector_sha256 = _parameter_hash(self.model.projector.state_dict())
            self.authority["upstream_projector_sha256"] = self.upstream_projector_sha256
            self.authority["exp4_representation_policy"] = "frozen_winner_projector_fresh_identical_scalar_head"
        self.authority_sha256 = sha256(_json(self.authority)).hexdigest()
        groups = [{"params": list(self.model.fusion_head.parameters()), "lr": .001}]
        added = [parameter for parameter in (*self.model.projector.parameters(), *self.model.decoder.parameters()) if parameter.requires_grad]
        if added:
            groups.append({"params": added, "lr": .0001})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=.0001)
        self.completed_updates = 0
        self.logs: list[dict[str, object]] = []
        self.zero_start_proof: dict[str, object] | None = None

    def next_batch(self, *, query_count: int = QUERY_COUNT) -> TrainingBatch:
        if query_count != QUERY_COUNT:
            raise ValueError("R9 requires exactly 256 queries per update")
        pair_id = sorted(self.records)[int(self.rng.integers(len(self.records)))]
        record = self.records[pair_id]
        count = len(record["query"][12]); replacement = count < QUERY_COUNT
        rows = self.rng.choice(count, size=QUERY_COUNT, replace=replacement)
        ids = record["candidates"][rows]
        def tensor(value):
            return torch.tensor(value, device=self.device)
        return TrainingBatch(pair_id,
            {layer: tensor(values[rows]) for layer, values in record["query"].items()},
            {layer: tensor(values) for layer, values in record["source"].items()}, tensor(ids),
            tensor(record["label_rgb"][:, ids]), tensor(record["label_rgb"][:, record["query_vertex_indices"][rows]]),
            tensor(record["raw_scores"][rows]) if "raw_scores" in record else None,
            sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest(), replacement)

    def _forward(self, batch: TrainingBatch):
        return self.model(batch.query, batch.source, batch.candidates, raw_scores=batch.raw_scores)

    def _check_zero(self, batch: TrainingBatch, result: dict[str, torch.Tensor]) -> None:
        baseline = copy.deepcopy(self.model)
        baseline.decoder.blocks = torch.nn.ModuleList()
        with torch.no_grad():
            expected = baseline(batch.query, batch.source, batch.candidates, raw_scores=batch.raw_scores)
        score_equal = torch.equal(result["scores"], expected["scores"])
        residual_equal = torch.equal(result["scores"], result["base_scores"])
        raw_equal = torch.equal(result["representation_scores"], result["raw_scores"])
        raw_required = self.tuple_info.projector_arm != "P1-PROJ128" and self.tuple_info.wave != "exp4"
        if not score_equal or not residual_equal or raw_required and not raw_equal:
            raise RuntimeError("R9 zero-start score proof failed")
        self.zero_start_proof = {"decoder_equals_d0": score_equal, "head_equals_base": residual_equal,
                                 "projector_equals_raw": raw_equal, "raw_equality_required": raw_required,
                                 "sampled_query_sha256": batch.sampled_query_sha256,
                                 "score_sha256": sha256(result["scores"].detach().cpu().numpy().tobytes()).hexdigest()}

    def step(self) -> dict[str, object]:
        if self.completed_updates >= UPDATES:
            raise RuntimeError("R9 fixed 1200-update budget exhausted")
        batch = self.next_batch()
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        result = self._forward(batch)
        if self.completed_updates == 0:
            self._check_zero(batch, result)
        fused = fuse_label_values(result["scores"], batch.label_rgb, temperature=self.temperature)
        task_loss = (fused - batch.target_rgb).abs().mean()
        preserve_loss = result["preservation_loss"]
        loss = task_loss + .05 * preserve_loss
        if not torch.isfinite(loss):
            raise RuntimeError("R9 nonfinite TRAIN loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in self.model.parameters()):
            raise RuntimeError("R9 nonfinite TRAIN gradient")
        self.optimizer.step(); self.completed_updates += 1
        row = {"update": self.completed_updates, "pair_id": batch.pair_id,
               "sampled_query_sha256": batch.sampled_query_sha256, "replacement": batch.replacement,
               "query_count": QUERY_COUNT, "candidate_count": 64, "label_cue_count": 16,
               "loss": float(loss.detach()), "task_loss": float(task_loss.detach()), "preservation_loss": float(preserve_loss.detach())}
        self.logs.append(row)
        return row
