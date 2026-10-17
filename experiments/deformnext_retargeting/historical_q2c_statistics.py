"""Exact historical pair/cue/seed aggregation for controlled Q2C reproduction."""
from collections import defaultdict
from math import ceil
import numpy as np
import torch
ERRORS = ("task_rgb_error", "wrong_rgb_error", "gt_rgb_error")
DIAGNOSTICS = ()
KEYS = ("backbone", "fold", "seed", "split", "pair_id", "arm", "original_cue_id", "query_slot")


def _finite_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty [{','.join(['?'] * ndim)}] array")
    return result

def pair_bootstrap_delta(pair_deltas: object, *, seed: int, resamples: int = 10_000) -> dict[str, object]:
    """Bootstrap a paired task-gain mean with pairs as the only replicas."""
    deltas = _finite_array(pair_deltas, name="pair_deltas", ndim=1)
    if not isinstance(seed, (int, np.integer)) or resamples < 1:
        raise ValueError("R8 bootstrap seed or resample count differs")
    generator = np.random.default_rng(int(seed))
    bootstrap = deltas[generator.integers(0, len(deltas), size=(int(resamples), len(deltas)))].mean(axis=1)
    return {
        "mean": float(deltas.mean()),
        "ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        "pair_count": len(deltas),
        "resamples": int(resamples),
    }

def _means(values: list[dict], metrics: list[str]) -> dict:
    return {metric: float(np.mean([row[metric] for row in values])) for metric in metrics}

def aggregate_rows(rows: list[dict]) -> dict:
    """query → original cue → model seed → pair → cross-pair.

    This API handles one backbone/split/fold-TRAIN population at a time. DEV
    may contain four disjoint OOF folds. G is formed once at pair/arm/seed.
    """
    if not rows: raise ValueError("R9 raw rows are required")
    if any("g_oracle" in row or "routing_ratio" in row for row in rows):
        raise ValueError("R9 cue-level ratio is forbidden")
    keys = [tuple(row.get(key) for key in KEYS) for row in rows]
    if len(keys) != len(set(keys)): raise ValueError("R9 duplicate raw key")
    if len({row["backbone"] for row in rows}) != 1 or len({row["split"] for row in rows}) != 1:
        raise ValueError("R9 population aggregation coverage differs")
    if rows[0]["split"] == "TRAIN" and len({row["fold"] for row in rows}) != 1:
        raise ValueError("R9 TRAIN folds must remain descriptive and separate")
    metrics = list(ERRORS) + [name for name in DIAGNOSTICS if all(name in row for row in rows)]
    if not np.isfinite([[row[name] for name in metrics] for row in rows]).all():
        raise ValueError("R9 raw components must be finite")
    groups = defaultdict(list)
    for row in rows: groups[(row["pair_id"], row["seed"], row["arm"], row["original_cue_id"])].append(row)
    cue = {key: _means(values, metrics) for key, values in groups.items()}
    pair_ids = sorted({row["pair_id"] for row in rows}); arms = sorted({row["arm"] for row in rows})
    for pair in pair_ids:
        eligible = {row["original_cue_id"] for row in rows if row["pair_id"] == pair}
        expected = {(pair, seed, arm, cue_id) for seed in (42017, 42018, 42019) for arm in arms for cue_id in eligible}
        if {key for key in cue if key[0] == pair} != expected:
            raise ValueError("R9 raw cue/seed/arm coverage differs")
        query_sets = [{(row["query_slot"], row["query_id"]) for row in values}
                      for key, values in groups.items() if key[0] == pair]
        if any(value != query_sets[0] for value in query_sets):
            raise ValueError("R9 repeated query-slot coverage differs")
        if len({row["fold"] for row in rows if row["pair_id"] == pair}) != 1:
            raise ValueError("R9 pair has more than one OOF fold")
    seed_groups = defaultdict(list)
    for key, value in cue.items(): seed_groups[key[:3]].append(value)
    seed = {}
    for key, values in seed_groups.items():
        components = _means(values, metrics)
        components["g_oracle"] = ((components["wrong_rgb_error"] - components["task_rgb_error"])
                                   / (components["wrong_rgb_error"] - components["gt_rgb_error"] + 1e-8))
        seed[key] = components
    pair_groups = defaultdict(list)
    for (pair_id, _, arm), value in seed.items(): pair_groups[(pair_id, arm)].append(value)
    with_ratio = metrics + ["g_oracle"]
    pair = {key: _means(values, with_ratio) for key, values in pair_groups.items()}
    arm_metrics = {arm: _means([value for (_, one_arm), value in pair.items() if one_arm == arm], with_ratio) for arm in arms}
    for arm in arms:
        means = [float(np.mean([value["task_rgb_error"] for (_, one_seed, one_arm), value in seed.items()
                                if one_seed == model_seed and one_arm == arm])) for model_seed in (42017, 42018, 42019)]
        arm_metrics[arm]["task_rgb_error_seed_range"] = [min(means), max(means)]
    return {"pair_count": len(pair_ids), "pair_ids": pair_ids, "arms": arm_metrics,
            "bootstrap_unit": "pair", "aggregation_order": "query->cue->seed->pair->cross-pair",
            "pair": [{"pair_id": key[0], "arm": key[1], **value} for key, value in pair.items()],
            "seed": [{"pair_id": key[0], "seed": key[1], "arm": key[2], **value} for key, value in seed.items()],
            "cue": [{"pair_id": key[0], "seed": key[1], "arm": key[2], "original_cue_id": key[3], **value} for key, value in cue.items()]}

def contrast(aggregate: dict, baseline: str, candidate: str) -> dict:
    pair = {(row["pair_id"], row["arm"]): row for row in aggregate["pair"]}
    delta = [pair[(pair_id, baseline)]["task_rgb_error"] - pair[(pair_id, candidate)]["task_rgb_error"] for pair_id in aggregate["pair_ids"]]
    bootstrap = pair_bootstrap_delta(delta, seed=20260916, resamples=10000)
    seed = {(row["pair_id"], row["seed"], row["arm"]): row for row in aggregate["seed"]}
    seed_gains = [float(np.mean([seed[(pair_id, one_seed, baseline)]["task_rgb_error"] - seed[(pair_id, one_seed, candidate)]["task_rgb_error"]
                                for pair_id in aggregate["pair_ids"]])) for one_seed in (42017, 42018, 42019)]
    cue = {(row["pair_id"], row["seed"], row["arm"], row["original_cue_id"]): row for row in aggregate["cue"]}
    by_layout = defaultdict(lambda: defaultdict(list))
    for (pair_id, one_seed, arm, cue_id), value in cue.items():
        if arm == baseline:
            by_layout[cue_id][pair_id].append(value["task_rgb_error"] - cue[(pair_id, one_seed, candidate, cue_id)]["task_rgb_error"])
    layout = {str(key): float(np.mean([np.mean(v) for v in pair_values.values()])) for key, pair_values in by_layout.items()}
    worst = float(np.mean(sorted(layout.values())[:ceil(len(layout) / 4)]))
    recovery = []
    for pair_id in aggregate["pair_ids"]:
        base, changed = pair[(pair_id, baseline)], pair[(pair_id, candidate)]
        if "oracle_task_rgb_error" in base:
            denominator = base["diagnostic_task_rgb_error"] - base["oracle_task_rgb_error"]
            if denominator > 0: recovery.append((base["diagnostic_task_rgb_error"] - changed["diagnostic_task_rgb_error"]) / denominator)
    return {**bootstrap, "pair_gains": delta, "seed_gains": seed_gains, "worst_cue_gain": worst,
            "cue_layout_gains": layout, "oracle_headroom_recovery": None if not recovery else float(np.mean(recovery))}
