"""Original G2 pair-bootstrap; model/noise seeds are reduced inside pairs."""
import numpy as np

def _ci(values, pair_ids):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(pair_ids),) or not np.isfinite(values).all():
        raise RuntimeError("pair-level statistic missing or nonfinite")
    rng = np.random.default_rng(20260918)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    samples = values[draws].mean(axis=1)
    return {"mean": float(values.mean()),
            "ci95": [float(x) for x in np.quantile(samples, (.025, .975))],
            "pair_values": dict(zip(pair_ids, values.tolist()))}
