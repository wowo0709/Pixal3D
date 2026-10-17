"""Original controlled RGB cue generator; training/evaluation values only."""
from __future__ import annotations
import numpy as np

def procedural_surface_cue(
    vertices: np.ndarray,
    *,
    active_vertices: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Attach a deterministic high-contrast RGB identity only to a surface set.

    The construction uses source-space positions and a held-out seed.  It is a
    *value* generator: callers must not feed its output into a matcher
    descriptor.  CAGE same-index topology then transports this identity to the
    target surface without exposing RGB to WHERE.
    """
    points = np.asarray(vertices, dtype=np.float32)
    active = np.asarray(active_vertices, dtype=bool)
    if points.ndim != 2 or points.shape[1:] != (3,) or active.shape != (len(points),):
        raise ValueError("procedural cue arrays differ")
    generator = np.random.default_rng(int(seed))
    palette = generator.uniform(.08, 1.0, size=(8, 3)).astype(np.float32)
    # Project on a seed-specific cue-free coordinate direction, then quantize.
    # This creates broad, visibly different surface regions without inserting
    # identity labels into the descriptor used for correspondence.
    direction = generator.normal(size=(3,)).astype(np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    phase = float(generator.uniform(-.5, .5))
    code = np.floor((points @ direction + phase + 1.5) * 3.0).astype(np.int64) % len(palette)
    output = np.zeros((len(points), 3), dtype=np.float32)
    output[active] = palette[code[active]]
    return output
