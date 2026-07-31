import numpy as np

try:
    from data_toolkit.utils import sphere_hammersley_sequence
except ModuleNotFoundError as error:
    if error.name != "data_toolkit":
        raise
    from utils import sphere_hammersley_sequence

from .config import RenderConfig
from .registry import camera_seed


def build_condition_views(
    asset_sha256: str, config: RenderConfig
) -> list[dict[str, float]]:
    rng = np.random.Generator(
        np.random.PCG64(camera_seed(asset_sha256, config.camera_policy))
    )
    offset = tuple(float(value) for value in rng.random(2))
    fov_min = np.deg2rad(config.fov_min_degrees)
    fov_max = np.deg2rad(config.fov_max_degrees)
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 2)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 2)
    radii = 1 / np.sqrt(
        rng.uniform(
            1 / radius_max**2,
            1 / radius_min**2,
            config.num_views,
        )
    )

    result = []
    for index, radius in enumerate(radii):
        yaw, pitch = sphere_hammersley_sequence(
            index, config.num_views, offset
        )
        fov = float(2 * np.arcsin(np.sqrt(3) / 2 / radius))
        result.append(
            {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "radius": float(radius),
                "fov": fov,
                "fov_degrees": float(np.rad2deg(fov)),
            }
        )
    return result
