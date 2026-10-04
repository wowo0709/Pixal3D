"""Front-first inputs for the prepared calibrated render contract."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def prepare_conditioning(manifest, view_indices, mesh_scale):
    """Select RGB images and matching K camera poses/intrinsics without cropping."""
    manifest = Path(manifest)
    frames = json.loads(manifest.read_text())['frames']
    images, angles, distances, transforms = [], [], [], []
    for index in view_indices:
        frame = frames[index]
        with Image.open(manifest.parent / frame['file_path']) as source:
            rgba = source.convert('RGBA')
        rgb = Image.new('RGB', rgba.size, (0, 0, 0))
        rgb.paste(rgba, mask=rgba.getchannel('A'))
        images.append(rgb)
        pose = np.asarray(frame['transform_matrix'], dtype=np.float32)
        angles.append(frame['camera_angle_x'])
        distances.append(float(np.linalg.norm(pose[:3, 3].astype(np.float64))))
        transforms.append(pose.tolist())
    return images, {'camera_angle_x': angles, 'distance': distances,
                    'transform_matrix': transforms, 'mesh_scale': mesh_scale}


def estimate_anchor_scale(coords, grid_resolution, old_scale, old_pose, new_pose):
    """
    - coords: n 3, occupied shape-latent grid cells in the cached anchor frame
    - old_pose, new_pose: 4 4, Blender camera-to-world transforms
    Returns an approximate new anchor scale from rotated cell centers.

    This is not an exact mesh-derived scale. The prepared shape-1024 latent has
    grid resolution 64. The rotation follows data_toolkit/utils.py:transform_mesh
    in this checkout; camera translations cancel for origin-facing poses.
    """
    # R_back @ canonical_camera_rotation is identity in transform_mesh.
    old_rotation = np.asarray(old_pose)[:3, :3].T
    new_rotation = np.asarray(new_pose)[:3, :3].T
    relative = new_rotation @ old_rotation.T
    points = (coords.astype(np.float64) + .5) / grid_resolution - .5  # n 3
    rotated = points @ relative.T  # n 3, in the new anchor frame
    return float(old_scale * .49999 / np.abs(rotated).max())
