import json
import math

import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def conditioning():
    try:
        from result_viewer import conditioning
    except ImportError:
        pytest.fail('Front-first conditioning has not been implemented.')
    return conditioning


def test_front_first_selection_reorders_images_intrinsics_and_poses_together(conditioning, tmp_path):
    frames = []
    for index in range(4):
        Image.new('RGBA', (8, 8), (20 * index, 0, 0, 255)).save(tmp_path / f'{index}.png')
        pose = np.eye(4)
        pose[1, 3] = -(index + 1)
        frames.append({'file_path': f'{index}.png', 'camera_angle_x': 0.4 + index * 0.1,
                       'transform_matrix': pose.tolist()})
    manifest = tmp_path / 'transforms.json'
    manifest.write_text(json.dumps({'frames': frames}))
    images, cameras = conditioning.prepare_conditioning(manifest, [3, 1, 0, 2], 0.9)
    assert [image.getpixel((0, 0))[0] for image in images] == [60, 20, 0, 40]
    np.testing.assert_allclose(cameras['camera_angle_x'], [0.7, 0.5, 0.4, 0.6])
    np.testing.assert_allclose(cameras['distance'], [4, 2, 1, 3])
    np.testing.assert_allclose(np.array(cameras['transform_matrix'])[:, 1, 3], [-4, -2, -1, -3])
    assert cameras['mesh_scale'] == 0.9
    assert all(image.size == (8, 8) for image in images), 'Calibrated images must not be cropped'


def test_anchor_scale_uses_rotated_surface_extent(conditioning):
    # Grid-4 centers form a .75 × .25 × .25 box. At 45° about Y its
    # X/Z extent reaches .5/sqrt(2), requiring scale .8*.49999/(.5/sqrt(2)).
    coords = np.array([[x, y, z] for x in [0, 3] for y in [1, 2] for z in [1, 2]])
    old = np.eye(4)
    old[:3, :3] = np.diag([-1, -1, 1])
    rotation = np.array([[math.sqrt(.5), 0, math.sqrt(.5)], [0, 1, 0],
                         [-math.sqrt(.5), 0, math.sqrt(.5)]])
    new = np.eye(4)
    new[:3, :3] = rotation.T @ old[:3, :3]
    value = conditioning.estimate_anchor_scale(coords, 4, 0.8, old, new)
    assert value == pytest.approx(1.1313482227, rel=1e-8)


def test_scale_rotation_matches_blender_camera_axes_for_asymmetric_support(conditioning):
    # The toolkit canonical camera cancels its axis alignment: the anchor
    # rotation is camera_rotation.T. Extra X/Y flips give the wrong extent.
    coords = np.array([[1, 3, 2], [2, 3, 2]])  # x=±.125, y=.375, z=.125
    old = np.eye(4)
    new = np.eye(4)
    new[:3, :3] = [[1, 0, 0], [0, math.sqrt(.75), .5], [0, -.5, math.sqrt(.75)]]
    value = conditioning.estimate_anchor_scale(coords, 4, 0.8, old, new)
    # max is Z = .375*sin(30°) + .125*cos(30°) = .295753175...
    assert value == pytest.approx(1.3524520890, rel=1e-8)
