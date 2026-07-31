import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def _point_at(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def main():
    args = _arguments()
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    selected_devices = []
    cpu_fallback = False
    for device in preferences.devices:
        device.use = device.type == "OPTIX"
        if device.use:
            selected_devices.append(device.name)
        if device.type == "CPU" and device.use:
            cpu_fallback = True
    if not selected_devices or cpu_fallback:
        raise RuntimeError("isolated OptiX device selection failed")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=1.5, location=(0, 0, 0))
    cube = bpy.context.object
    material = bpy.data.materials.new("CubeMaterial")
    material.diffuse_color = (0.12, 0.48, 0.8, 1.0)
    cube.data.materials.append(material)

    bpy.ops.object.camera_add(location=(3.0, -3.0, 2.2))
    camera = bpy.context.object
    _point_at(camera, (0.0, 0.0, 0.0))
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type="AREA", location=(2.5, -2.0, 3.5))
    key = bpy.context.object
    key.data.energy = 900
    key.data.shape = "DISK"
    key.data.size = 5
    _point_at(key, (0.0, 0.0, 0.0))
    bpy.ops.object.light_add(type="AREA", location=(-2.0, 1.5, 1.5))
    fill = bpy.context.object
    fill.data.energy = 500
    fill.data.size = 4
    _point_at(fill, (0.0, 0.0, 0.0))

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 8
    scene.render.resolution_x = 64
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(args.output)
    scene.world.color = (0.04, 0.04, 0.04)
    bpy.ops.render.render(write_still=True)

    args.metadata.write_text(
        json.dumps(
            {
                "selected_devices": selected_devices,
                "cpu_fallback_detected": cpu_fallback,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
