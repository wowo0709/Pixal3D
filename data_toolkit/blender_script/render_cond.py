import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json
import glob
from PIL import Image


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}


def init_render(engine='CYCLES', resolution=512):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3
    bpy.context.scene.cycles.transmission_bounces = 3
    bpy.context.scene.cycles.use_denoising = True
    

def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)
        

def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam


def init_uniform_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    # Create environment light
    if bpy.context.scene.world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    else:
        world = bpy.context.scene.world
        
    # Enabling nodes
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    
    # Remove default nodes
    for node in nodes:
        nodes.remove(node)

    # Create background node
    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = 1.0
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


def init_random_lighting(camera_dir: np.ndarray) -> None:
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    # Create environment light
    if bpy.context.scene.world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    else:
        world = bpy.context.scene.world
        
    # Enabling nodes
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    
    # Remove default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Random place lights
    num_lights = np.random.randint(1, 4)
    total_strength = 1.5
    for i in range(num_lights):
        new_light = bpy.data.objects.new(f"Light_{i}", bpy.data.lights.new(f"Light_{i}", type="POINT"))
        bpy.context.collection.objects.link(new_light)
        
        new_light_distance = 1 / np.random.uniform(1/100, 1/10)
        new_light_dir = np.random.randn(3)
        new_light_dir[2] += 0.6
        new_light_dir = new_light_dir / np.linalg.norm(new_light_dir)
        new_light_location = new_light_dir * new_light_distance
        new_light_camera_strength_ratio = max(np.sum(camera_dir * new_light_dir) * 0.5 + 0.5, 0)
        new_light_max_energy = total_strength / (np.sum(camera_dir * new_light_dir) * 0.45 + 0.55)
        new_light_strength = np.sqrt(np.random.uniform(0.01, 1)) * new_light_max_energy
        new_light_camera_strength = new_light_camera_strength_ratio * new_light_strength
        total_strength -= new_light_camera_strength
        
        new_light.location = (new_light_location[0], new_light_location[1], new_light_location[2])
        new_light.data.color = (1.0, 1.0, 1.0)
        new_light.data.energy = new_light_strength * new_light_distance**2 * 31.4
        new_light.data.shadow_soft_size = np.random.uniform(0.1, 0.1 * new_light_distance)
        
    # Create background node
    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = total_strength
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    else:
        import_function(filepath=object_path)
        

def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    # bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def unhide_all_objects() -> None:
    """Unhides all objects in the scene.

    Returns:
        None
    """
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)
        
        
def convert_to_meshes() -> None:
    """Converts all objects in the scene to meshes.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
    
        
def triangulate_meshes() -> None:
    """Triangulates all meshes in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset


def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix


def check_mask_boundary_distance(
    image_path: str,
    threshold: int = 0,
    min_boundary_distance: float = 130,
) -> Tuple[bool, bool, int]:
    """Check the rendered object's mask distance to image boundary.
    
    Args:
        image_path: Path to the rendered PNG image with alpha channel.
        threshold: Alpha value threshold to consider as valid mask (0-255).
        
    Returns:
        Tuple of (touches_boundary, too_far_from_boundary, min_distance):
        - touches_boundary: True if the mask touches any boundary
        - too_far_from_boundary: True if the mask is too far from all boundaries (>80 pixels)
        - min_distance: Minimum distance from mask to any boundary
    """
    img = Image.open(image_path)
    if img.mode != 'RGBA':
        return False, False, 0
    
    # Get alpha channel
    alpha = np.array(img)[:, :, 3]
    h, w = alpha.shape
    
    # Find all pixels with alpha > threshold (mask pixels)
    mask_pixels = np.where(alpha > threshold)
    
    if len(mask_pixels[0]) == 0:
        # No valid mask pixels
        return False, True, max(h, w)
    
    # Calculate distances to each boundary
    min_row = np.min(mask_pixels[0])  # Distance to top edge
    max_row = np.max(mask_pixels[0])  # Distance to bottom edge
    min_col = np.min(mask_pixels[1])  # Distance to left edge
    max_col = np.max(mask_pixels[1])  # Distance to right edge
    
    dist_top = min_row
    dist_bottom = (h - 1) - max_row
    dist_left = min_col
    dist_right = (w - 1) - max_col
    
    min_distance = min(dist_top, dist_bottom, dist_left, dist_right)
    
    # Check if touches boundary (distance <= 0)
    touches_boundary = min_distance <= 0
    
    # Check if too far from boundary for the configured render resolution.
    too_far = min_distance > min_boundary_distance
    
    return touches_boundary, too_far, min_distance


def main(arg):
    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
    print('[INFO] Scene initialized.')
    
    # normalize scene
    scale, offset = normalize_scene()
    print('[INFO] Scene normalized.')
    
    # Initialize camera and lighting
    cam = init_camera()
    init_uniform_lighting()
    print('[INFO] Camera and lighting initialized.')

    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = arg.cycles_device
    preferences.get_devices()
    selected_devices = []
    for device in preferences.devices:
        device.use = device.type == arg.cycles_device
        if device.use:
            selected_devices.append(device.name)
    if not selected_devices:
        raise RuntimeError(f"No {arg.cycles_device} device selected")
    bpy.context.scene.cycles.device = "GPU"

    # ============= Render conditional views =============
    init_render(engine=arg.engine, resolution=arg.cond_resolution)
    # Create a list of views
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "selected_devices": selected_devices,
        "frames": []
    }
    views = json.loads(arg.cond_views)
    
    # Parameters for boundary check and radius adjustment
    max_retry = 10  # Maximum number of retries per view
    radius_increase_factor = 1.1  # Increase radius by 10% when too close to boundary
    radius_decrease_factor = 0.9  # Decrease radius by 10% when too far from boundary
    min_boundary_distance = 130 * arg.cond_resolution / 1024
    
    for i, view in enumerate(views):
        current_radius = view['radius']
        retry_count = 0
        
        while retry_count < max_retry:
            cam_dir = np.array([
                np.cos(view['yaw']) * np.cos(view['pitch']),
                np.sin(view['yaw']) * np.cos(view['pitch']),
                np.sin(view['pitch'])
            ])
            init_random_lighting(cam_dir)
            cam.location = (
                current_radius * cam_dir[0],
                current_radius * cam_dir[1],
                current_radius * cam_dir[2]
            )
            cam.data.lens = 16 / np.tan(view['fov'] / 2)
            
            output_path = os.path.join(arg.cond_output_folder, f'{i:03d}.png')
            bpy.context.scene.render.filepath = output_path
                
            # Render the scene
            bpy.ops.render.render(write_still=True)
            bpy.context.view_layer.update()
            
            # Check mask boundary distance
            touches_boundary, too_far, min_dist = check_mask_boundary_distance(
                output_path,
                min_boundary_distance=min_boundary_distance,
            )
            
            if touches_boundary:
                # Object is too close to boundary, increase radius
                retry_count += 1
                old_radius = current_radius
                current_radius *= radius_increase_factor
                print(f'[WARNING] View {i}: Mask touches boundary (dist={min_dist}px). Increasing radius from {old_radius:.4f} to {current_radius:.4f} (retry {retry_count}/{max_retry})')
            elif too_far:
                # Object is too far from boundary (>80px), decrease radius
                retry_count += 1
                old_radius = current_radius
                current_radius *= radius_decrease_factor
                print(f'[INFO] View {i}: Mask too far from boundary (dist={min_dist}px > {min_boundary_distance}px). Decreasing radius from {old_radius:.4f} to {current_radius:.4f} (retry {retry_count}/{max_retry})')
            else:
                # Good distance, stop retrying
                if retry_count > 0:
                    print(f'[INFO] View {i}: Mask boundary distance OK (dist={min_dist}px) after {retry_count} retries. Final radius: {current_radius:.4f}')
                else:
                    print(f'[INFO] View {i}: Mask boundary distance OK (dist={min_dist}px). Radius: {current_radius:.4f}')
                break
        
        if retry_count >= max_retry:
            print(f'[WARNING] View {i}: Max retries reached. Using final radius: {current_radius:.4f} (dist={min_dist}px)')
            
        # Save camera parameters (with potentially updated radius)
        metadata = {
            "file_path": f'{i:03d}.png',
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam),
            "radius": current_radius,  # Save the actual radius used
            "original_radius": view['radius'],  # Save original for reference
            "retries": retry_count
        }
        to_export["frames"].append(metadata)
    
    # Save the camera parameters
    with open(os.path.join(arg.cond_output_folder, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--cond_views', type=str, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--cond_output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--cond_resolution', type=int, default=1024, help='Resolution of the conditional images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument("--cycles_device", type=str, default="OPTIX", help="Cycles compute device type.")
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
