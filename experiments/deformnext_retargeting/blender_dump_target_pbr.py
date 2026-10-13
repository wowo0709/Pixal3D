#!/usr/bin/env python3
"""Run inside Blender to materialize source PBR on each deformed target mesh."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import bpy
import numpy as np


def remap(path: str, mappings: list[tuple[str, str]]) -> Path:
    for source, target in mappings:
        if path == source or path.startswith(source.rstrip("/") + "/"):
            return Path(target + path[len(source):])
    return Path(path)


def _depends_on_vertex_color(socket, seen=None) -> bool:
    """Return whether a shader input is driven by a vertex-color node."""
    seen = set() if seen is None else seen
    for link in socket.links:
        node = link.from_node
        identity = node.as_pointer()
        if identity in seen:
            continue
        seen.add(identity)
        if node.type == "VERTEX_COLOR":
            return True
        if any(_depends_on_vertex_color(value, seen) for value in node.inputs):
            return True
    return False


def _materialize_vertex_color_fallback(
    materials: list, material_ids: np.ndarray, corner_color: np.ndarray
) -> None:
    """Collapse unsupported vertex-color shaders to their material-wise mean.

    TRELLIS.2's PBR dump format has no per-corner color channel.  Canonical
    ingest still preserves those colors in ``geometry.npz``.  For the small
    set of assets whose GLB shader reads COLOR_0, use the mean color of the
    corresponding material instead of rejecting the otherwise valid sample.
    Textured and constant Principled materials remain untouched.
    """
    for index, material in enumerate(materials):
        if not material.use_nodes:
            continue
        principled = material.node_tree.nodes.get("Principled BSDF")
        if principled is None:
            continue
        selected = corner_color[material_ids == index]
        if selected.size == 0:
            rgba = np.ones(4, dtype=np.float32)
        else:
            rgba = np.clip(selected.reshape(-1, 4).mean(axis=0), 0.0, 1.0)
        for name, value in (("Base Color", rgba), ("Alpha", float(rgba[3]))):
            socket = principled.inputs.get(name)
            if socket is None or not socket.is_linked or not _depends_on_vertex_color(socket):
                continue
            for link in tuple(socket.links):
                material.node_tree.links.remove(link)
            socket.default_value = value


def rebuild_target(canonical: Path, target_geometry: Path) -> None:
    from data_toolkit.blender_script import dump_pbr

    dump_pbr.init_scene()
    dump_pbr.load_object(str(canonical / "canonical.glb"))
    metadata = json.loads((canonical / "canonical.json").read_text())
    with np.load(canonical / "geometry.npz") as source:
        faces = np.asarray(source["F"], dtype=np.int32)
        material_ids = np.asarray(source["material_id"], dtype=np.int32)
        corner_uv = np.asarray(source["corner_uv"], dtype=np.float32)
        corner_color = np.asarray(source["corner_color"], dtype=np.float32)
        source_count = len(source["V"])
    if target_geometry.suffix == ".npy":
        vertices = np.asarray(np.load(target_geometry), dtype=np.float32)
    else:
        with np.load(target_geometry) as target:
            vertices = np.asarray(target["V"], dtype=np.float32)
    if vertices.shape != (source_count, 3) or not np.isfinite(vertices).all():
        raise ValueError("target vertices violate the topology correspondence contract")

    materials = []
    for name in metadata["material_names"]:
        material = bpy.data.materials.get(name)
        if material is None:
            raise ValueError(f"canonical material is missing: {name}")
        materials.append(material)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    mesh = bpy.data.meshes.new("deformed_canonical_face_order")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    for material in materials:
        mesh.materials.append(material)
    for face_id, polygon in enumerate(mesh.polygons):
        polygon.material_index = int(material_ids[face_id])
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set("uv", corner_uv.reshape(-1))
    if np.any(corner_color < 0.999):
        colors = mesh.color_attributes.new(
            name="COLOR_0", type="FLOAT_COLOR", domain="CORNER"
        )
        colors.data.foreach_set("color", corner_color.reshape(-1))
    _materialize_vertex_color_fallback(materials, material_ids, corner_color)
    obj = bpy.data.objects.new("deformed_canonical", mesh)
    bpy.context.collection.objects.link(obj)


def dump_one(row: dict, output: Path, mappings: list[tuple[str, str]]) -> None:
    from data_toolkit.blender_script import dump_pbr

    canonical = remap(row["source_geometry"], mappings).parent
    target = remap(row["target_vertices"], mappings)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    error_path = Path(f"{temporary}_error.txt")
    try:
        rebuild_target(canonical, target)
        # dump_pbr treats a .blend input as an already-loaded scene.  Using
        # that path preserves the exact parser/normalization while avoiding a
        # redundant target GLB export and immediate re-import for every row.
        dump_pbr.main(SimpleNamespace(
            object="deformnext-current-scene.blend", output_path=str(temporary)
        ))
        with temporary.open("rb") as stream:
            value = pickle.load(stream)
        # Material-free/vertex-color assets are valid: prepare_pbr_dump adds
        # the same neutral fallback used by the production voxelizer.
        if not value.get("objects"):
            raise ValueError("target PBR dump is empty")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)


def main() -> None:
    if "--" not in sys.argv:
        raise ValueError("expected Blender arguments after --")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[], metavar="FROM=TO")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    sys.path.insert(0, str(args.repo))
    mappings = [tuple(value.split("=", 1)) for value in args.path_map]
    all_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    rows = [row for index, row in enumerate(all_rows) if index % args.world_size == args.rank]
    if args.limit is not None:
        rows = rows[:args.limit]
    completed = skipped = failed = 0
    failures = args.output_root / f"failures-rank{args.rank:03d}.jsonl"
    for row in rows:
        output = args.output_root / "pbr_dumps" / f"{row['sample_id']}.pickle"
        if output.is_file():
            try:
                with output.open("rb") as stream:
                    pickle.load(stream)
                skipped += 1
                continue
            except Exception:
                output.unlink(missing_ok=True)
        try:
            dump_one(row, output, mappings)
            completed += 1
        except Exception as error:
            failed += 1
            with failures.open("a") as stream:
                stream.write(json.dumps({
                    "sample_id": row["sample_id"],
                    "error": f"{type(error).__name__}: {error}",
                }) + "\n")
        if (completed + skipped + failed) % 10 == 0:
            print(json.dumps({
                "rank": args.rank, "completed": completed,
                "skipped": skipped, "failed": failed,
            }), flush=True)
    print(json.dumps({
        "rank": args.rank, "assigned": len(rows), "completed": completed,
        "skipped": skipped, "failed": failed,
    }), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
