"""Deterministic leaf-command substitute for the pipeline integration test."""

import csv
import fcntl
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import pickle
import tempfile


_VXZ_TEMPLATE = None


def _option(name):
    try:
        return os.sys.argv[os.sys.argv.index(name) + 1]
    except (IndexError, ValueError) as error:
        raise ValueError(f"missing fake worker option: {name}") from error


def _sync_parent(path):
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_parent(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(path, value):
    _atomic_bytes(
        path,
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )


def _atomic_csv(path, fieldnames, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode("utf-8"))


def _instances():
    values = tuple(Path(_option("--instances")).read_text().splitlines())
    if not values:
        raise ValueError("fake worker instances must not be empty")
    return values


def _ranked_instances():
    values = _instances()
    if "--rank" not in os.sys.argv:
        return values
    rank = int(_option("--rank"))
    world_size = int(_option("--world_size"))
    return tuple(
        value for index, value in enumerate(values) if index % world_size == rank
    )


def _record_invocation(script):
    instances = Path(_option("--instances"))
    counts_path = instances.parent / "leaf-command-counts.json"
    lock_path = instances.parent / "leaf-command-counts.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            counts = (
                json.loads(counts_path.read_text()) if counts_path.exists() else {}
            )
            counts[script] = counts.get(script, 0) + 1
            _atomic_json(counts_path, counts)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _download():
    metadata_root = Path(_option("--root"))
    download_root = Path(_option("--download_root"))
    selected = set(_instances())
    with (metadata_root / "metadata.csv").open(newline="") as stream:
        records = {
            row["sha256"]: row for row in csv.DictReader(stream)
            if row["sha256"] in selected
        }
    if set(records) != selected:
        raise ValueError("fake download metadata does not cover instances")
    raw_records = []
    for asset_sha in sorted(selected):
        record = records[asset_sha]
        payload = record["fixture_payload"].encode("ascii")
        if sha256(payload).hexdigest() != asset_sha:
            raise ValueError("fake download payload checksum mismatch")
        relative = Path(record["file_identifier"])
        _atomic_bytes(download_root / relative, payload)
        raw_records.append(
            {"sha256": asset_sha, "local_path": relative.as_posix()}
        )
    _atomic_csv(
        download_root / "raw/metadata.csv",
        ("sha256", "local_path"),
        raw_records,
    )


def _dump(directory, pbr=False):
    root = Path(_option("--pbr_dump_root" if pbr else "--mesh_dump_root"))
    value = {"objects": []}
    if pbr:
        value["materials"] = []
    for asset_sha in _instances():
        _atomic_bytes(
            root / directory / f"{asset_sha}.pickle",
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
        )


def _asset_stats():
    root = Path(_option("--root")) / "asset_stats/new_records"
    _atomic_csv(
        root / "part_fake.csv",
        ("sha256", "num_faces", "num_vertices"),
        (
            {"sha256": asset_sha, "num_faces": 12, "num_vertices": 8}
            for asset_sha in _instances()
        ),
    )


def _render():
    from PIL import Image, ImageDraw

    root = Path(_option("--render_cond_root")) / "renders_cond"
    views = int(_option("--num_cond_views"))
    resolution = int(_option("--cond_resolution"))
    frames = [
        {
            "file_path": f"{view:03d}.png",
            "camera_angle_x": 0.5,
            "radius": 2.0,
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        for view in range(views)
    ]
    for asset_sha in _ranked_instances():
        output = root / asset_sha
        for view in range(views):
            image = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
            inset = resolution // 4
            ImageDraw.Draw(image).rectangle(
                (inset, inset, resolution - inset - 1, resolution - inset - 1),
                fill=(32 + view, 96, 160, 255),
            )
            temporary = None
            path = output / f"{view:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with tempfile.NamedTemporaryFile(
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                image.save(temporary, format="PNG")
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                _sync_parent(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        _atomic_json(output / "transforms.json", {"frames": frames})


def _atomic_vxz(path):
    global _VXZ_TEMPLATE
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shared_template = os.environ.get("PIXAL3D_FAKE_VXZ_TEMPLATE")
    if shared_template and Path(shared_template).is_file():
        _atomic_bytes(path, Path(shared_template).read_bytes())
        return
    if os.environ.get("PIXAL3D_FAKE_FAST") == "1" and _VXZ_TEMPLATE is not None:
        _atomic_bytes(path, _VXZ_TEMPLATE)
        return
    import o_voxel
    import torch

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".vxz",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        o_voxel.io.write_vxz(
            str(temporary),
            torch.tensor([[1, 2, 3]], dtype=torch.int32),
            {"value": torch.tensor([[1]], dtype=torch.uint8)},
            num_threads=1,
        )
        o_voxel.io.read_vxz_info(str(temporary))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_parent(path)
        o_voxel.io.read_vxz_info(str(path))
        if os.environ.get("PIXAL3D_FAKE_FAST") == "1":
            _VXZ_TEMPLATE = path.read_bytes()
            if shared_template:
                _atomic_bytes(Path(shared_template), _VXZ_TEMPLATE)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _voxel(directory):
    resolution = int(_option("--resolution"))
    root_name = (
        f"dual_grid_view_{resolution}"
        if directory == "dual_grid"
        else f"pbr_voxels_view_fix_{resolution}"
    )
    root = Path(
        _option("--dual_grid_root" if directory == "dual_grid" else "--pbr_voxel_root")
    ) / root_name
    for asset_sha in _instances():
        for view in (0, 1):
            output = root / asset_sha / f"view{view:02d}.vxz"
            _atomic_json(
                output.with_name(f"view{view:02d}_scale.json"),
                {"total_scale": 1.0},
            )
            _atomic_vxz(output)


def _atomic_sparse(path, resolution, ss=False):
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            if ss:
                np.savez_compressed(stream, z=np.array([[1.0]], dtype=np.float32))
            else:
                np.savez_compressed(
                    stream,
                    feats=np.array([[1.0, 2.0]], dtype=np.float32),
                    coords=np.array([[1, 2, 3]], dtype=np.int32),
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_parent(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    _atomic_json(path.with_name(f"{path.stem}_scale.json"), {"total_scale": 1.0})


def _encode(family):
    resolution = int(_option("--resolution"))
    if family == "shape":
        root = Path(_option("--shape_latent_root")) / (
            f"shape_latents/shape_enc_next_dc_f16c32_fp16_{resolution}_view"
        )
    elif family == "pbr":
        root = Path(_option("--pbr_latent_root")) / (
            f"pbr_latents/tex_enc_next_dc_f16c32_fp16_{resolution}_view_fix"
        )
    else:
        root = Path(_option("--ss_latent_root")) / (
            f"ss_latents/ss_enc_conv3d_16l8_fp16_{resolution}_view"
        )
    for asset_sha in _ranked_instances():
        for view in (0, 1):
            _atomic_sparse(
                root / asset_sha / f"view{view:02d}.npz",
                resolution,
                ss=family == "ss",
            )


def main():
    script = _option("--original-script")
    _record_invocation(script)
    handlers = {
        "download.py": _download,
        "dump_mesh.py": lambda: _dump("mesh_dumps"),
        "dump_pbr.py": lambda: _dump("pbr_dumps", pbr=True),
        "asset_stats.py": _asset_stats,
        "render_cond.py": _render,
        "dual_grid_view.py": lambda: _voxel("dual_grid"),
        "voxelize_pbr_view.py": lambda: _voxel("pbr"),
        "encode_shape_latent_view.py": lambda: _encode("shape"),
        "encode_pbr_latent_view.py": lambda: _encode("pbr"),
        "encode_ss_latent_view.py": lambda: _encode("ss"),
    }
    try:
        handler = handlers[script]
    except KeyError as error:
        raise ValueError(f"unsupported fake leaf script: {script}") from error
    handler()


if __name__ == "__main__":
    main()
