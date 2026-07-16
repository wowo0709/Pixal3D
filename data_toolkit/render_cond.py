import argparse
import ctypes
import errno
import importlib
import json
import math
import os
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from easydict import EasyDict as edict
import pandas as pd
from PIL import Image
from tqdm import tqdm
try:
    from data_toolkit.pipeline.blender import ensure_blender
    from data_toolkit.pipeline.camera import build_condition_views
    from data_toolkit.pipeline.config import RenderConfig
except ModuleNotFoundError as error:
    if error.name != "data_toolkit":
        raise
    from pipeline.blender import ensure_blender
    from pipeline.camera import build_condition_views
    from pipeline.config import RenderConfig


DEFAULT_BLENDER_TOOL_ROOT = Path("/tmp")
OBJAVERSE_ALIASES = {
    "ObjaverseXL_sketchfab": "sketchfab",
    "ObjaverseXL_github": "github",
}
AT_FDCWD = -100
RENAME_EXCHANGE = 2
LIBC = ctypes.CDLL(None, use_errno=True)
RENAMEAT2 = getattr(LIBC, "renameat2", None)
if RENAMEAT2 is not None:
    RENAMEAT2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    RENAMEAT2.restype = ctypes.c_int


def _import_adapter(adapter_name: str):
    package_name = f"data_toolkit.datasets.{adapter_name}"
    legacy_name = f"datasets.{adapter_name}"
    candidates = (
        (package_name, legacy_name)
        if __package__
        else (legacy_name, package_name)
    )
    try:
        return importlib.import_module(candidates[0])
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if not (
            candidates[0] == missing
            or candidates[0].startswith(f"{missing}.")
        ):
            raise
    return importlib.import_module(candidates[1])


def _install_blender(tool_root: Path = DEFAULT_BLENDER_TOOL_ROOT) -> Path:
    return ensure_blender(tool_root)


def _finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_render_output(
    path: Path, expected_views: int, resolution: int
) -> None:
    transforms_path = path / "transforms.json"
    if not transforms_path.is_file():
        raise ValueError(f"missing render metadata: {transforms_path}")
    try:
        transforms = json.loads(transforms_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid render metadata: {transforms_path}") from error

    if not isinstance(transforms, dict):
        raise ValueError(f"invalid render metadata: {transforms_path}")
    selected_devices = transforms.get("selected_devices")
    if (
        not isinstance(selected_devices, list)
        or not selected_devices
        or any(
            not isinstance(name, str)
            or not name.strip()
            or "CPU" in name.upper()
            for name in selected_devices
        )
    ):
        raise ValueError(f"invalid selected devices: {transforms_path}")

    frames = transforms.get("frames")
    if not isinstance(frames, list) or len(frames) != expected_views:
        raise ValueError(
            f"expected {expected_views} render frames in {transforms_path}"
        )
    for index, frame in enumerate(frames):
        expected_name = f"{index:03d}.png"
        if not isinstance(frame, dict) or frame.get("file_path") != expected_name:
            raise ValueError(f"invalid render frame {index}: {transforms_path}")
        if not _finite_number(frame.get("camera_angle_x")):
            raise ValueError(f"invalid camera angle for frame {index}")
        if not _finite_number(frame.get("radius")) or frame["radius"] <= 0:
            raise ValueError(f"invalid camera radius for frame {index}")
        matrix = frame.get("transform_matrix")
        if (
            not isinstance(matrix, list)
            or len(matrix) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
            or any(not _finite_number(value) for row in matrix for value in row)
        ):
            raise ValueError(f"invalid transform for frame {index}")
        image_path = path / expected_name
        if not image_path.is_file():
            raise ValueError(f"missing render image: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.load()
                image_format = image.format
                image_mode = image.mode
                image_size = image.size
        except OSError as error:
            raise ValueError(f"invalid render image: {image_path}") from error
        if (
            image_format != "PNG"
            or image_mode != "RGBA"
            or image_size != (resolution, resolution)
        ):
            raise ValueError(
                f"invalid render image contract: {image_path} "
                f"({image_format}, {image_mode}, {image_size})"
            )


def _cleanup_legacy_previous(final: Path) -> None:
    previous = final.with_name(f".{final.name}.previous")
    if previous.is_dir():
        shutil.rmtree(previous)
    elif previous.exists():
        previous.unlink()


def _rename_exchange(left: Path, right: Path) -> None:
    if RENAMEAT2 is None:
        raise OSError(errno.ENOSYS, "libc renameat2 is unavailable")
    ctypes.set_errno(0)
    result = RENAMEAT2(
        AT_FDCWD,
        os.fsencode(left),
        AT_FDCWD,
        os.fsencode(right),
        RENAME_EXCHANGE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            f"{left} <-> {right}",
        )


def _publish_render_output(temporary: Path, final: Path) -> None:
    if not final.exists():
        os.replace(temporary, final)
        return

    _rename_exchange(temporary, final)
    shutil.rmtree(temporary)


def _render_cond(
    file_path,
    sha256,
    root,
    config,
    blender_path,
    timeout_seconds,
):
    cond_views = build_condition_views(sha256, config)
    final = Path(root) / "renders_cond" / sha256
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{sha256}.", dir=final.parent)
    )
    args = [
        str(blender_path),
        "-b",
        "-P",
        os.path.join(os.path.dirname(__file__), "blender_script", "render_cond.py"),
        "--",
        "--object",
        os.path.expanduser(file_path),
        "--cond_views",
        json.dumps(cond_views),
        "--cond_resolution",
        str(config.resolution),
        "--cond_output_folder",
        str(temporary),
        "--engine",
        "CYCLES",
        "--cycles_device",
        config.cycles_device,
    ]
    if file_path.endswith(".blend"):
        args.insert(1, file_path)
    try:
        subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=timeout_seconds,
        )
        _validate_render_output(
            temporary, config.num_views, config.resolution
        )
        _publish_render_output(temporary, final)
        _cleanup_legacy_previous(final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"sha256": sha256, "cond_rendered": True}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("dataset name is required")

    canonical_source = OBJAVERSE_ALIASES.get(argv[0])
    adapter_name = "ObjaverseXL" if canonical_source else argv[0]
    dataset_utils = _import_adapter(adapter_name)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True, help="Directory to save the metadata"
    )
    parser.add_argument(
        "--download_root",
        type=str,
        default=None,
        help="Directory containing the downloaded files",
    )
    parser.add_argument(
        "--render_cond_root",
        type=str,
        default=None,
        help="Directory to save the condition renders",
    )
    parser.add_argument(
        "--filter_low_aesthetic_score",
        type=float,
        default=None,
        help="Filter objects with aesthetic score lower than this value",
    )
    parser.add_argument(
        "--instances", type=str, default=None, help="Instances to process"
    )
    parser.add_argument(
        "--num_cond_views",
        type=int,
        default=8,
        help="Number of conditional views to render",
    )
    parser.add_argument("--cond_resolution", type=int, default=512)
    parser.add_argument("--blender_path", type=str, default=None)
    parser.add_argument("--cycles_device", type=str, default="OPTIX")
    parser.add_argument("--timeout_seconds", type=int, default=900)
    dataset_utils.add_args(parser)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=8)
    opt = edict(vars(parser.parse_args(argv[1:])))
    if canonical_source is not None:
        opt.source = canonical_source
    opt.download_root = opt.download_root or opt.root
    opt.render_cond_root = opt.render_cond_root or opt.root

    os.makedirs(
        os.path.join(opt.render_cond_root, "renders_cond", "new_records"),
        exist_ok=True,
    )
    blender_path = (
        Path(opt.blender_path).expanduser()
        if opt.blender_path
        else _install_blender()
    )
    render_config = RenderConfig(
        num_views=opt.num_cond_views,
        resolution=opt.cond_resolution,
        fov_min_degrees=10.0,
        fov_max_degrees=70.0,
        camera_policy="pixal3d-mv-camera-v1",
        blender_version="4.5.1",
        cycles_device=opt.cycles_device,
    )

    # get file list
    if not os.path.exists(os.path.join(opt.root, "metadata.csv")):
        raise ValueError("metadata.csv not found")
    metadata = pd.read_csv(
        os.path.join(opt.root, "metadata.csv"), dtype={"sha256": str}
    ).set_index("sha256")
    aesthetic_path = os.path.join(
        opt.root, "aesthetic_scores", "metadata.csv"
    )
    if os.path.exists(aesthetic_path):
        metadata = metadata.combine_first(
            pd.read_csv(aesthetic_path, dtype={"sha256": str}).set_index(
                "sha256"
            )
        )
    downloaded_path = os.path.join(opt.download_root, "raw", "metadata.csv")
    if os.path.exists(downloaded_path):
        metadata = metadata.combine_first(
            pd.read_csv(downloaded_path, dtype={"sha256": str}).set_index(
                "sha256"
            )
        )
    rendered_path = os.path.join(
        opt.render_cond_root, "renders_cond", "metadata.csv"
    )
    if os.path.exists(rendered_path):
        metadata = metadata.combine_first(
            pd.read_csv(rendered_path, dtype={"sha256": str}).set_index(
                "sha256"
            )
        )
    metadata = metadata.reset_index()
    if opt.instances is None:
        metadata = metadata[metadata["local_path"].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[
                metadata["aesthetic_score"] >= opt.filter_low_aesthetic_score
            ]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r") as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata.iloc[start:end]
    records = []

    # filter out objects that are already processed
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, tqdm(
        total=len(metadata), desc="Filtering existing objects"
    ) as pbar:
        def check_sha256(sha256):
            final = (
                Path(opt.render_cond_root) / "renders_cond" / sha256
            )
            try:
                _validate_render_output(
                    final, render_config.num_views, render_config.resolution
                )
            except ValueError:
                pass
            else:
                records.append({"sha256": sha256, "cond_rendered": True})
                _cleanup_legacy_previous(final)
            finally:
                pbar.update()
        list(executor.map(check_sha256, metadata["sha256"].values))
    existing_sha256 = set(r["sha256"] for r in records)
    metadata = metadata[~metadata["sha256"].isin(existing_sha256)]

    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(
        _render_cond,
        root=opt.render_cond_root,
        config=render_config,
        blender_path=blender_path,
        timeout_seconds=opt.timeout_seconds,
    )
    cond_rendered = dataset_utils.foreach_instance(
        metadata,
        opt.download_root,
        func,
        max_workers=opt.max_workers,
        desc="Rendering objects",
    )
    cond_rendered = pd.concat(
        [cond_rendered, pd.DataFrame.from_records(records)]
    )
    cond_rendered.to_csv(
        os.path.join(
            opt.render_cond_root,
            "renders_cond",
            "new_records",
            f"part_{opt.rank}.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    main()
