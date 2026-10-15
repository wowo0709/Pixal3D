"""Strict manifest datasets for DeformNeXt retargeting experiments.

Each item has one geometry-only M1 anchor followed by exactly three textured
M0 source views.  Transport descriptors and native Pixal values are immutable
cache inputs.  The loader refuses a textured target anchor so the target RGB
used for supervision/evaluation cannot leak into conditioning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from ..modules.sparse import SparseTensor
from .sparse_structure_latent import SparseStructureLatentVisMixin
from .structured_latent_svpbr import SLatPbrVisMixin


TRANSPORT_KEYS = (
    "target_keys",
    "source_keys",
    "source_values",
    "source_valid",
    "patch_grid",
    "spatial_indices",
    "target_query",
    "deformation",
    "patch_ids",
)

TRANSPORT_KEYS_BY_ARM = {
    "v1": (),
    "b1": ("target_keys", "source_keys", "source_values", "source_valid", "patch_grid", "patch_ids"),
    "b2": ("target_keys", "source_keys", "source_values", "source_valid", "patch_grid", "patch_ids"),
    "b3": ("target_keys", "source_keys", "source_values", "source_valid", "patch_grid", "patch_ids"),
    "c1": ("target_keys", "source_keys", "source_values", "source_valid", "target_query", "patch_grid", "patch_ids"),
    "c2": ("target_keys", "source_keys", "source_values", "source_valid", "target_query", "deformation", "patch_grid", "patch_ids"),
}

CACHED_TRANSPORT_KEYS_BY_ARM = {
    "v1": TRANSPORT_KEYS_BY_ARM["v1"],
    "b1": ("source_values", "source_valid", "patch_grid", "patch_ids"),
    "b2": ("source_values", "source_valid", "patch_grid", "patch_ids"),
    "b3": ("source_values", "source_valid", "patch_grid", "patch_ids"),
    "c1": ("source_values", "source_valid", "target_query", "patch_grid", "patch_ids"),
    "c2": ("source_values", "source_valid", "target_query", "patch_grid", "patch_ids"),
}

MATCH_KEYS_BY_ARM = {
    "b1": ("semantic_indices",),
    "b2": ("semantic_indices", "matching_logits", "candidate_valid"),
    "b3": ("semantic_indices", "matching_logits", "candidate_valid"),
    "c1": ("semantic_indices", "matching_logits", "candidate_valid"),
    "c2": (
        "semantic_indices", "matching_logits", "candidate_valid", "deformation"
    ),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    if not rows:
        raise ValueError(f"empty DeformNeXt manifest: {path}")
    return rows


def _path(value: str, manifest_dir: Path) -> Path:
    result = Path(value)
    return result if result.is_absolute() else manifest_dir / result


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGBA").resize(
            (image_size, image_size), Image.Resampling.LANCZOS
        )
        rgba = torch.from_numpy(np.asarray(image).copy()).float() / 255.0
    return rgba[..., :3].permute(2, 0, 1) * rgba[..., 3].unsqueeze(0)


def _load_array(path: Path, preferred: str, expected_schema: Optional[str] = None) -> torch.Tensor:
    if path.suffix == ".npy":
        if expected_schema is not None:
            raise ValueError("versioned SS latent requires NPZ metadata")
        array = np.load(path)
    else:
        with np.load(path) as archive:
            if expected_schema is not None and ("schema" not in archive or str(archive["schema"]) != expected_schema):
                raise ValueError(f"SS latent schema mismatch: {path}")
            if preferred not in archive:
                raise KeyError(f"{path} has no {preferred!r} array")
            array = archive[preferred]
    # np.load already materializes an owning ndarray for NPZ members.  The
    # tensor keeps that ndarray alive after the archive closes, so a second
    # full copy only wastes CPU time and memory bandwidth.
    return torch.from_numpy(np.asarray(array)).float()


def _latent_arrays(path: Path, expected_schema: Optional[str] = None, expected_scale: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path) as archive:
        if expected_schema is not None:
            if 'schema' not in archive or str(archive['schema']) != expected_schema:
                raise ValueError(f'sparse latent schema mismatch: {path}')
            if 'mesh_scale' not in archive or not np.isclose(float(archive['mesh_scale']), expected_scale, rtol=1e-5, atol=1e-6):
                raise ValueError(f'sparse latent scale mismatch: {path}')
        coords = torch.from_numpy(np.asarray(archive["coords"])).int()
        feats = torch.from_numpy(np.asarray(archive["feats"])).float()
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"invalid sparse coordinates in {path}: {tuple(coords.shape)}")
    if coords.shape[1] == 4:
        if torch.any(coords[:, 0] != 0):
            raise ValueError(f"stored sparse batch column must be zero: {path}")
        coords = coords[:, 1:]
    if feats.ndim != 2 or feats.shape[0] != coords.shape[0]:
        raise ValueError(f"coordinate/feature mismatch in {path}")
    return coords, feats


def _bounded_load_groups(load, num_groups):
    """Balance sparse tokens without exceeding the dense image microbatch cap."""
    if not 1 <= num_groups <= len(load):
        raise ValueError('accumulation groups must not exceed the asset count')
    capacity = (len(load) + num_groups - 1) // num_groups
    groups = [[] for _ in range(num_groups)]
    totals = [0] * num_groups
    for index in sorted(range(len(load)), key=lambda i: load[i], reverse=True):
        eligible = [group for group in range(num_groups) if len(groups[group]) < capacity]
        group = min(eligible, key=lambda group: totals[group])
        groups[group].append(index)
        totals[group] += load[index]
    return groups


class _DeformNeXtBase(Dataset):
    def _resolve(self, value: str) -> Path:
        for source, target in self.path_map:
            if value == source or value.startswith(source.rstrip("/") + "/"):
                value = target + value[len(source):]
                break
        return _path(value, self.manifest_dir)

    def _init_manifest(
        self,
        manifest: str,
        *,
        image_size: int,
        expected_split: Optional[str],
        verify_files: bool,
        path_map: Optional[dict[str, str]],
        transport_arm: Optional[str],
        match_root: Optional[str],
        projection_root: Optional[str],
        transport_implementation: str = "legacy-v4",
        expected_q2c_sha256: Optional[str] = None,
    ) -> None:
        self.manifest_path = Path(manifest).resolve()
        self.manifest_dir = self.manifest_path.parent
        self.rows = _read_jsonl(self.manifest_path)
        self.image_size = int(image_size)
        self.path_map = tuple((str(source), str(target)) for source, target in (path_map or {}).items())
        if transport_arm is not None and transport_arm not in TRANSPORT_KEYS_BY_ARM:
            raise ValueError(f"unknown transport arm: {transport_arm}")
        self.transport_arm = transport_arm
        if transport_implementation not in {"legacy-v4", "r9-native-v5"}:
            raise ValueError("unknown transport implementation")
        self.transport_implementation = transport_implementation
        self.expected_q2c_sha256 = expected_q2c_sha256
        self.match_root = None if match_root is None else self._resolve(str(match_root))
        self.projection_root = None if projection_root is None else self._resolve(str(projection_root))
        self.transport_keys = (
            TRANSPORT_KEYS if transport_arm is None else (
                CACHED_TRANSPORT_KEYS_BY_ARM[transport_arm]
                if self.match_root is not None else TRANSPORT_KEYS_BY_ARM[transport_arm]
            )
        )
        self.value_range = (0, 1)
        self.instances = []
        for row in self.rows:
            cache_fields = () if transport_arm == "v1" else getattr(self, "transport_fields", ("transport_cache",))
            required = {
                "sample_id", "split", "target_anchor", "target_anchor_kind",
                "source_images", "camera_angle_x", "transform_matrix",
                "mesh_scale", *cache_fields,
            }
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(f"{row.get('sample_id', '<unknown>')} missing {missing}")
            if expected_split is not None and row["split"] != expected_split:
                raise ValueError(
                    f"manifest contains {row['split']} row but expected {expected_split}"
                )
            if row["target_anchor_kind"] != "geometry_depth":
                raise ValueError("target anchor must be a geometry_depth artifact")
            anchor = self._resolve(row["target_anchor"]).resolve()
            if "render_cond" in anchor.parts:
                raise ValueError("textured render_cond target is forbidden as target anchor")
            if len(row["source_images"]) != 3:
                raise ValueError("each item must have exactly three source images")
            if len(row["camera_angle_x"]) != 4 or len(row["transform_matrix"]) != 4:
                raise ValueError("camera arrays must follow [target,source1,source2,source3]")
            if verify_files:
                checks = [anchor, *(self._resolve(row[field]) for field in cache_fields)]
                if self.match_root is not None and transport_arm != "v1":
                    checks.append(self.match_root / f"{row['sample_id']}.npz")
                if self.projection_root is not None and transport_arm != "v1":
                    checks.append(self.projection_root / f"{row['sample_id']}.npz")
                checks += [self._resolve(p) for p in row["source_images"]]
                missing_files = [str(p) for p in checks if not p.is_file()]
                if missing_files:
                    raise FileNotFoundError(missing_files[0])
            self.instances.append((self.manifest_dir, row["sample_id"], row["split"]))

    def __len__(self) -> int:
        return len(self.rows)

    def __str__(self) -> str:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row["split"]] = counts.get(row["split"], 0) + 1
        return f"{type(self).__name__}(samples={len(self)}, splits={counts})"

    def _common(self, row: dict[str, Any]) -> dict[str, Any]:
        anchor = self._resolve(row["target_anchor"])
        sources = [self._resolve(p) for p in row["source_images"]]
        images = [_load_image(anchor, self.image_size)]
        images.extend(_load_image(path, self.image_size) for path in sources)
        transforms = torch.as_tensor(row["transform_matrix"], dtype=torch.float32)
        angles = torch.as_tensor(row["camera_angle_x"], dtype=torch.float32)
        if transforms.shape != (4, 4, 4) or angles.shape != (4,):
            raise ValueError(f"invalid cameras for {row['sample_id']}")
        distances = torch.linalg.vector_norm(transforms[:, :3, 3], dim=-1)
        pack: dict[str, Any] = {
            "cond": torch.stack(images),
            "camera_angle_x": angles,
            "camera_distance": distances,
            "transform_matrix": transforms,
            "mesh_scale": torch.tensor(float(row["mesh_scale"]), dtype=torch.float32),
            "view_indices": torch.arange(4),
            "_dataset_name": "DeformNeXt",
            "_sha256": row["sample_id"],
        }
        if self.transport_arm == "v1":
            return pack
        cache_fields = getattr(self, "transport_fields", ("transport_cache",))
        for cache_field in cache_fields:
            with np.load(self._resolve(row[cache_field])) as cache:
                for key in self.transport_keys:
                    if key in cache:
                        value = np.asarray(cache[key])
                        tensor = torch.from_numpy(value)
                        if key == "source_valid":
                            tensor = tensor.bool()
                        elif key in {"spatial_indices", "patch_grid", "patch_ids"}:
                            tensor = tensor.long()
                        pack[f"transport_{key}"] = tensor
        if self.match_root is not None and self.transport_arm != "v1":
            with np.load(self.match_root / f"{row['sample_id']}.npz") as cache:
                if self.transport_implementation == "r9-native-v5":
                    from ..modules.r9_matching import MATCHER_CONTRACT, ATTENTION_CONTRACT, RAW_MATCH_SCHEMA, RAW_MATCH_CONTRACT
                    raw = self.transport_arm == 'b1' and str(cache['schema']) == RAW_MATCH_SCHEMA
                    if raw:
                        if (json.loads(str(cache['raw_matching_contract'])) != RAW_MATCH_CONTRACT
                                or str(cache['sample_id']) != row['sample_id']
                                or str(cache['split']) != row['split']
                                or cache['semantic_indices'].shape != (1024, 1)):
                            raise ValueError('incompatible RAW Top-1 match cache contract')
                    elif (str(cache['schema']) != 'deformnext-match-r9-v5'
                            or json.loads(str(cache['matcher_contract'])) != MATCHER_CONTRACT
                            or str(cache['aggregation']) != 'global-categorical-v1'
                            or 'attention_contract' not in cache
                            or json.loads(str(cache['attention_contract'])) != ATTENTION_CONTRACT
                            or str(cache['proposal_policy']) != 'GLOBAL64_VALID_OR_NATIVE'):
                        raise ValueError('incompatible R9 match cache contract')
                    if not raw and (self.expected_q2c_sha256 is None or str(cache['q2c_sha256']) != self.expected_q2c_sha256):
                        raise ValueError('R9 match cache checkpoint hash mismatch')
                for key in MATCH_KEYS_BY_ARM[self.transport_arm]:
                    if key not in cache:
                        raise ValueError(
                            f"match cache for {row['sample_id']} missing {key}"
                        )
                    tensor = torch.from_numpy(np.asarray(cache[key]))
                    if key == "semantic_indices":
                        tensor = tensor.long()
                    elif key == "candidate_valid":
                        tensor = tensor.bool()
                    pack[f"transport_{key}"] = tensor
        if self.transport_implementation == "r9-native-v5":
            from ..modules.r9_matching import frame_raw
            for key in ('target_keys', 'source_keys'):
                full = f'transport_{key}'
                if full in pack:
                    if self.transport_arm == 'c2' and self.match_root is None:
                        pack['transport_' + key.replace('_keys', '_context')] = pack[full]
                    pack[full] = frame_raw(pack[full])
            # This mask is part of the input geometry-only anchor, not a GT
            # target RGB, M0 topology label, or source visibility oracle.
            with Image.open(self._resolve(row['target_anchor'])) as anchor:
                alpha = np.asarray(anchor.convert('RGBA').resize((512, 512), Image.Resampling.NEAREST))[..., 3]
                pack['transport_target_valid'] = torch.from_numpy((alpha[8::16, 8::16] > 127).copy()).flatten()
            if self.match_root is None:
                # Derive C2 context from the selected FRAME tokens online.
                pack.pop('transport_deformation', None)
        if self.projection_root is not None:
            with np.load(self.projection_root / f"{row['sample_id']}.npz") as cache:
                if str(cache['schema']) != 'deformnext-anchor-projection-v2':
                    raise ValueError('projection cache has incompatible coordinate schema')
                key = 'patch_ids' if 'transport_cache_pbr' in self.transport_fields else 'patch_grid'
                pack[f'transport_{key}'] = torch.from_numpy(np.asarray(cache[key])).long()
        mandatory = {"source_values", "source_valid"}
        if self.transport_arm != "v1" and self.match_root is None:
            mandatory.update(("target_keys", "source_keys"))
        if self.transport_arm in {"c1", "c2"}:
            mandatory.add("target_query")
        absent = sorted(f"transport_{key}" for key in mandatory if f"transport_{key}" not in pack)
        if absent:
            raise ValueError(f"transport cache for {row['sample_id']} missing {absent}")
        if "transport_patch_grid" not in pack and "transport_patch_ids" not in pack:
            raise ValueError(f"transport cache for {row['sample_id']} has no patch mapping")
        return pack


class DeformNeXtSS64(SparseStructureLatentVisMixin, _DeformNeXtBase):
    def __init__(
        self,
        manifest: str,
        *,
        image_size: int = 512,
        normalization: Optional[dict] = None,
        expected_split: Optional[str] = None,
        verify_files: bool = True,
        path_map: Optional[dict[str, str]] = None,
        transport_arm: Optional[str] = None,
        match_root: Optional[str] = None,
        projection_root: Optional[str] = None,
        expected_ss_schema: Optional[str] = None,
        transport_implementation: str = "legacy-v4",
        expected_q2c_sha256: Optional[str] = None,
        pretrained_ss_dec: str = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
        **_: Any,
    ) -> None:
        Dataset.__init__(self)
        self.transport_fields = ("transport_cache",)
        self.normalization = normalization
        self.expected_ss_schema = expected_ss_schema
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = None
        self.ss_dec_ckpt = None
        self.ss_dec = None
        if normalization is not None:
            self.mean = torch.tensor(normalization["mean"]).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(normalization["std"]).reshape(-1, 1, 1, 1)
        self._init_manifest(
            manifest, image_size=image_size, expected_split=expected_split,
            verify_files=verify_files, path_map=path_map, transport_arm=transport_arm,
            match_root=match_root, projection_root=projection_root,
            transport_implementation=transport_implementation, expected_q2c_sha256=expected_q2c_sha256,
        )
        if self.expected_ss_schema is not None:
            for row in self.rows:
                if row.get('ss_latent_schema') != self.expected_ss_schema:
                    raise ValueError('SS manifest does not identify the required canonical latent schema')

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        pack = self._common(row)
        if self.transport_arm != "v1" and "transport_patch_grid" not in pack:
            raise ValueError("SS64 transport requires patch_grid")
        z = _load_array(self._resolve(row["ss_latent"]), "z", self.expected_ss_schema)
        if self.normalization is not None:
            z = (z - self.mean) / self.std
        pack["x_0"] = z
        return pack

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]], split_size=None):
        if split_size is not None:
            raise ValueError("SS64 dense collation does not use split_size")
        return default_collate(batch)


class DeformNeXtPBR1024(SLatPbrVisMixin, _DeformNeXtBase):
    def __init__(
        self,
        manifest: str,
        *,
        image_size: int = 1024,
        resolution: int = 1024,
        max_tokens: int = 32768,
        pbr_slat_normalization: Optional[dict] = None,
        shape_slat_normalization: Optional[dict] = None,
        expected_split: Optional[str] = None,
        verify_files: bool = True,
        path_map: Optional[dict[str, str]] = None,
        transport_arm: Optional[str] = None,
        match_root: Optional[str] = None,
        projection_root: Optional[str] = None,
        transport_implementation: str = "legacy-v4",
        expected_q2c_sha256: Optional[str] = None,
        attrs: Iterable[str] = ("base_color", "metallic", "roughness", "alpha"),
        **_: Any,
    ) -> None:
        Dataset.__init__(self)
        # Omega keys/validity/spatial labels are shared with SS.  The PBR sidecar
        # contains only its 2048-D DINO+NAF values, query and sparse mapping.
        self.transport_fields = ("transport_cache", "transport_cache_pbr")
        self.resolution = resolution
        self.max_tokens = max_tokens
        self.pbr_slat_normalization = pbr_slat_normalization
        self.shape_slat_normalization = shape_slat_normalization
        self.pbr_slat_dec = self.shape_slat_dec = None
        self.pretrained_pbr_slat_dec = "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16"
        self.pretrained_shape_slat_dec = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
        self.pbr_slat_dec_path = self.pbr_slat_dec_ckpt = None
        self.shape_slat_dec_path = self.shape_slat_dec_ckpt = None
        if pbr_slat_normalization is not None:
            self.pbr_slat_mean = torch.tensor(pbr_slat_normalization["mean"]).reshape(1, -1)
            self.pbr_slat_std = torch.tensor(pbr_slat_normalization["std"]).reshape(1, -1)
        if shape_slat_normalization is not None:
            self.shape_slat_mean = torch.tensor(shape_slat_normalization["mean"]).reshape(1, -1)
            self.shape_slat_std = torch.tensor(shape_slat_normalization["std"]).reshape(1, -1)
        channels = {"base_color": 3, "metallic": 1, "roughness": 1, "emissive": 3, "alpha": 1}
        self.layout = {}
        start = 0
        for attr in attrs:
            self.layout[attr] = slice(start, start + channels[attr])
            start += channels[attr]
        self._init_manifest(
            manifest, image_size=image_size, expected_split=expected_split,
            verify_files=verify_files, path_map=path_map, transport_arm=transport_arm,
            match_root=match_root, projection_root=projection_root,
            transport_implementation=transport_implementation, expected_q2c_sha256=expected_q2c_sha256,
        )
        # Match the production PBR dataset contract: assets above max_tokens are
        # excluded before sampling.  DeformNeXt manifests created before the PBR
        # latent pass do not contain token counts, so materialize the count from
        # the NPZ header once and retain it on the in-memory row.
        retained = []
        for row, instance in zip(self.rows, self.instances):
            tokens = row.get("pbr_tokens")
            # Versioned geometry can change sparse support cardinality. Legacy
            # manifest counts must not decide filtering after re-encoding.
            if tokens is None or row.get('latent_geometry_schema') is not None:
                with np.load(self._resolve(row["pbr_latent"])) as archive:
                    tokens = len(archive["coords"])
                row["pbr_tokens"] = int(tokens)
            if int(tokens) <= self.max_tokens:
                retained.append((row, instance))
        self.filtered_over_max_tokens = len(self.rows) - len(retained)
        self.rows = [row for row, _ in retained]
        self.instances = [instance for _, instance in retained]
        self.loads = [int(row["pbr_tokens"]) for row in self.rows]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        pack = self._common(row)
        schema = row.get('latent_geometry_schema')
        coords, pbr = _latent_arrays(self._resolve(row["pbr_latent"]), schema, row['mesh_scale'])
        shape_coords, shape = _latent_arrays(self._resolve(row["shape_latent"]), schema, row['mesh_scale'])
        if not torch.equal(coords, shape_coords):
            raise ValueError(f"PBR/shape sparse support differs for {row['sample_id']}")
        if len(coords) > self.max_tokens:
            raise ValueError(f"PBR token count exceeds max_tokens for {row['sample_id']}")
        if self.pbr_slat_normalization is not None:
            pbr = (pbr - self.pbr_slat_mean) / self.pbr_slat_std
        if self.shape_slat_normalization is not None:
            shape = (shape - self.shape_slat_mean) / self.shape_slat_std
        pack.update(coords=coords, pbr_feats=pbr, shape_feats=shape)
        if self.transport_arm != "v1" and "transport_patch_ids" not in pack:
            raise ValueError("PBR1024 transport requires patch_ids")
        if self.transport_arm != "v1" and len(pack["transport_patch_ids"]) != len(coords):
            raise ValueError(f"PBR patch mapping/support mismatch for {row['sample_id']}")
        return pack

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]], split_size=None):
        groups = ([list(range(len(batch)))] if split_size is None else
                  _bounded_load_groups([len(item["coords"]) for item in batch], split_size))
        outputs = []
        for group in groups:
            items = [batch[index] for index in group]
            coords = []
            layout = []
            start = 0
            for batch_index, item in enumerate(items):
                count = len(item["coords"])
                coords.append(torch.cat((
                    torch.full((count, 1), batch_index, dtype=torch.int32),
                    item["coords"],
                ), dim=-1))
                layout.append(slice(start, start + count))
                start += count
            all_coords = torch.cat(coords)
            pack: dict[str, Any] = {}
            for name, key in (("x_0", "pbr_feats"), ("concat_cond", "shape_feats")):
                sparse = SparseTensor(
                    feats=torch.cat([item[key] for item in items]),
                    coords=all_coords.clone(),
                )
                sparse._shape = torch.Size([len(items), items[0][key].shape[-1]])
                sparse.register_spatial_cache("layout", layout)
                pack[name] = sparse
            for key in items[0]:
                if key in {"coords", "pbr_feats", "shape_feats"}:
                    continue
                values = [item[key] for item in items]
                if key == "transport_patch_ids":
                    pack[key] = torch.cat(values)
                else:
                    pack[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
            outputs.append(pack)
        return outputs[0] if split_size is None else outputs
