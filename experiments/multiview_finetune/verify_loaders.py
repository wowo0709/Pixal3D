"""CPU loader checks for the native full-EBS and K=1/6 tensor contracts."""
import argparse
import functools
import json
import os
import resource
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from configure import REPO, STAGES

sys.path.insert(0, str(REPO))


def check_pack(pack, image_size):
    b, k, c, h, w = pack["cond"].shape  # [B,K,3,H,W]
    assert 1 <= k <= 6 and c == 3 and h == w == image_size
    assert pack["transform_matrix"].shape == (b, k, 4, 4)
    assert torch.equal(pack["view_indices"][:, 0], torch.as_tensor(pack["view_idx"]))
    assert torch.isfinite(pack["cond"]).all()
    latent = pack["x_0"]
    values = latent if isinstance(latent, torch.Tensor) else latent.feats
    assert torch.isfinite(values).all()
    if "concat_cond" in pack:
        assert torch.equal(latent.coords, pack["concat_cond"].coords)
    return b, k


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--roots-dir", type=Path, help="Override train mappings directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs-dir", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

    with patch.object(torch.cuda, "get_device_name", return_value="A100"):
        from pixal3d import datasets
        CLASSES = {name: getattr(datasets, name) for name in (
            "MultiViewImageConditionedSparseStructureLatentView",
            "MultiViewImageConditionedSLatShapeView", "MultiViewImageConditionedSLatPbrView")}
    torch.set_num_threads(2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    report = {}
    memberships = {}
    for stage in STAGES:
        config = json.loads((args.configs_dir / (stage + ".json")).read_text())
        dataset = CLASSES[config["dataset"]["name"]](
            ((args.roots_dir / ("train_" + stage + "_roots.json")).read_text()
             if args.roots_dir else config["data_dir"]),
            **config["dataset"]["args"])
        memberships[stage] = {(name, sha) for _, sha, name in dataset.instances}
        indices = {}
        for index, (_, _, name) in enumerate(dataset.instances):
            indices.setdefault(name.split(":")[0], index)
        endpoints = []
        for source, index in indices.items():
            for k in (1, 6):
                dataset.min_condition_views = dataset.max_condition_views = k
                pack = dataset.collate_fn([dataset[index]])
                pack = pack if isinstance(pack, dict) else pack[0]
                _, actual_k = check_pack(pack, config["dataset"]["args"]["image_size"])
                assert actual_k == k
                endpoints.append({"source": source, "K": k})
        dataset.min_condition_views, dataset.max_condition_views = 1, 6
        trainer = config["trainer"]["args"]
        collate = dataset.collate_fn if stage == "ss64" else functools.partial(dataset.collate_fn, split_size=trainer["batch_split"])
        loader = DataLoader(dataset, batch_size=trainer["batch_size_per_gpu"],
            shuffle=True, num_workers=2, persistent_workers=False,
            collate_fn=collate, drop_last=True, prefetch_factor=1)
        iterations = []
        for batch in loader:
            if isinstance(batch, dict):
                sizes = [int(batch["cond"].shape[0])]
                views = [int(batch["cond"].shape[1])]
                check_pack(batch, config["dataset"]["args"]["image_size"])
            else:
                pairs = [check_pack(pack, config["dataset"]["args"]["image_size"]) for pack in batch]
                sizes, views = zip(*pairs)
            assert sum(sizes) == trainer["batch_size_per_gpu"] and len(set(views)) == 1
            iterations.append({"assets": sum(sizes), "K": views[0]})
            if len(iterations) == 3:
                break
        report[stage] = {"eligible_assets": len(dataset), "sources": list(indices),
                         "endpoint_checks": endpoints, "loader_iterations": iterations}
        print(stage, json.dumps(report[stage]), flush=True)
        del loader, dataset
    assert all(x == memberships["ss64"] for x in memberships.values())
    assert not torch.cuda.is_initialized()
    args.output.write_text(json.dumps({"stages": report, "common_membership": True,
                                      "cuda_initialized": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
