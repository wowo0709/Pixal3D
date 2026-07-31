from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


REPO_ID = "TencentARC/Pixal3D"
OUTPUT = Path("/root/node17/data/pixal3d/train/checkpoints/single_view")
CHECKPOINTS = {
    "ss_flow_img_dit_1_3B_64_bf16": "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    "slat_flow_img2shape_dit_1_3B_512_bf16": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    "slat_flow_img2shape_dit_1_3B_1024_bf16": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    "slat_flow_imgshape2tex_dit_1_3B_1024_bf16": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
}


def convert_checkpoint(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    state_dict = load_file(str(source), device="cpu")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, target)
    reloaded = torch.load(target, map_location="cpu", weights_only=True)
    if reloaded.keys() != state_dict.keys():
        raise RuntimeError(f"checkpoint keys changed during conversion: {source}")
    for key in state_dict:
        if not torch.equal(reloaded[key], state_dict[key]):
            raise RuntimeError(f"checkpoint tensor changed during conversion: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for stem, filename in CHECKPOINTS.items():
        source = Path(hf_hub_download(repo_id=args.repo_id, filename=filename))
        target = args.output_root / f"{stem}.pt"
        convert_checkpoint(source, target)
        print(target)


if __name__ == "__main__":
    main()
