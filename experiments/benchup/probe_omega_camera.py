"""Diagnostic-only VGGT-Omega camera/depth predictions from explicit source views."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', nargs='+', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if len(a.images) < 2 or not torch.cuda.is_available():
        raise ValueError('at least two images and an explicitly assigned CUDA device required')
    if len(set(a.images)) != len(a.images) or not all(x.is_file() for x in a.images):
        raise ValueError('input images must be unique files')
    model = VGGTOmega().eval()
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.cuda()
    images = load_and_preprocess_images([str(x) for x in a.images], image_resolution=512).cuda()
    with torch.inference_mode():
        pred = model(images)
        ext, intr = encoding_to_camera(pred['pose_enc'], pred['images'].shape[-2:])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, extrinsics=ext.float().cpu().numpy(),
                        intrinsics=intr.float().cpu().numpy(), depth=pred['depth'].float().cpu().numpy(),
                        depth_conf=pred['depth_conf'].float().cpu().numpy())
    info = {'status': 'diagnostic_not_calibrated', 'source_only': True,
            'checkpoint_sha256': digest(a.checkpoint),
            'images': [{'path': str(x), 'sha256': digest(x)} for x in a.images],
            'output': str(a.output), 'output_sha256': digest(a.output)}
    a.output.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
