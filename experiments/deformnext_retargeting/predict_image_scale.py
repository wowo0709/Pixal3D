"""Freeze image-inferred object scales without opening geometry or scale labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .probe_image_estimated_scale import features


INPUT_FIELDS = ('target_anchor', 'source_images', 'transform_matrix', 'camera_angle_x')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--file4-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-samples', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.expected_samples < 1:
        raise ValueError('fresh output and explicit population required')
    saved = joblib.load(args.model)
    if saved.get('feature_schema') != 'RGBA128-mask-bbox-area-camera-v1':
        raise ValueError('frozen image scale feature schema differs')
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    ids = [row['sample_id'] for row in rows]
    if (len(ids) != args.expected_samples or len(set(ids)) != len(ids)
            or set(ids) & set(saved['training_sample_ids'])):
        raise ValueError('wrong population or training inference overlap')
    results = []
    for row in rows:
        # Explicitly narrow the feature input before extraction. M0/M1 meshes,
        # latent sidecars, depth, target labels and manifest mesh_scale are absent.
        permitted = {key: row[key] for key in INPUT_FIELDS}
        vector = np.asarray(features(permitted, args.file4_cache), dtype=np.float32)[None]
        scale = float(np.exp(saved['estimator'].predict(vector)[0]))
        if not .4 < scale < 1.6:
            raise ValueError('image scale outside frozen model contract: ' + row['sample_id'])
        results.append({'sample_id': row['sample_id'], 'predicted_mesh_scale': scale,
                        'input_modalities': 'target and three source RGB/A with calibrated cameras',
                        'M0_or_M1_mesh_or_GT_scale_used_at_inference': False})
    args.output.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in results))
    receipt = {'status': 'PASS_IMAGE_ONLY_SCALE_PREDICTIONS',
               'samples': len(results), 'model_sha256': digest(args.model),
               'manifest_sha256': digest(args.manifest), 'output_sha256': digest(args.output),
               'train_inference_sample_overlap': 0,
               'source_M0_and_target_M1_mesh_not_opened': True,
               'GT_instance_scale_not_read_into_feature_vector': True,
               'scale_min': min(row['predicted_mesh_scale'] for row in results),
               'scale_max': max(row['predicted_mesh_scale'] for row in results)}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
