"""DEV-only probe: infer canonical scale from permitted RGB/A and cameras.

Training labels are available only for the training split. Validation labels
measure whether an image estimator could replace a target-mesh-derived scale
at inference. No TEST labels or source mesh enter features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor


def features(row: dict, path_prefix: Path) -> list[float]:
    result = []
    for image_path, matrix, fov in zip(
            [row['target_anchor'], *row['source_images']],
            row['transform_matrix'], row['camera_angle_x']):
        path = path_prefix / image_path.removeprefix('/file4/')
        with Image.open(path) as image:
            alpha = image.getchannel('A')
            mask = alpha.point(lambda value: 255 if value >= 128 else 0)
            box = mask.getbbox()
            if box is None:
                raise ValueError('empty image silhouette: ' + str(path))
            width, height = image.size
            pixels = np.asarray(mask, dtype=np.uint8)
        x0, y0, x1, y1 = box
        camera = np.asarray(matrix, dtype=np.float64)
        distance = np.linalg.norm(camera[:3, 3])
        result.extend([(x1-x0)/width, (y1-y0)/height,
                       (x0+x1-width)/width, (y0+y1-height)/height,
                       np.count_nonzero(pixels)/pixels.size,
                       float(distance), float(fov)])
    return result


def population(path: Path, path_prefix: Path, limit: int | None):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if limit is not None:
        rows = rows[:limit]
    X, y = [], []
    for index, row in enumerate(rows, 1):
        X.append(features(row, path_prefix))
        y.append(float(row['mesh_scale']))
        if index % 500 == 0:
            print(json.dumps({'split': path.stem, 'features_done': index,
                              'features_total': len(rows)}), flush=True)
    return rows, np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def scores(pred: np.ndarray, target: np.ndarray) -> dict:
    err = np.abs(pred-target) / target
    return {'median_abs_relative_error': float(np.median(err)),
            'mean_abs_relative_error': float(np.mean(err)),
            'p90_abs_relative_error': float(np.quantile(err, .9)),
            'fraction_above_5pct': float(np.mean(err > .05)),
            'fraction_above_10pct': float(np.mean(err > .1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--file4-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-output', type=Path)
    parser.add_argument('--max-train', type=int)
    args = parser.parse_args()
    if (args.output.exists() or args.model_output is not None and args.model_output.exists()
            or args.max_train is not None and args.max_train < 500):
        raise ValueError('fresh output and adequate training subset required')
    train_rows, X_train, y_train = population(args.train, args.file4_cache, args.max_train)
    val_rows, X_val, y_val = population(args.validation, args.file4_cache, None)
    if set(row['sample_id'] for row in train_rows) & set(row['sample_id'] for row in val_rows):
        raise ValueError('training and validation samples overlap')
    estimator = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15,
                                               l2_regularization=.1, random_state=0)
    estimator.fit(X_train, np.log(y_train))
    predicted = np.exp(estimator.predict(X_val))
    if args.model_output is not None:
        joblib.dump({'estimator': estimator,
                     'feature_schema': 'RGBA128-mask-bbox-area-camera-v1',
                     'training_manifest': str(args.train),
                     'training_sample_ids': [row['sample_id'] for row in train_rows]},
                    args.model_output)
    result = {'status': 'DEV_IMAGE_SCALE_PROBE_ONLY', 'train_samples': len(train_rows),
              'validation_samples': len(val_rows), 'features': 'four RGBA silhouettes and known cameras',
              'source_M0_mesh_or_test_label_used_for_estimation': False,
              'GT_M1_scale_used_as_training_label_only': True,
              'model_output': str(args.model_output) if args.model_output else None,
              'baseline_constant1': scores(np.ones_like(y_val), y_val),
              'image_estimator': scores(predicted, y_val),
              'per_asset': [{'sample_id': row['sample_id'], 'predicted_scale': float(p),
                             'validation_label_scale': float(y)}
                            for row, p, y in zip(val_rows, predicted, y_val)]}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'train_samples',
         'validation_samples', 'baseline_constant1', 'image_estimator')}), flush=True)


if __name__ == '__main__':
    main()
