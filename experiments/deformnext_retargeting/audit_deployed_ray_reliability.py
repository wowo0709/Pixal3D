"""Check the CUDA deployment port against the actual fixed7 VAL predictions.

The pilot used CPU/NumPy inference feature assembly and a CUDA head. This
probe uses the deployment Torch feature assembly with the same native
cached observations. No threshold, checkpoint or parameter is selected.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from pixal3d.modules.ray_correspondence_reliability import (
    RayCorrespondenceReliability, reliability_features)
from .ray_correspondence_reliability import binary_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '2'
            or '2' not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')):
        raise ValueError('assigned physical GPU2 required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.03, 0)
    torch.use_deterministic_algorithms(True)
    plan = json.loads(args.plan.read_text())
    assert all(sha(p) == d for p, d in plan['source_SHA256'].items())
    cached = Path(plan['cache'])
    root = Path(plan['reliability_root'])
    pilot = json.loads((root/'fit-fixed7-v1/evaluation.json').read_text())
    known = {r['sample_id']: r for r in pilot['per_asset']}
    meta = json.loads((root/'cache-v1/cache.json').read_text())
    feature_index = {r['sample_id']: r for r in meta['records'] if r['role'] == 'validation'}
    stored = np.load(root/'cache-v1/observations.npz', allow_pickle=False)
    rows = list(map(json.loads, (cached/'cache-index.jsonl').read_text().splitlines()))
    model = RayCorrespondenceReliability().cuda().eval()
    model.load_pilot(root/'fit-fixed7-v1/head.pt', plan['head_SHA256'])
    assert len(rows) == len(known) == 32
    records, labels, probability, expected = [], [], [], []
    with torch.no_grad():
        for row in rows:
            path = cached/row['inputs_relative_path']
            assert sha(path) == row['inputs_SHA256']
            with np.load(path, allow_pickle=False) as data:
                arrays = [data[k].copy() for k in ['native_query', 'context_query',
                    'scalar_prior_logits', 'candidate_valid', 'candidate_indices']]
            features, p = [], []
            for begin in range(0, len(arrays[0]), 256):
                inputs = [torch.from_numpy(a[begin:begin+256]).cuda() for a in arrays]
                features.append(reliability_features(*inputs).cpu().numpy())
                p.append(model(*inputs).cpu().numpy())
            actual_x = np.concatenate(features) if features else np.empty((0, 2060), np.float32)
            actual_p = np.concatenate(p) if p else np.empty(0, np.float32)
            teacher = Path(plan['teacher_root'])/(row['sample_id']+'.npz')
            with np.load(path, allow_pickle=False) as data, np.load(teacher, allow_pickle=False) as gt:
                foreground = gt['target_foreground'][data['query_patch_indices']].astype(bool)
                assert str(gt['input_sha256']) == row['GT_teacher_fingerprint']
            old = known[row['sample_id']]
            saved_x = stored[feature_index[row['sample_id']]['array_prefix']+'_x']
            assert old['source_cluster_id'] == row['source_cluster_id'] and len(saved_x) == int(foreground.sum())
            feature_delta = np.abs(actual_x[foreground]-saved_x)
            prob_delta = np.abs(actual_p[foreground]-np.array(old['probability']))
            maximum = float(prob_delta.max()) if len(prob_delta) else 0.
            assert maximum <= plan['maximum_probability_tolerance'], (row['sample_id'], maximum)
            records.append(dict(sample_id=row['sample_id'], rays=len(prob_delta),
                maximum_feature_difference=float(feature_delta.max()) if feature_delta.size else 0.,
                maximum_probability_difference=maximum))
            labels.extend(old['labels'])
            probability.extend(actual_p[foreground].tolist())
            expected.extend(old['probability'])
    threshold = pilot['calibration_selection']['threshold']
    metric = binary_metrics(labels, probability, threshold)
    assert abs(metric['AUROC']-pilot['actual_validation']['AUROC']) <= .001
    assert abs(metric['Brier']-pilot['actual_validation']['Brier']) <= .0001
    assert not any(p.requires_grad for p in model.parameters())
    assert all(sha(p) == d for p, d in plan['source_SHA256'].items())
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='PASS_ACTUAL_CUDA_DEPLOYED_RELIABILITY_PORT', assets=32,
        known_rays=len(labels), native_query_partition=256,
        frozen_actual_checkpoint=True, optimizer_updates=0, GT_in_head_forward=False,
        per_asset=records, deployed_validation=metric,
        original_validation=pilot['actual_validation'],
        maximum_probability_difference=float(np.max(np.abs(np.array(probability)-np.array(expected)))),
        threshold_classification_changes=int(((np.array(probability) >= threshold)
            != (np.array(expected) >= threshold)).sum()),
        probability_tolerance=plan['maximum_probability_tolerance'],
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        source_SHA256=plan['source_SHA256'], generated_quality_improvement_claimed=False)
    (output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['status', 'known_rays',
        'maximum_probability_difference', 'threshold_classification_changes', 'deployed_validation']}))


if __name__ == '__main__':
    main()
