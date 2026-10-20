"""Archive actual native SS outputs without changing a pinned evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--evaluation-module', required=True, choices=[
        'evaluate_deterministic_checkpoint', 'evaluate_consistent_ray_retargeting',
        'evaluate_direct_ray_retargeting'])
    parser.add_argument('--reference-evaluation', type=Path, required=True)
    local, remaining = parser.parse_known_args()
    from . import quick_metrics as quick
    output = Path(remaining[remaining.index('--output')+1])
    if output.exists():
        raise FileExistsError('fresh archive/evaluation namespace required')
    reference = json.loads(local.reference_evaluation.read_text())
    original = quick._ss_metrics
    captured = []

    def archived(trainer, dataset, indices, config):
        original_decode = dataset.decode_latent
        tensors = []

        def decode(latent, *args, **kwargs):
            result = original_decode(latent, *args, **kwargs)
            tensors.append((latent.detach().float().cpu().numpy(),
                            result.detach().float().cpu().numpy()))
            return result

        dataset.decode_latent = decode
        try:
            result = original(trainer, dataset, indices, config)
        finally:
            dataset.decode_latent = original_decode
        if len(tensors) != 1:
            raise ValueError('one native concatenated prediction/target SS decode required')
        latent, decoded = tensors[0]
        count = len(indices)
        if len(latent) != 2*count or decoded.shape != (2*count,1,64,64,64):
            raise ValueError('native predicted/GT SS64 pairs required')
        folder = output/'ss-artifacts'
        folder.mkdir(exist_ok=False)
        seeds = quick.panel_seeds(dataset, indices, config)
        for i, (index, seed) in enumerate(zip(indices, seeds)):
            row = dataset.rows[index]
            name = hashlib.sha256(row['sample_id'].encode()).hexdigest()+'.npz'
            prediction = np.packbits((decoded[i,0]>0).reshape(-1))
            target = np.packbits((decoded[count+i,0]>0).reshape(-1))
            path = folder/name
            np.savez_compressed(path, prediction_latent=latent[i], target_latent=latent[count+i],
                prediction_occupancy_packed=prediction, target_occupancy_packed=target,
                occupancy_shape=np.array([64,64,64], dtype=np.int32),
                sample_id=np.asarray(row['sample_id']), seed=np.int64(seed))
            captured.append(dict(sample_id=row['sample_id'], seed=seed,
                artifact=str(path.relative_to(output)), artifact_SHA256=sha(path),
                prediction_voxels_SHA256=hashlib.sha256(prediction.tobytes()).hexdigest(),
                target_voxels_SHA256=hashlib.sha256(target.tobytes()).hexdigest()))
        return result

    entry = importlib.import_module('.'+local.evaluation_module, __package__)
    bindings = {str(Path(p).resolve()):sha(p) for p in [__file__, quick.__file__, entry.__file__]}
    quick._ss_metrics = archived
    previous = sys.argv
    try:
        sys.argv = [entry.__file__, *remaining]
        entry.main()
    finally:
        quick._ss_metrics = original
        sys.argv = previous
    path = output/'evaluation.json'
    actual = json.loads(path.read_text())
    if (actual['quick_metrics'] != reference['quick_metrics']
            or actual['per_asset'] != reference['per_asset']
            or actual['optimizer_updates_during_evaluation'] != 0
            or len(captured) != len(actual['per_asset'])):
        raise ValueError('archive replay must preserve every previous record and metric exactly')
    for archive, record in zip(captured, actual['per_asset']):
        if (archive['sample_id'] != record['sample_id'] or archive['seed'] != record['seed']
                or archive['target_voxels_SHA256'] != record['target_voxels_sha256']):
            raise ValueError('archive identity or GT occupancy hash differs')
    if any(sha(p) != digest for p,digest in bindings.items()):
        raise ValueError('archive source changed during evaluation')
    actual['native_SS_artifacts'] = dict(
        status='PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_EXACT_REFERENCE_REPLAY',
        original_metrics_and_all_records_exact=True, artifacts=captured,
        reference_evaluation=str(local.reference_evaluation),
        reference_SHA256=sha(local.reference_evaluation), source_SHA256=bindings,
        predicted_SS_is_available_for_full_cascade=True, GT_SS_is_for_scoring_only=True,
        full_cascade_generated=False, optimizer_updates=0)
    path.write_text(json.dumps(actual, indent=2)+'\n')
    print(json.dumps(dict(status=actual['native_SS_artifacts']['status'],assets=len(captured))),flush=True)


if __name__ == '__main__':
    main()
