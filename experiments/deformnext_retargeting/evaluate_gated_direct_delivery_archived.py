"""Capture actual gated final SS predictions during their original evaluation.

The sampler and decode run once. Saved GT arrays are scoring artifacts, while
predicted latents/occupancies are retained for a possible selected cascade.
This is an original evaluation capture, not replay parity against a past run.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .evaluate_ss_artifact_subset import capture_ss_artifacts


def digest_array(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, str(value.dtype))).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    from . import quick_metrics as quick
    from . import evaluate_gated_direct_delivery as native
    from .cascade_view_roles import file_sha256 as sha
    output = Path(sys.argv[sys.argv.index('--output')+1])
    if output.exists():
        raise FileExistsError('fresh gated evaluation/archive namespace required')
    source = {str(Path(p).resolve()):sha(p) for p in [__file__, native.__file__, quick.__file__]}
    captured = []
    with capture_ss_artifacts(quick, output, captured):
        native.main()
    path = output/'evaluation.json'; actual = json.loads(path.read_text())
    if (len(captured) != len(actual['per_asset']) or len(captured) != 128
            or actual['optimizer_updates_during_evaluation'] != 0):
        raise ValueError('all128 original frozen gated predictions must be captured')
    for artifact, row in zip(captured, actual['per_asset']):
        if (artifact['sample_id'] != row['sample_id'] or artifact['seed'] != row['seed']
                or artifact['target_voxels_SHA256'] != row['target_voxels_sha256']
                or any(artifact['occupancy_counts'][k] != row[k] for k in ['tp','fp','fn'])):
            raise ValueError('actual archive changed sample/GT/seed/occupancy counts')
        with np.load(output/artifact['artifact']) as data:
            artifact.update(prediction_latent_SHA256=digest_array(data['prediction_latent']),
                            target_latent_SHA256=digest_array(data['target_latent']))
    if any(sha(p) != value for p,value in source.items()):
        raise ValueError('gated archive source changed during original evaluation')
    actual['native_SS_artifacts'] = dict(status='PASS_ACTUAL_ORIGINAL_GATED_SS128_PREDICTION_CAPTURE',
        archived_asset_count=128, same_forward_prediction_and_target_capture=True,
        artifacts=captured, source_SHA256=source, optimizer_updates=0,
        separate_historical_reference_replay_not_claimed=True,
        predicted_SS_available_for_selected_cascade=True, GT_SS_is_for_scoring_only=True,
        full_cascade_generated=False)
    path.write_text(json.dumps(actual, indent=2)+'\n')


if __name__ == '__main__':
    main()
