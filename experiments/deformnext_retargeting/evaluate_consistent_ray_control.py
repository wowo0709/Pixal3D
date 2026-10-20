"""Pinned saved-weight diagnostic; preserve the native model and RGB inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .consistent_ray_controls import MODES, saved_weight_control


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--saved-weight-control', choices=MODES, required=True)
    local, remaining = parser.parse_known_args()
    if remaining[remaining.index('--stage')+1] != 'ss64':
        raise ValueError('only the verified SS control path is supported')
    output = Path(remaining[remaining.index('--output')+1])
    from . import evaluate_consistent_ray_retargeting as native
    previous = sys.argv
    try:
        sys.argv = [str(Path(native.__file__)), *remaining]
        with saved_weight_control(local.saved_weight_control) as audit:
            native.main()
    finally:
        sys.argv = previous
    if any(audit[k] < 1 for k in ['readout_calls', 'correction_calls', 'reliability_calls']):
        raise ValueError('the actual learned candidate path was not executed')
    path = output/'evaluation.json'
    record = json.loads(path.read_text())
    if record['optimizer_updates_during_evaluation'] != 0:
        raise ValueError('saved-weight diagnostics cannot update a model')
    record['diagnostic_only'] = True
    record['consistent_saved_weight_control'] = dict(
        **audit, saved_weight_diagnostic=True, retrained_ablation=False,
        original_Top1_R_carrier_unchanged=True, native_RGB_and_GT_noise_protocol_unchanged=True,
        physical_deformation_channels=2048, channel_removal_claimed=False,
        NoScore_payload_scope='native-top1 collapses the added TopK path to native semantic Top1',
        score_prior_off_is_NOT_user_NoScore=True, confidence_input_held_fixed_for_attention_controls=True,
        decoder_bypass_scope='only alpha gets raw Omega target key; reliability retains decoded D1 context',
        source_SHA256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(),Path(__file__).with_name('consistent_ray_controls.py')]})
    path.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
