"""Native SS identity/gradient proof for an explicitly calibrated initializer.

An intermediate auxiliary epoch may be used for disposable engineering
checks. It never authorizes scientific training or substitutes for the
fixed7 initializer's own native proof and continuation/capacity checks.
"""
from pathlib import Path
import argparse
import json
import sys

from pixal3d.modules.calibrated_ray_retargeting import (
    CalibratedDirectRayNoQueryConditioner, CalibratedDirectRayRetargetingConditioner, file_sha)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--matching-initializer-json', type=Path, required=True)
    parser.add_argument('--delivery-variant', choices=['Direct-DQ', 'Direct-NoDQ'], required=True)
    local, remaining = parser.parse_known_args()
    declaration = json.loads(local.matching_initializer_json.read_text())
    if (declaration.get('purpose') != 'disposable-native-initializer-check'
            or declaration.get('scientific_training_eligible') is not False
            or declaration.get('fit_epoch') not in range(1, 8)):
        raise ValueError('explicit disposable auxiliary epoch declaration required')
    matching = declaration['calibrated_matching']
    from pixal3d.modules import calibrated_ray_retargeting
    sources = {str(Path(path).resolve()): file_sha(path) for path in [Path(__file__),
        Path(calibrated_ray_retargeting.__file__), local.matching_initializer_json,
        Path(matching['checkpoint']), Path(matching['reference_conditioner'])]}
    if local.delivery_variant == 'Direct-DQ':
        from . import verify_direct_ray_retargeting_gradient as native
        name, parent = 'DirectRayRetargetingConditioner', CalibratedDirectRayRetargetingConditioner
    else:
        from . import verify_ray_feature_ss_gradient as native
        name, parent = 'RayFeatureAttentionConditioner', CalibratedDirectRayNoQueryConditioner
    observations = []

    class Recorded(parent):
        def forward(self, **kwargs):
            if not observations:
                observations.append(self.assert_calibrated_initializer())
            return super().forward(**kwargs)

    def factory(*args, **kwargs):
        return Recorded(*args, calibrated_matching=matching, **kwargs)

    original, previous = getattr(native, name), sys.argv
    setattr(native, name, factory)
    try:
        sys.argv = [str(native.__file__), *remaining]
        native.main()
    finally:
        setattr(native, name, original)
        sys.argv = previous
    output = Path(remaining[remaining.index('--output')+1])/'audit.json'
    result = json.loads(output.read_text())
    if (len(observations) != 1 or any(file_sha(path) != digest for path, digest in sources.items())
            or result['scientific_training_updates'] != 0):
        raise ValueError('actual native initial-state observation and immutable source required')
    if local.delivery_variant == 'Direct-DQ':
        if (not result['initial_evidence_exact'] or not result['all10_new_tensors_receive_actual_SS_gradients']
                or result['initial_loss'] != result['new_initial_loss']):
            raise ValueError('actual calibrated Direct ten-tensor native proof required')
    elif (not result['actual_initial_evidence_exact']
            or not result['nine_feature_tensors_receive_gradient_by_update3']):
        raise ValueError('actual calibrated NoDQ nine-tensor native proof required')
    result['calibrated_matching_initializer'] = dict(variant=local.delivery_variant,
        fit_epoch=declaration['fit_epoch'], scientific_training_eligible=False,
        purpose='engineering-only;fixed7-own-proof-still-required', actual_first_forward=observations[0],
        loaded_calibrated_matching=matching, same_native_R_and_LoRA_reference=True,
        no_observation_zeroing=True)
    result['source_SHA256'].update(sources)
    output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
