"""Native saved-weight SS appearance-off evaluation with learned DQ retained."""
from pathlib import Path
from contextlib import ExitStack
import argparse
import hashlib
import json
import sys


def main():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument('--expected-control-sha', required=True)
    p.add_argument('--expected-adapter-sha', required=True)
    local, remaining = p.parse_known_args()
    from pixal3d.modules import direct_ray_retargeting as module
    from . import appearance_ray_control as control
    from . import evaluate_direct_ray_retargeting as native
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def guard():
        if sha(__file__) != local.expected_adapter_sha or sha(control.__file__) != local.expected_control_sha:
            raise ValueError('appearance-only intervention source changed')

    guard()
    original, previous = module.DirectRayRetargetingConditioner, sys.argv
    events, instances = [], []
    with ExitStack() as contexts:
        class Controlled(original):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                instances.append(self)
                contexts.enter_context(control.appearance_path_off(self, events))

        module.DirectRayRetargetingConditioner = Controlled
        try:
            sys.argv = [str(native.__file__), *remaining]
            native.main()
        finally:
            module.DirectRayRetargetingConditioner = original
            sys.argv = previous
    guard()
    path = Path(remaining[remaining.index('--output')+1])/'evaluation.json'
    result = json.loads(path.read_text())
    if (len(instances) != 1 or result['optimizer_updates_during_evaluation'] != 0
            or not any(row.get('DQ_projection_output_RMS', 0.) > 0 for row in events)):
        raise ValueError('actual learned nonzero DQ path must execute while appearance is muted')
    result['diagnostic_only'] = True
    result['ray_deformation_path_control'] = dict(name='appearance-path-off',
        definition='Mute added transformed appearance output only;keep learned matching probabilities,RAW difference and nonzero DQ projection,old Top1 carrier and NN/R weights.',
        events=events, source_SHA256={str(Path(__file__).resolve()): local.expected_adapter_sha,
                                    str(Path(control.__file__).resolve()): local.expected_control_sha},
        no_fake_zero_observations=True, physical_DQ_channels_retained=True,
        not_retrained_module_ablation=True)
    path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
