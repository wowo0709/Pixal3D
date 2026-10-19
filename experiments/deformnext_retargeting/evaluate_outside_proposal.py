"""Read-only SS/PBR added-path values from outside the semantic proposal set."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--expected-outside-source-sha',required=True)
    parser.add_argument('--expected-outside-adapter-sha',required=True)
    local,remaining = parser.parse_known_args()
    from pixal3d.modules import direct_ray_retargeting as module
    from . import outside_proposal_control as controls

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def guard():
        if sha(controls.__file__)!=local.expected_outside_source_sha or sha(__file__)!=local.expected_outside_adapter_sha:
            raise ValueError('declared outside-proposal diagnostic source changed')

    guard()
    for flag in ['--transport-control','--r9-path-control','--task-value-control']:
        if flag in remaining and remaining[remaining.index(flag)+1]!='normal':
            raise ValueError('outside-proposal diagnostic cannot combine other interventions')
    stage = remaining[remaining.index('--stage')+1]
    if stage=='ss64':
        from . import evaluate_direct_ray_retargeting as entry
    elif stage=='pbr1024':
        from . import evaluate_direct_pbr as entry
    else:
        raise ValueError('explicit SS64 or PBR1024 Direct evaluator required')
    original = module.DirectRayRetargetingConditioner
    instances,events = [],[]
    with ExitStack() as contexts:
        class ControlledDirect(original):
            def __init__(self,*args,**kwargs):
                if instances:
                    raise ValueError('one actual Direct conditioner required')
                super().__init__(*args,**kwargs)
                instances.append(self)
                contexts.enter_context(controls.outside_proposal_control(self,events))

        module.DirectRayRetargetingConditioner = ControlledDirect
        previous = sys.argv
        try:
            sys.argv = [str(entry.__file__),*remaining]
            entry.main()
        finally:
            sys.argv = previous
            module.DirectRayRetargetingConditioner = original
    guard()
    if len(instances)!=1 or not events:
        raise ValueError('outside-proposal intervention did not actually execute')
    output = Path(remaining[remaining.index('--output')+1])
    path = output/'evaluation.json'
    result = json.loads(path.read_text())
    if result['optimizer_updates_during_evaluation']!=0:
        raise ValueError('this diagnostic must not update model weights')
    moved = sum(row['reassigned_outside_valid_slots'] for row in events)
    result['diagnostic_only'] = True
    result['outside_proposal_value_control'] = dict(
        status='EXECUTED' if moved else 'NO_OUTSIDE_DONORS_IN_THIS_PANEL',
        reassigned_valid_slots=moved,events=events,
        source_SHA256={str(Path(controls.__file__).resolve()):local.expected_outside_source_sha,
                       str(Path(__file__).resolve()):local.expected_outside_adapter_sha},
        donors='same asset,RGB-valid source patches outside each semantic Top64 set',
        original_Top1_carrier_and_keys_and_scores_and_probabilities_retained=True,
        appearance_and_RAW_values_replaced_together=True,GT_added=False,
        scope='Saved-weight proposal-value diagnostic;not native-spatial NoCorr or a retrained ablation.')
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
