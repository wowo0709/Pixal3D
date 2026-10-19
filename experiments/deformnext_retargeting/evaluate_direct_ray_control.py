"""Explicit read-only feature-path interventions around the frozen SS evaluator."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--deformation-path-control',required=True,
        choices=['wrong-value-address','score-prior-off','decoder-context-bypass','added-path-off','DQ-projection-off'])
    parser.add_argument('--expected-control-sha',required=True)
    parser.add_argument('--expected-adapter-sha',required=True)
    local,remaining=parser.parse_known_args()
    from pixal3d.modules import direct_ray_retargeting as module
    from . import ray_deformation_controls as controls
    from . import evaluate_direct_ray_retargeting as entry

    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def guard():
        if sha(controls.__file__)!=local.expected_control_sha or sha(__file__)!=local.expected_adapter_sha:
            raise ValueError('declared feature-control source changed')
    guard()
    for name in ['--task-value-control','--r9-path-control','--transport-control']:
        if name in remaining and remaining[remaining.index(name)+1]!='normal':
            raise ValueError('feature intervention cannot be combined with another control')
    original=module.DirectRayRetargetingConditioner
    events=[];instances=[]
    with ExitStack() as contexts:
        class ControlledFeatureConditioner(original):
            def __init__(self,*args,**kwargs):
                if instances:raise ValueError('one feature conditioner expected per evaluation')
                super().__init__(*args,**kwargs)
                instances.append(self)
                contexts.enter_context(controls.deformation_control(self,local.deformation_path_control,events))
        # This intervention is explicitly recorded in the output. Original
        # forward source, saved weights and every training process stay intact.
        module.DirectRayRetargetingConditioner=ControlledFeatureConditioner
        previous=sys.argv
        try:
            sys.argv=[str(entry.__file__),*remaining]
            entry.main()
        finally:
            sys.argv=previous
            module.DirectRayRetargetingConditioner=original
    guard()
    if len(instances)!=1 or not events:
        raise ValueError('feature intervention did not actually execute')
    output=Path(remaining[remaining.index('--output')+1]);path=output/'evaluation.json'
    result=json.loads(path.read_text())
    if result['optimizer_updates_during_evaluation']!=0:
        raise ValueError('saved-weight control must never update a model')
    result['diagnostic_only']=True
    result['ray_deformation_path_control']=dict(name=local.deformation_path_control,
        definition=controls.DEFINITIONS[local.deformation_path_control],events=events,
        source_SHA256={str(Path(controls.__file__).resolve()):local.expected_control_sha,
                       str(Path(__file__).resolve()):local.expected_adapter_sha},
        cached_Top1_carrier_retained=True,GT_added=False,
        scope='Saved-weight feature-path intervention;not a retrained ablation or the user NoScore/NoCorr/physically-absent-NoDQ arm.')
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
