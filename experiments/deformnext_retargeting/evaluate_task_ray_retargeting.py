"""Explicit read-only TaskRay evaluation; old proof covers the common R only."""
import argparse,json,sys
from pathlib import Path

import torch

from .task_ray_science_contract import closed_json,digest


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--task-native-probe',type=Path,required=True)
    p.add_argument('--task-native-probe-receipt',type=Path,required=True)
    p.add_argument('--task-campaign-root',type=Path,required=True)
    p.add_argument('--task-phase',choices=['initial','trained'],required=True)
    local,remaining=p.parse_known_args()
    proof=closed_json(local.task_native_probe,local.task_native_probe_receipt,local.task_campaign_root)
    if (proof['status']!='PASS_ACTUAL_BOUNDED_TASK_RAY369_NATIVE_PROFILE'
            or proof['actual_optimizer_states']!=369
            or not proof['new_attention_receives_actual_SS_gradient_not_only_KL']):
        raise ValueError('actual native TaskRay learnability proof required')
    checkpoint=Path(remaining[remaining.index('--conditioner-checkpoint')+1])
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if len(state)!=46 or not {'task_attention.query.weight','task_attention.key.weight'}<=set(state):
        raise ValueError('complete46-state TaskRay conditioner required')
    if local.task_phase=='initial' and (torch.count_nonzero(state['task_attention.query.weight'])
            or torch.count_nonzero(state['retargeter.deformation_projection.weight'])):
        raise ValueError('zero initial Q and geometry projections required')
    from pixal3d.modules.task_ray_retargeting import TaskRayRetargetingConditioner
    from pixal3d.modules import ray_consistent_retargeting as model_slot
    from . import evaluate_consistent_ray_retargeting as common
    original=model_slot.RayConsistentRetargetingConditioner
    bindings={str(Path(q).resolve()):digest(q) for q in [__file__,common.__file__]}
    model_slot.RayConsistentRetargetingConditioner=TaskRayRetargetingConditioner
    previous=sys.argv
    try:
        sys.argv=[common.__file__,*remaining]
        common.main()
    finally:
        sys.argv=previous
        model_slot.RayConsistentRetargetingConditioner=original
    path=Path(remaining[remaining.index('--output')+1])/'evaluation.json'
    result=json.loads(path.read_text())
    if result['optimizer_updates_during_evaluation']!=0 or any(digest(q)!=s for q,s in bindings.items()):
        raise ValueError('read-only TaskRay evaluation/source preservation failed')
    result['task_ray_evaluation']=dict(actual_conditioner_class='TaskRayRetargetingConditioner',
        phase=local.task_phase,conditioner_state_tensors=46,query_channels=4096,
        actual_deformation_query_channels=2048,semantic7_frozen=True,reliability4_frozen=True,
        task_attention2_frozen_for_evaluation=True,old_identity_proof_covers_common_R_only=True,
        own_native_TaskRay_probe=str(local.task_native_probe),process_local_constructor_binding=True,
        shared_source_files_unchanged=True,optimizer_updates=0,source_SHA256=bindings)
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
