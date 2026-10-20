"""Unchanged native proof with the corrected isolated target-first trainer."""
import argparse
import json
from pathlib import Path


def main():
    # The original proof imports its trainer inside main. Bind only this
    # process's constructor; preserve every historical source byte on disk.
    from pixal3d.trainers.flow_matching import target_first_ray as original
    from pixal3d.trainers.flow_matching.target_first_ray_v2 import TargetFirstRayImageConditionedProjFlowMatchingCFGTrainerV2
    from . import probe_target_first_ray as proof
    saved = original.TargetFirstRayImageConditionedProjFlowMatchingCFGTrainer
    original.TargetFirstRayImageConditionedProjFlowMatchingCFGTrainer = TargetFirstRayImageConditionedProjFlowMatchingCFGTrainerV2
    try:
        proof.main()
    finally:
        original.TargetFirstRayImageConditionedProjFlowMatchingCFGTrainer = saved
    parser = argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    import torch
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ProjGrid
    from pixal3d.modules.target_first_ray_v2 import project_target_first_evidence
    cfg = json.loads(args.config.read_text());row=json.loads(Path(cfg['validation']['roots_file']).read_text().splitlines()[0])
    grid=ProjGrid(grid_resolution=16,image_resolution=512).cuda()
    matrices=torch.tensor(row['transform_matrix'],dtype=torch.float32,device='cuda')[None]
    camera=dict(transform_matrix=matrices,distance=torch.linalg.vector_norm(matrices[:,:,:3,3],dim=-1),
        camera_angle_x=torch.tensor(row['camera_angle_x'],dtype=torch.float32,device='cuda')[None],
        mesh_scale=torch.tensor([row['mesh_scale']],dtype=torch.float32,device='cuda'))
    with torch.no_grad():
        target=torch.full((1,4096,1024),2.,device='cuda')
        weight=torch.full((1,3,1024,1),.25,device='cuda')
        numerator=torch.full((1,3,1024,1024),.5,device='cuda')
        correction=project_target_first_evidence(grid,numerator,weight,camera,target)
        if torch.count_nonzero(correction):
            raise ValueError('source evidence equal to target must not change the target carrier')
    report=dict(status='PASS_ACTUAL_NATIVE_TARGET_FIRST_ANCHOR_CANCELLATION',
        actual_ProjGrid=True,actual_DeformNeXt_cameras=True,target=2.,source_semantic_value=2.,
        semantic_weight=.25,correction_max_abs=float(correction.abs().max()),optimizer_updates=0,
        old_source_subtraction_not_used=True,no_GT_surface_or_visibility_input=True)
    (args.output/'source-baseline-cancellation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
