"""Run the appearance-only Direct arm through the frozen native375-state loop."""
from pathlib import Path
import json
import sys

from .train_ray_feature_joint import worker as native_worker


def worker(local_rank,config):
    from pixal3d.modules.direct_ray_retargeting_no_query import DirectRayRetargetingNoQueryConditioner
    from .direct_no_query_joint_runtime import DirectNoQueryJointTrainer
    from . import ray_feature_joint_runtime as runtime
    original_trainer = runtime.RayFeatureJointTrainer
    original_module = runtime.RayFeatureAttentionConditioner
    runtime.RayFeatureJointTrainer = DirectNoQueryJointTrainer
    runtime.RayFeatureAttentionConditioner = DirectRayRetargetingNoQueryConditioner
    try:
        # This function runs inside every spawned rank. A patch only in the
        # parent's main would silently construct the old module in DDP children.
        return native_worker(local_rank,config)
    finally:
        runtime.RayFeatureJointTrainer = original_trainer
        runtime.RayFeatureAttentionConditioner = original_module


def main():
    from . import train_ray_feature_joint as native
    config_path = Path(sys.argv[sys.argv.index('--config')+1])
    cfg = json.loads(config_path.read_text())
    if (cfg['dataset']['name']!='DeformNeXtSS64'
            or cfg['validation']['dataset_name']!='DeformNeXtSS64'
            or cfg['trainer']['args']['anchor_stage'].get('variant')!='direct-physical-NoDQ'
            or cfg['dataset']['args'].get('deformation_root') is not None):
        raise ValueError('explicit appearance-only Direct SS recipe required')
    original = native.worker
    native.worker = worker
    try:
        native.main()
    finally:
        native.worker = original
    out = Path(cfg['output_dir'])
    mode = sys.argv[sys.argv.index('--mode')+1]
    world = int(sys.argv[sys.argv.index('--num-gpus')+1])
    for rank in range(world):
        init = json.loads((out/f'NoDQ-stage-initialization.rank{rank}.json').read_text())
        if init['deformation_projection_present'] or init['physical_deformation_channels']!=0:
            raise ValueError('each actual rank must report structural query absence')
    if mode!='train':
        for path in [out/'joint-profile.json',*[out/f'joint-profile.rank{i}.json' for i in range(world)]]:
            report = json.loads(path.read_text())
            if (report['actual_optimizer_states']!=375
                    or report['changed_trainable_tensors_by_group']!={'default':360,'deform_conditioner':6,'ray_readout':9}
                    or report['final_optimizer_steps_by_group']!={'default':[99],'deform_conditioner':[99],'ray_readout':[3]}):
                raise ValueError('actual all375 tensors and common366 Adam96-to99 proof required')
            report.update(variant='direct-physical-NoDQ',physical_deformation_channels=0,
                deformation_projection_present=False,retargeter_input_channels=2048)
            path.write_text(json.dumps(report,indent=2)+'\n')
    elif (out/'stage-completed.json').exists():
        path = out/'stage-completed.json';report = json.loads(path.read_text())
        report.update(variant='direct-physical-NoDQ',physical_deformation_channels=0,
            deformation_projection_present=False,retargeter_input_channels=2048)
        path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
