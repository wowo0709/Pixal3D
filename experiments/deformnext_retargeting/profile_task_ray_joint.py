"""Disposable native369-state probe, restricted to an assigned GPU and micro2."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from easydict import EasyDict
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    a = p.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    allowed = os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(',')
    if len(visible) != 1 or visible[0] not in {'0','1','2','3'} or visible[0] not in allowed:
        raise ValueError('exactly one assigned physical GPU0-3 required')
    cfg = EasyDict(json.loads(a.config.read_text()))
    if (cfg.trainer.name != 'TaskRayJointTrainer' or cfg.training_seed != 101
            or cfg.use_wandb or cfg.auto_retry != 0 or cfg.get('load_ckpt') is not None
            or cfg.trainer.args.batch_size_per_gpu != 2 or cfg.trainer.args.batch_split != 1
            or not cfg.trainer.args.anchor_stage.diagnostic_profile):
        raise ValueError('disposable explicit seed101/micro2 native engineering recipe required')
    output = Path(cfg.output_dir)
    output.mkdir(parents=True,exist_ok=False)
    cfg.update(config=str(a.config),num_gpus=1,num_nodes=1,node_rank=0,
        master_addr='127.0.0.1',master_port='31771',multiview_runtime=True,
        tryrun=False,profile=False,_DQ_execution_mode='profile')
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.20,0)
    torch.set_num_threads(2)
    from pixal3d import datasets,trainers
    from pixal3d.datasets.deformnext_deformation_query import DeformNeXtDeformationQuerySS64
    from experiments.multiview_finetune.runtime import install_hooks
    from .deformation_query_metrics import install_metric_dispatch
    from .task_ray_joint_runtime import TaskRayJointTrainer
    import train as native
    install_hooks()
    install_metric_dispatch()
    datasets.DeformNeXtDeformationQuerySS64 = DeformNeXtDeformationQuerySS64

    class Probe(TaskRayJointTrainer):
        def run(self):
            def frozen_hash(module):
                digest = hashlib.sha256()
                for name,value in module.named_parameters():
                    if not value.requires_grad:
                        digest.update(name.encode())
                        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
                return digest.hexdigest()
            before = {name:frozen_hash(m) for name,m in self.models.items()}
            snapshots = {id(p):p.detach().cpu().clone() for g in self.optimizer.param_groups for p in g['params']}
            assert len(self.optimizer.state) == 366
            reference = json.loads(Path(cfg.task_probe.reference_micro2_profile).read_text())
            updates = []
            torch.cuda.reset_peak_memory_stats()
            began = time.monotonic()
            for _ in range(3):
                started = time.monotonic()
                result = self.run_step(self.load_data())
                self.step += 1
                grads = {name:float(p.grad.detach().float().abs().sum()) if p.grad is not None else None
                    for name,p in self.deform_conditioner.named_parameters()
                    if name.startswith(('retargeter.deformation_projection.','task_attention.'))}
                updates.append(dict(step=self.step,loss=result['loss'],seconds=time.monotonic()-started,
                    new_gradient_L1=grads,statistics=self.deform_conditioner.last_feature_attention_stats))
            after = {name:frozen_hash(m) for name,m in self.models.items()}
            changed = {g['model_name']:sum(not torch.equal(p.detach().cpu(),snapshots[id(p)])
                       for p in g['params']) for g in self.optimizer.param_groups}
            steps = {g['model_name']:sorted({int(self.optimizer.state[p]['step']) for p in g['params']})
                     for g in self.optimizer.param_groups}
            original = reference['updates'][0]['loss']
            assert all(updates[0]['loss'][name] == value for name,value in original.items()),(updates[0]['loss'],original)
            assert before == after and len(self.optimizer.state) == 369
            assert changed == dict(default=360,deform_conditioner=6,deformation_query=1,task_ray_attention=2)
            assert steps == dict(default=[99],deform_conditioner=[99],deformation_query=[3],task_ray_attention=[3])
            assert len(updates[-1]['new_gradient_L1']) == 3
            assert all(v is not None and math.isfinite(v) and v>0 for v in updates[-1]['new_gradient_L1'].values())
            flow_gradients = self._task_flow_only_gradient_L1
            assert len(flow_gradients) == 3 and all(v is not None and math.isfinite(v) and v>0 for v in flow_gradients)
            report = dict(status='PASS_ACTUAL_BOUNDED_TASK_RAY369_NATIVE_PROFILE',
                common_Adam_states_restored=366,actual_optimizer_states=369,
                changed_trainable_tensors=changed,optimizer_steps=steps,updates=updates,
                initial_common_R96_loss_fields_exact=True,all_new3_gradients_positive_by3=True,
                actual_SS_MSE_only_gradient_L1=flow_gradients,
                new_attention_receives_actual_SS_gradient_not_only_KL=True,
                all_frozen_NN_semantic_decoder_and_reliability_exact=True,
                semantic_prior7_frozen=True,task_attention2_trainable=True,reliability4_frozen=True,
                task_query_channels=4096,actual_deformation_query_channels=2048,
                task_logit_bound=1.,task_KL_weight=.001,global_batch=2,microbatch=2,
                world_size=1,accumulation=1,allocator_fraction=.20,
                seconds=time.monotonic()-began,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                prototype_discarded=True,scientific_training_updates=0,
                scope='Actual native TRAIN micro2 correctness only;notDDPcapacityorquality')
            (Path(self.output_dir)/'task-ray-profile.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
    trainers.TaskRayJointTrainer = Probe
    native.main(0,cfg)


if __name__ == '__main__':
    main()
