"""TaskRay369 native DDP2 full-batch capacity and matched scientific training."""
import argparse,hashlib,json,math,os,time
from pathlib import Path

from easydict import EasyDict
import torch
import torch.multiprocessing as mp


def worker(rank,config):
    from pixal3d import trainers,datasets
    from pixal3d.datasets.deformnext_deformation_query import DeformNeXtDeformationQuerySS64
    from experiments.multiview_finetune.runtime import install_hooks
    from .deformation_query_metrics import install_metric_dispatch
    from .task_ray_scientific_runtime import TaskRayScientificTrainer
    import train as native
    torch.cuda.set_device(rank);torch.set_num_threads(2)
    install_hooks();install_metric_dispatch()
    datasets.DeformNeXtDeformationQuerySS64=DeformNeXtDeformationQuerySS64
    selected=TaskRayScientificTrainer
    if config._task_mode=='capacity':
        class Capacity(TaskRayScientificTrainer):
            def run(self):
                def frozen(module):
                    digest=hashlib.sha256()
                    values=[(name,value) for name,value in module.named_parameters() if not value.requires_grad]
                    values+=list(module.named_buffers())
                    for name,value in values:
                        digest.update(name.encode());digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
                    return digest.hexdigest()
                before={name:frozen(m) for name,m in self.models.items()}
                initial=len(self.optimizer.state)
                snapshots={id(p):p.detach().cpu().clone() for g in self.optimizer.param_groups for p in g['params']}
                root=Path(self.anchor_stage['task_capacity_reference_run'])
                file='runtime_batch.jsonl' if self.rank==0 else 'runtime_batch.rank1.jsonl'
                reference=next(row for row in map(json.loads,(root/file).read_text().splitlines()) if row['step']==97)['loss']
                updates=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
                for _ in range(3):
                    begin=time.monotonic();result=self.run_step(self.load_data());self.step+=1
                    gradients={name:float(p.grad.detach().float().abs().sum()) if p.grad is not None else None
                        for name,p in self.deform_conditioner.named_parameters()
                        if name.startswith(('task_attention.','retargeter.deformation_projection.'))}
                    updates.append(dict(step=self.step,loss=result['loss'],new_gradient_L1=gradients,seconds=time.monotonic()-begin))
                first=dict(updates[0]['loss']);first.pop('task_ray_KL')
                if first!=reference:raise ValueError('first native97 common R96 loss fields differ')
                changed={g['model_name']:sum(not torch.equal(p.detach().cpu(),snapshots[id(p)]) for p in g['params']) for g in self.optimizer.param_groups}
                steps={g['model_name']:sorted({int(self.optimizer.state[p]['step']) for p in g['params']}) for g in self.optimizer.param_groups}
                if any(v is None or not math.isfinite(v) or v<=0 for v in updates[-1]['new_gradient_L1'].values()):
                    raise ValueError('all new three tensors must receive gradients')
                after={name:frozen(m) for name,m in self.models.items()}
                self.check_ddp()
                report=dict(status='PASS_ACTUAL_TASK_RAY369_DDP2_CAPACITY',rank=self.rank,
                    world_size=self.world_size,microbatch=self.batch_size_per_gpu,global_batch=self.batch_size,
                    accumulation=self.batch_split,initial_optimizer_states=initial,actual_optimizer_states=len(self.optimizer.state),
                    changed_trainable_tensors=changed,optimizer_steps=steps,
                    all_frozen_parameters_and_buffers_exact=before==after,post_update_DDP_equal=True,
                    first97_common_loss_fields_exact=True,updates=updates,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-started,
                    scientific_training_updates=0,prototype_discarded=True)
                from .task_ray_science_contract import validate_capacity
                validate_capacity(report)
                out=Path(self.output_dir)
                (out/f'capacity.rank{self.rank}.json').write_text(json.dumps(report,indent=2)+'\n')
                torch.distributed.barrier()
                if self.is_master:(out/'capacity.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(report),flush=True)
        selected=Capacity
    trainers.TaskRayScientificTrainer=selected
    native.main(rank,config)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--mode',choices=['capacity','train'],required=True)
    p.add_argument('--num-gpus',type=int,required=True);p.add_argument('--master-port',type=int,required=True)
    p.add_argument('--resume-step',type=int)
    a=p.parse_args();visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');allowed=set(os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','))
    if a.num_gpus!=2 or len(visible)!=2 or len(set(visible))!=2 or not set(visible)<=({'0','1','2','3'}&allowed):
        raise ValueError('exact assigned DDP2 GPU pair required')
    c=EasyDict(json.loads(a.config.read_text()));s=c.trainer.args.anchor_stage
    if (c.trainer.name!='TaskRayScientificTrainer' or c.dataset.name!='DeformNeXtDeformationQuerySS64'
            or c.training_seed!=101 or c.auto_retry!=0 or c.ckpt!='none' or c.load_ckpt is not None or c.load_dir!=''
            or s.task_execution_mode!=a.mode or s.diagnostic_profile!=(a.mode=='capacity')
            or (a.mode=='capacity' and c.use_wandb)):
        raise ValueError('explicit matched TaskRay config/mode required')
    c.update(config=str(a.config),num_gpus=2,num_nodes=1,node_rank=0,master_addr='127.0.0.1',
        master_port=str(a.master_port),multiview_runtime=True,tryrun=False,profile=False,
        _DQ_execution_mode=a.mode,_task_mode=a.mode)
    out=Path(c.output_dir)
    if a.resume_step is None:
        out.mkdir(parents=True,exist_ok=False);(out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    else:
        saved=EasyDict(json.loads((out/'config.json').read_text()))
        if (a.mode!='train' or not 96<a.resume_step<480 or saved.trainer!=c.trainer
                or saved.dataset!=c.dataset or saved.validation!=c.validation
                or saved.training_seed!=c.training_seed):raise ValueError('same recipe/native intermediate resume required')
        c.update(load_dir=str(out),load_ckpt=a.resume_step)
    mp.spawn(worker,args=(c,),nprocs=2,join=True)
    if a.mode=='train':
        progress=json.loads((out/'progress.json').read_text())
        trio=[out/'ckpts'/f'{name}_step0000480.pt' for name in ['denoiser','deform_conditioner','misc']]
        if progress['stage']!='completed' or progress['current']!=480 or not all(p.is_file() for p in trio):
            raise ValueError('actual fixed480 completion/checkpoint trio required')
        (out/'stage-completed.json').write_text(json.dumps(dict(status='PASS_ACTUAL_TASK_RAY_SCIENTIFIC480',
            source_step=96,additional_updates=384,total_updates=480,global_batch=96,world_size=2,
            microbatch_per_rank=48,accumulation=1,final_checkpoint_trio=list(map(str,trio)),
            model_adopted=False,full_goal_complete=False),indent=2)+'\n')


if __name__=='__main__':main()
