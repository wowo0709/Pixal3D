"""Re-evaluate saved native V1 weights with current deterministic quick metrics."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['ss64','pbr1024'],required=True)
    p.add_argument('--config-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--arm-config',type=Path)
    p.add_argument('--conditioner-checkpoint',type=Path)
    args=p.parse_args()
    from pixal3d import datasets,models,trainers
    from .quick_metrics import evaluate_quick
    from PIL import Image
    torch.set_num_threads(4);torch.cuda.set_device(0)
    cfg=json.loads((args.arm_config or (args.config_root/f'{args.stage}-v1.json')).read_text())
    if args.arm_config and not args.conditioner_checkpoint:
        raise ValueError('semantic checkpoint evaluation requires its trained conditioner')
    ds=getattr(datasets,cfg['dataset']['name'])(cfg['validation']['roots_file'],**cfg['validation']['dataset_args'])
    with torch.device('meta'):
        md={name:getattr(models,spec['name'])(**spec['args']) for name,spec in cfg['models'].items()}
    kw=dict(cfg['trainer']['args']);kw.update(continuation_ckpt={'denoiser':str(args.checkpoint)},
        batch_size_per_gpu=1,batch_split=1,num_workers=0,prefetch_data=False,max_steps=1)
    args.output.mkdir(parents=True,exist_ok=True)
    trainer=getattr(trainers,cfg['trainer']['name'])(md,ds,**kw,output_dir=str(args.output),load_dir=None,step=None,wandb_run=None)
    if args.conditioner_checkpoint:
        trainer.deform_conditioner.load_state_dict(torch.load(args.conditioner_checkpoint,map_location='cpu',weights_only=True),strict=True)
    trainer.models['denoiser'].eval();trainer.p_uncond=0
    with torch.no_grad(),torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
        metrics,panel,ids=evaluate_quick(trainer,ds,cfg['validation'])
    result={'status':'PASS','stage':args.stage,'step':args.step,'weights':'raw','checkpoint':str(args.checkpoint),
            'arm':cfg['experiment']['arm'],
            'quick_metrics':metrics,'quick_panel_ids':ids,'ao_sampling_seed':0,'optimizer_updates':0}
    (args.output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    Image.fromarray(panel).save(args.output/'panel.png')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
