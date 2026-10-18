"""Re-evaluate saved native V1 weights with current deterministic quick metrics."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import torch
from .conditioning_controls import TransportControl, transport_diagnostic_context
from .r9_diagnostic_context import DEFINITIONS as R9_DEFINITIONS, r9_diagnostic_context
from .task_attention_controls import task_value_context


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['ss64','pbr1024'],required=True)
    p.add_argument('--config-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--arm-config',type=Path)
    p.add_argument('--conditioner-checkpoint',type=Path)
    p.add_argument('--conditioning-mode',choices=['edited-rgb','source-only'],default='edited-rgb')
    p.add_argument('--transport-control',choices=('normal',*TransportControl.MODES),default='normal',
                   help='Diagnostic-only transport perturbation; retains the edited-RGB four-view input')
    p.add_argument('--r9-path-control',choices=('normal',*R9_DEFINITIONS),default='normal',
                   help='Saved-weight control for original R9 candidate; separate from matching-v2 evidence controls')
    p.add_argument('--task-value-control',choices=['normal','wrong-address'],default='normal',
                   help='Permute only retrieved attention values; preserve candidate keys, frozen/learned weights and scores')
    p.add_argument('--panel-size',type=int)
    p.add_argument('--panel-file',type=Path)
    p.add_argument('--seed-map-file',type=Path)
    p.add_argument('--generation-batch',type=int,default=1)
    args=p.parse_args()
    if args.transport_control!='normal' and args.conditioning_mode!='edited-rgb':
        raise ValueError('transport diagnostics retain the edited-RGB input; do not combine input controls')
    if args.r9_path_control!='normal' and (args.transport_control!='normal' or args.conditioning_mode!='edited-rgb'):
        raise ValueError('R9 diagnostics retain the edited-RGB input and cannot be combined with other controls')
    if args.task_value_control!='normal' and (args.transport_control!='normal' or args.r9_path_control!='normal' or args.conditioning_mode!='edited-rgb'):
        raise ValueError('task value diagnostic retains edited RGB and cannot combine other controls')
    from pixal3d import datasets,models,trainers
    from .quick_metrics import evaluate_quick
    from PIL import Image
    torch.set_num_threads(4);torch.cuda.set_device(0)
    cfg=json.loads((args.arm_config or (args.config_root/f'{args.stage}-v1.json')).read_text())
    semantic=cfg['trainer']['args'].get('deform_conditioner') is not None
    if semantic and not args.conditioner_checkpoint:
        raise ValueError('semantic checkpoint evaluation requires its trained conditioner')
    if not semantic and args.conditioner_checkpoint:
        raise ValueError('native V1 has no conditioner checkpoint')
    ds=getattr(datasets,cfg['dataset']['name'])(cfg['validation']['roots_file'],**cfg['validation']['dataset_args'])
    if args.conditioning_mode!='edited-rgb':
        from .conditioning_controls import ConditioningControl
        ds=ConditioningControl(ds,args.conditioning_mode)
    if args.transport_control!='normal':
        ds=TransportControl(ds,args.transport_control)
    if args.panel_size is not None:cfg['validation']['quick_metrics']['panel_size']=args.panel_size
    cfg['validation']['quick_metrics'].update(generation_batch_size=args.generation_batch,log_progress=True)
    if args.panel_file:cfg['validation']['quick_metrics']['panel_file']=str(args.panel_file)
    if args.seed_map_file:cfg['validation']['quick_metrics']['seed_map_file']=str(args.seed_map_file)
    with torch.device('meta'):
        md={name:getattr(models,spec['name'])(**spec['args']) for name,spec in cfg['models'].items()}
    kw=dict(cfg['trainer']['args']);kw.update(continuation_ckpt={'denoiser':str(args.checkpoint)},
        batch_size_per_gpu=1,batch_split=1,num_workers=0,prefetch_data=False,max_steps=1)
    args.output.mkdir(parents=True,exist_ok=True)
    trainer=getattr(trainers,cfg['trainer']['name'])(md,ds,**kw,output_dir=str(args.output),load_dir=None,step=None,wandb_run=None)
    if args.conditioner_checkpoint:
        trainer.deform_conditioner.load_state_dict(torch.load(args.conditioner_checkpoint,map_location='cpu',weights_only=True),strict=True)
        trainer.deform_conditioner.eval()
    trainer.models['denoiser'].eval();trainer.p_uncond=0
    r9_events=[];task_events=[]
    with transport_diagnostic_context(trainer.deform_conditioner,args.transport_control), \
            r9_diagnostic_context(trainer.deform_conditioner,args.r9_path_control,r9_events), \
            task_value_context(trainer.deform_conditioner,args.task_value_control,task_events), \
            torch.no_grad(),torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
        metrics,panel,ids=evaluate_quick(trainer,ds,cfg['validation'])
    result={'status':'PASS','stage':args.stage,'step':args.step,'weights':'raw','checkpoint':str(args.checkpoint),
            'conditioner_checkpoint':str(args.conditioner_checkpoint) if args.conditioner_checkpoint else None,
            'arm_config':str(args.arm_config or (args.config_root/f'{args.stage}-v1.json')),
            'arm':cfg['experiment']['arm'],
            'quick_metrics':metrics,'quick_panel_ids':ids,'ao_sampling_seed':0,'optimizer_updates':0,
            'optimizer_updates_during_evaluation':0,'per_asset':getattr(trainer,'_quick_per_asset',[]),
            'generation_batch_size':args.generation_batch,
            'sampling_steps':cfg['validation']['quick_metrics']['sampling_steps'],'guidance':3.,
            'panel_file':str(args.panel_file) if args.panel_file else None,
            'seed_map_file':str(args.seed_map_file) if args.seed_map_file else None}
    result.update(conditioning_mode=args.conditioning_mode,target_modality='rgb',
                  transport_control=args.transport_control,diagnostic_only=args.transport_control!='normal' or args.r9_path_control!='normal' or args.task_value_control!='normal',
                  transport_control_definition=TransportControl.DEFINITIONS.get(args.transport_control),
                  r9_path_control=args.r9_path_control,r9_path_control_definition=R9_DEFINITIONS.get(args.r9_path_control),
                  r9_path_control_events=r9_events,
                  task_value_control=args.task_value_control,task_value_control_events=task_events,
                  pbr_geometry='GT_target_shape_fixed' if args.stage=='pbr1024' else None,
                  input_states=['M0']*4 if args.conditioning_mode=='source-only' else ['M1','M0','M0','M0'],
                  input_view_count=4,
                  input_camera_ids=ds.rows[0].get('conditioning_camera_ids'),
                  manifest_sha256=hashlib.sha256(Path(cfg['validation']['roots_file']).read_bytes()).hexdigest())
    (args.output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    Image.fromarray(panel).save(args.output/'panel.png')
    # Save the actual tensors entering the model, rather than merely the
    # source filenames. This exposes wrong modality/order or duplicated views.
    from PIL import ImageDraw
    examples=[i for i,row in enumerate(ds.rows) if row['sample_id'] in ids][:4]
    contact=Image.new('RGB',(4*128,len(examples)*148));draw=ImageDraw.Draw(contact)
    for line,index in enumerate(examples):
        item=ds[index]
        if item['cond'].shape[0]!=4:raise ValueError('four-view baseline requires exactly four input images')
        for view,frame in enumerate(item['cond']):
            rgb=(frame.clamp(0,1).permute(1,2,0).numpy()*255).astype('uint8')
            contact.paste(Image.fromarray(rgb).resize((128,128)),(view*128,line*148+20))
            cams=ds.rows[index].get('conditioning_camera_ids',['?']*4)
            draw.text((view*128,line*148),f"{result['input_states'][view]} camera {cams[view]}",fill='white')
    contact.save(args.output/'actual_inputs.png')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
