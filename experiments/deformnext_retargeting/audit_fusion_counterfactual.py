"""Frozen-checkpoint paired fusion diagnostics; no optimizer updates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['ss64','pbr1024'],required=True)
    parser.add_argument('--config-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from pixal3d import datasets,models,trainers
    from pixal3d.modules import sparse as sp
    from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
    from pixal3d.utils.data_utils import recursive_to_device
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import compute_multiview_projection_matrices
    from experiments.deformnext_retargeting.quick_metrics import stratified_indices,binary_metrics,_render_pbr_anchor
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    args.output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((args.config_root/f'{args.stage}-v1.json').read_text())
    ds=getattr(datasets,cfg['dataset']['name'])(cfg['validation']['roots_file'],**cfg['validation']['dataset_args'])
    bc=json.loads((args.config_root/f'{args.stage}-b3.json').read_text())
    bds=getattr(datasets,bc['dataset']['name'])(bc['validation']['roots_file'],**bc['validation']['dataset_args'])
    with torch.device('meta'):
        model_dict={key:getattr(models,spec['name'])(**spec['args']) for key,spec in cfg['models'].items()}
    kwargs=dict(cfg['trainer']['args']);kwargs.update(num_workers=0,batch_size_per_gpu=1,batch_split=1,max_steps=1)
    trainer=getattr(trainers,cfg['trainer']['name'])(model_dict,ds,**kwargs,output_dir=str(args.output),load_dir=None,step=None,wandb_run=None)
    trainer.p_uncond=0
    trainer.models['denoiser'].eval()
    modules={}
    for arm in ['b1','b2','b3']:
        spec=json.loads((args.config_root/f'{args.stage}-{arm}.json').read_text())['trainer']['args']['deform_conditioner']
        modules[arm]=DeformationAwareConditioner(**spec).eval().cuda()
    names=['native','cached_spatial','b1','b2','b3','b3_blend_0.1','b3_blend_0.25','b3_blend_0.5']
    predictions={name:[] for name in names};truth=[];squared={name:0. for name in names};base_squared={name:0. for name in names};pixels={name:0 for name in names}
    report={'diagnostic_only':True,'optimizer_updates':0,'stage':args.stage,'checkpoint':kwargs['continuation_ckpt'],
            'steps':12,'cfg':3.,'sample_ids':[],'results':{}}
    selected=stratified_indices(ds,4 if args.stage=='ss64' else 2)
    bindices={row['sample_id']:index for index,row in enumerate(bds.rows)}
    with torch.inference_mode():
        for offset,index in enumerate(selected):
            pack=recursive_to_device(ds.collate_fn([ds[index]]),'cuda')
            target=pack.pop('x_0');pack.pop('_dataset_name',None);pack.pop('_sha256',None)
            if args.stage=='pbr1024':pack['coords']=target.coords
            bpack=recursive_to_device(bds.collate_fn([bds[bindices[ds.rows[index]['sample_id']]]]),'cuda')
            transport={key.removeprefix('transport_'):value for key,value in bpack.items() if key.startswith('transport_')}
            torch.manual_seed(42+offset);torch.cuda.manual_seed(42+offset)
            noise=target.replace(torch.randn_like(target.feats)) if args.stage=='pbr1024' else torch.randn_like(target)
            with torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
                native=trainer.get_inference_cond(**pack)
                image_model=getattr(trainer,'_image_cond_module',trainer.image_cond_model)
                projection,_=compute_multiview_projection_matrices(pack['transform_matrix'],pack['camera_distance'],image_model.fixed_projection_transform)
                anchor,_=trainer.encode_image_proj(pack['cond'][:,0],camera_angle_x=pack['camera_angle_x'][:,0],distance=pack['camera_distance'][:,0],mesh_scale=pack['mesh_scale'],transform_matrix=projection[:,0],coords=pack.get('coords'))
                key='proj'
                conditions={'native':native['cond']}
                for arm,module in modules.items():
                    trainer.deform_conditioner=module
                    condition={'global':native['cond']['global'],'proj':anchor['proj']}
                    conditions[arm]=trainer._inject_deform_transport(condition,transport,pack.get('coords'),source_views=3)
                source_sum=None
                for view in range(1,4):
                    value=transport['source_values'][:,view-1].reshape(1,32,32,-1).float()
                    projected=image_model.proj_grid(value,pack['camera_angle_x'][:,view],pack['camera_distance'][:,view],pack['mesh_scale'],projection[:,view])
                    if args.stage=='pbr1024':
                        xyz=target.coords[:,1:].long();res=image_model.grid_resolution
                        projected=projected.reshape(1,res,res,res,-1)[0,xyz[:,0],xyz[:,1],xyz[:,2]]
                    source_sum=projected if source_sum is None else source_sum+projected
                conditions['cached_spatial']={'global':native['cond']['global'],'proj':(anchor['proj']+source_sum)/4}
                for alpha in [.1,.25,.5]:
                    conditions[f'b3_blend_{alpha}']={'global':native['cond']['global'],'proj':native['cond']['proj']+alpha*(conditions['b3']['proj']-native['cond']['proj'])}
                current=[]
                for name in names:
                    inference=dict(native,cond=conditions[name])
                    case_noise=noise.replace(noise.feats.clone()) if args.stage=='pbr1024' else noise.clone()
                    prediction=trainer.get_sampler().sample(trainer.models['denoiser'],noise=case_noise,**inference,steps=12,guidance_strength=3.,verbose=False).samples
                    current.append(prediction)
                    if args.stage=='ss64':predictions[name].append(prediction)
                if args.stage=='ss64':truth.append(target)
            if args.stage=='pbr1024':
                z=sp.sparse_cat(current+[target]);shape=sp.sparse_cat([pack['concat_cond']]*(len(names)+1))
                with torch.autocast('cuda',enabled=False):
                    reps=ds.decode_latent(z.float(),shape.float(),batch_size=1)
                    camera=(float(pack['camera_angle_x'][0,0]),float(pack['camera_distance'][0,0]),float(pack['mesh_scale'][0]))
                    renders=_render_pbr_anchor(ds,reps,[camera]*len(reps),256)
                    gt=renders['shaded'][-1];gt_base=renders['base_color'][-1]
                    for j,name in enumerate(names):
                        mask=((renders['alpha'][j]>.5)|(renders['alpha'][-1]>.5)).reshape(1,256,256).expand_as(gt)
                        squared[name]+=float((renders['shaded'][j]-gt).square()[mask].sum())
                        base_squared[name]+=float((renders['base_color'][j]-gt_base).square()[mask].sum())
                        pixels[name]+=int(mask.sum())
                    del reps,renders,z,shape
            report['sample_ids'].append(ds.rows[index]['sample_id'])
            print('Completed sample',offset,flush=True)
        if args.stage=='ss64':
            with torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
                decoded=ds.decode_latent(torch.cat([p for name in names for p in predictions[name]]+truth),batch_size=2)[:,0]>0
            count=len(selected);gt=decoded[-count:]
            for j,name in enumerate(names):report['results'][name]=binary_metrics(decoded[j*count:(j+1)*count],gt)
        else:
            import math
            for name in names:
                report['results'][name]={'foreground_psnr':-10*math.log10(squared[name]/pixels[name]),'foreground_base_color_psnr':-10*math.log10(base_squared[name]/pixels[name])}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['results']),flush=True)


if __name__=='__main__':main()
