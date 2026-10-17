"""Test target/source projection balance while retaining all four RGB views."""
import argparse
import copy
import json
from pathlib import Path
import types

import torch


def blend_projected(native, anchor, strength):
    if not 0 <= strength <= .5:
        raise ValueError('probe retains positive weight for all three source views')
    return native + strength * (anchor - native)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('config-root','checkpoint','panels','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--calibration',action='store_true')
    p.add_argument('--strengths',default='0,.25,.5')
    p.add_argument('--panel-size',type=int,default=32)
    a=p.parse_args();strengths=[float(s) for s in a.strengths.split(',')]
    if any(not 0<=s<=.5 for s in strengths):raise ValueError('strength outside predeclared range')
    from pixal3d import datasets,models,trainers
    from .quick_metrics import evaluate_quick
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import compute_multiview_projection_matrices
    from PIL import Image
    torch.set_num_threads(4);torch.cuda.set_device(0)
    cfg=json.loads((a.config_root/'ss64-v1.json').read_text());args=copy.deepcopy(cfg['validation']['dataset_args'])
    manifest=cfg['validation']['roots_file']
    if a.calibration:manifest=str(a.panels/'head_calibration.jsonl');args['expected_split']='train'
    ds=getattr(datasets,cfg['dataset']['name'])(manifest,**args)
    quick=cfg['validation']['quick_metrics']
    quick.update(panel_file=str(a.panels/('calibration-32.json' if a.calibration else f'panel-{a.panel_size}.json')),
        seed_map_file=str(a.panels/('calibration-seeds.json' if a.calibration else 'seed-map.json')),
        generation_batch_size=8,log_progress=True)
    with torch.device('meta'):md={n:getattr(models,s['name'])(**s['args']) for n,s in cfg['models'].items()}
    kw=copy.deepcopy(cfg['trainer']['args']);kw.update(continuation_ckpt={'denoiser':str(a.checkpoint)},
        batch_size_per_gpu=1,batch_split=1,num_workers=0,prefetch_data=False,max_steps=1)
    a.output.mkdir(parents=True,exist_ok=True)
    trainer=getattr(trainers,cfg['trainer']['name'])(md,ds,**kw,output_dir=str(a.output),
        load_dir=None,step=None,wandb_run=None)
    trainer.models['denoiser'].eval();trainer.p_uncond=0
    original=trainer._encode_deform_views;state={'strength':0.}
    def encode(self,images,camera_info,transport,**kwargs):
        if images.ndim!=5 or images.shape[1]!=4:raise ValueError('four distinct input views required')
        condition,negative=original(images,camera_info,transport,**kwargs)
        if state['strength']==0:return condition,negative
        if transport is not None:raise ValueError('role-only probe cannot also change matching')
        module=self._image_cond_module
        projection,_=compute_multiview_projection_matrices(camera_info['transform_matrix'],camera_info['distance'],module.fixed_projection_transform)
        anchor,_=self.encode_image_proj(images[:,0],camera_angle_x=camera_info['camera_angle_x'][:,0],
            distance=camera_info['distance'][:,0],mesh_scale=camera_info['mesh_scale'],transform_matrix=projection[:,0],coords=camera_info.get('coords'))
        condition['proj']=blend_projected(condition['proj'],anchor['proj'],state['strength'])
        return condition,negative
    trainer._encode_deform_views=types.MethodType(encode,trainer);results={}
    for strength in strengths:
        state['strength']=strength;name='strength-'+str(strength);out=a.output/name;out.mkdir(exist_ok=True)
        with torch.inference_mode():metrics,panel,ids=evaluate_quick(trainer,ds,cfg['validation'])
        result={'status':'PASS','diagnostic_only':True,'GT_input':False,'optimizer_updates':0,
            'case':name,'strength':strength,'input_views':4,'target_projection_weight':.25+.75*strength,
            'each_source_projection_weight':.25*(1-strength),'global_condition':'native four-view mean unchanged',
            'split':'train-calibration' if a.calibration else 'validation','checkpoint':str(a.checkpoint),
            'sample_ids':ids,'quick_metrics':metrics,'per_asset':trainer._quick_per_asset,'sampling_steps':12,'guidance':3.}
        (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n');Image.fromarray(panel).save(out/'panel.png')
        results[name]=result;print(json.dumps({'case':name,'metrics':metrics}),flush=True)
    baseline=results.get('strength-0.0');best=max(results.values(),key=lambda r:r['quick_metrics']['voxel_iou'])
    report={'status':'finished','selection_scope':'train-calibration' if a.calibration else 'validation',
        'selected_strength':best['strength'],'cases':{n:r['quick_metrics'] for n,r in results.items()},
        'gain_vs_native':None if baseline is None else best['quick_metrics']['voxel_iou']-baseline['quick_metrics']['voxel_iou'],
        'no_target_only_arm':True,'all_four_views_retained':True}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
