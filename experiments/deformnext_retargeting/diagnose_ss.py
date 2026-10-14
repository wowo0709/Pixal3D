"""Controlled SS diagnostics; RGB oracle inputs are never training arms."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pixal3d import datasets, models, trainers
from pixal3d.utils.data_utils import recursive_to_device
from experiments.deformnext_retargeting.quick_metrics import binary_metrics, stratified_indices, _panel_image
from pixal3d.datasets.deformnext import _load_image


def render_mask(occupancy, pack):
    import utils3d
    from pixal3d.representations import Voxel
    from pixal3d.renderers import VoxelRenderer
    coords = occupancy.nonzero()
    resolution = occupancy.shape[-1]
    rotated = torch.stack((coords[:, 0], resolution - 1 - coords[:, 2], coords[:, 1]), 1)
    scale = float(pack['mesh_scale'][0])
    radius = float(pack['camera_distance'][0, 0])
    fov = float(pack['camera_angle_x'][0, 0])
    attrs = torch.ones(len(coords), 3, device=coords.device)
    voxel = Voxel(origin=[-0.5/scale]*3, voxel_size=1/resolution/scale,
                  coords=rotated, attrs=attrs, layout={'color': slice(0, 3)})
    ext = utils3d.torch.extrinsics_look_at(torch.tensor([0., -radius, 0.], device=coords.device),
                 torch.zeros(3, device=coords.device), torch.tensor([0., 0., 1.], device=coords.device))
    intr = utils3d.torch.intrinsics_from_fov(fov_x=torch.tensor(fov, device=coords.device),
                                            fov_y=torch.tensor(fov, device=coords.device))
    renderer = VoxelRenderer({'resolution': 512, 'near': 0.01, 'far': 100., 'ssaa': 1})
    return renderer.render(voxel, ext, intr)['alpha'] > 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    ds = getattr(datasets, cfg['dataset']['name'])(cfg['validation']['roots_file'], **cfg['validation']['dataset_args'])
    with torch.device('meta'):
        model_dict = {key: getattr(models, spec['name'])(**spec['args']) for key,spec in cfg['models'].items()}
    kwargs = dict(cfg['trainer']['args']);kwargs.update(num_workers=0,batch_size_per_gpu=1,batch_split=1,max_steps=1)
    trainer = getattr(trainers, cfg['trainer']['name'])(model_dict, ds, **kwargs,
             output_dir=str(args.output), load_dir=None, step=None, wandb_run=None)
    trainer.p_uncond = 0
    trainer.models['denoiser'].eval()
    # Same seeds, targets and checkpoint throughout. RGB cases deliberately
    # use unavailable target texture only to locate an input-domain failure.
    cases = [('editing',12,3.,1.), ('editing',32,3.,1.),
             ('editing',32,1.,1.), ('editing',32,7.5,3.),
             ('rgb_anchor_oracle',12,3.,1.), ('rgb_same_state_oracle',12,3.,1.)]
    if args.baseline_only:
        cases=cases[:1]
    indices = stratified_indices(ds,4)
    results = []
    truth = []
    for offset,index in enumerate(indices):
        pack = recursive_to_device(ds.collate_fn([ds[index]]), trainer.device)
        target = pack.pop('x_0');pack.pop('_dataset_name',None);pack.pop('_sha256',None)
        with torch.no_grad(),torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
            target_decoded = ds.decode_latent(target)[0,0] > 0
        rgb = ds._resolve(ds.rows[index]['target_rgb_supervision'])
        anchor = ds._resolve(ds.rows[index]['target_anchor'])
        expected = torch.from_numpy(np.asarray(Image.open(anchor).convert('RGBA'))[:,:,3].copy()).cuda() > 127
        actual = render_mask(target_decoded,pack)
        gt_stats = binary_metrics(actual,expected)
        truth.append({'sample_id':ds.rows[index]['sample_id'],'category':ds.rows[index]['category'],
                      'gt_ss_render_mask_iou':gt_stats['voxel_iou'],'occupied_voxels':int(target_decoded.sum())})
        torch.manual_seed(42+offset);torch.cuda.manual_seed(42+offset)
        noise = torch.randn_like(target)
        for mode,steps,cfg_strength,rescale in cases:
            data = dict(pack)
            if mode == 'rgb_anchor_oracle':
                data['cond'] = pack['cond'].clone()
                data['cond'][:,0] = _load_image(rgb,512).cuda()
            elif mode == 'rgb_same_state_oracle':
                data['cond'] = _load_image(rgb,512).cuda()[None,None]
                for key in ('camera_angle_x','camera_distance','transform_matrix'):
                    data[key] = pack[key][:,:1]
            with torch.no_grad(),torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
                cond = trainer.get_inference_cond(**data)
                prediction = trainer.get_sampler().sample(trainer.models['denoiser'],noise=noise.clone(),**cond,
                      steps=steps,guidance_strength=cfg_strength,rescale_t=rescale,verbose=False).samples
                pred_decoded = ds.decode_latent(prediction)[0,0] > 0
            stats = binary_metrics(pred_decoded,target_decoded)
            row={'sample_id':ds.rows[index]['sample_id'],'category':ds.rows[index]['category'],
                 'mode':mode,'steps':steps,'cfg':cfg_strength,'rescale_t':rescale,**stats}
            results.append(row);print(json.dumps(row),flush=True)
            gt_img=target_decoded.float().amax(0)[None,None].repeat(1,3,1,1)
            pred_img=pred_decoded.float().amax(0)[None,None].repeat(1,3,1,1)
            Image.fromarray(_panel_image(pred_img,gt_img)).save(args.output/f'{offset}-{mode}-steps{steps}-cfg{cfg_strength}-t{rescale}.png')
    report={'source_checkpoint':kwargs['continuation_ckpt'],'diagnostic_only':True,
            'gt_render_checks':truth,'results':results}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(truth),flush=True)


if __name__ == '__main__':
    main()
