"""Predeclared epoch-one execution/quality gates; not a performance promotion."""
from __future__ import annotations
import math


def review(config, records, baseline=None):
    epoch=config['experiment']['updates_per_epoch']
    initial=next(r for r in records if r['step']==0)
    current=next(r for r in reversed(records) if r['step']==epoch)
    if initial['quick_panel_ids']!=current['quick_panel_ids']:
        raise ValueError('pilot panel changed')
    a,b=initial['quick_metrics'],current['quick_metrics']
    if not all(math.isfinite(float(x)) for x in [*a.values(),*b.values()]):
        return {'passed':False,'reason':'nonfinite validation metric'}
    if baseline is not None and baseline['quick_panel_ids']!=current['quick_panel_ids']:
        raise ValueError('native V1 comparison panel differs')
    native=None if baseline is None else baseline['quick_metrics']
    stage=config['dataset']['name'];criteria={}
    if stage=='DeformNeXtSS64':
        floor=max(a['voxel_iou']-.02, native['voxel_iou']-.05 if native else 0.)
        criteria['voxel_iou_minimum']=floor
        passed=b['voxel_iou']>=floor
    elif stage=='DeformNeXtPBR1024':
        floor=max(a['foreground_psnr']-1.,native['foreground_psnr']-2. if native else -math.inf)
        ceiling=min(a['crop_lpips']+.03,native['crop_lpips']+.05 if native else math.inf)
        criteria.update(foreground_psnr_minimum=floor,crop_lpips_maximum=ceiling)
        passed=b['foreground_psnr']>=floor and b['crop_lpips']<=ceiling
    else:raise ValueError('unsupported pilot stage')
    return {'passed':passed,'gate':'execution_and_no_catastrophic_regression_not_promotion',
            'criteria':criteria,'initial':a,'epoch_one':b,'native_v1_epoch_one':native}
