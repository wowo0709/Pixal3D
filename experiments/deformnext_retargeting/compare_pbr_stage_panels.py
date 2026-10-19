"""Paired fixed-panel PBR stage diagnostics with explicit training budgets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def aggregate(rows,key,indices):
    values = np.asarray([row[key] for row in rows])[indices]
    if key == 'foreground_psnr':
        weights = np.asarray([row['foreground_fraction'] for row in rows])[indices]
        return -10*np.log10((weights*10**(-values/10)).sum(-1)/weights.sum(-1))
    return values.mean(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    paths = json.loads(args.inputs.read_text())
    models = {name:json.loads(Path(path).read_text()) for name,path in paths.items()}
    baseline = models['matched-v1']
    ids = [row['sample_id'] for row in baseline['per_asset']]
    if len(ids)!=32 or len(set(ids))!=32 or len({r['source_cluster_id'] for r in baseline['per_asset']})!=32:
        raise ValueError('fixed32 cases from32 distinct source clusters required')
    for name,result in models.items():
        if (result['status']!='PASS' or result['optimizer_updates_during_evaluation']!=0
                or result['generation_batch_size']!=1 or result['sampling_steps']!=12
                or result['guidance']!=3. or result['conditioning_mode']!='edited-rgb'
                or [row['sample_id'] for row in result['per_asset']]!=ids
                or not result['streaming_PBR']['fixed_GT_target_Shape']):
            raise ValueError('paired RGB anchor-view PBR stage protocol required: '+name)
        for left,right in zip(result['per_asset'],baseline['per_asset']):
            for key in ['seed','source_cluster_id','conditioning_input_sha256','actual_noise_SHA256',
                        'actual_RGB_input_SHA256','actual_GT_PBR_latent_SHA256','actual_GT_Shape_condition_SHA256']:
                if left[key]!=right[key]:
                    raise ValueError('paired actual PBR input differs: '+name+' '+key)
    if models['direct']['step']!=1905 or baseline['step']!=1905:
        raise ValueError('primary comparison must use both fixed1905 final weights')
    if ('outside-proposal' in models and
            (models['outside-proposal']['step']!=1905
             or models['outside-proposal']['checkpoint']!=models['direct']['checkpoint']
             or models['outside-proposal']['conditioner_checkpoint']!=models['direct']['conditioner_checkpoint']
             or models['outside-proposal']['outside_proposal_value_control']['status']!='EXECUTED')):
        raise ValueError('outside-proposal diagnostic must retain the final Direct weights')
    draws = np.random.default_rng(20261020).integers(0,32,size=(4000,32))
    contrasts = {}
    comparisons = [('direct','matched-v1')]
    if 'outside-proposal' in models:
        comparisons.append(('direct','outside-proposal'))
    if 'original-v1-step762' in models:
        comparisons += [('matched-v1','original-v1-step762'),('direct','original-v1-step762')]
    for candidate,reference in comparisons:
        metrics = {}
        for key in ['foreground_psnr','crop_lpips','crop_ssim']:
            a,b = models[candidate]['per_asset'],models[reference]['per_asset']
            delta = aggregate(a,key,draws)-aggregate(b,key,draws)
            metrics[key] = dict(candidate_minus_baseline=float(aggregate(a,key,np.arange(32))-aggregate(b,key,np.arange(32))),
                paired_source_cluster_CI95=np.quantile(delta,[.025,.975]).tolist(),higher_is_better=key!='crop_lpips')
        contrasts[candidate+' minus '+reference] = dict(metrics=metrics,
            equal_training_budget=models[candidate]['step']==models[reference]['step'],
            candidate_total_steps=models[candidate]['step'],reference_total_steps=models[reference]['step'])
    primary = contrasts['direct minus matched-v1']['metrics']
    args.output.parent.mkdir(parents=True,exist_ok=False)
    with args.output.open('x') as stream:
        json.dump(dict(status='PASS_PAIRED_MATCHED_PBR_FINAL1905_STAGE32',assets=32,source_clusters=32,
            generation_noises=1,bootstrap_draws=4000,training_seed_replication=False,contrasts=contrasts,
            models={name:result['quick_metrics'] for name,result in models.items()},
            primary_supports_further_replication=primary['foreground_psnr']['paired_source_cluster_CI95'][0]>0
                and primary['crop_lpips']['paired_source_cluster_CI95'][1]<0,
            fixed_GT_target_Shape=True,render_view='input anchor camera08',full_cascade=False,novel_view=False,
            target_crop_LPIPS_is_not_source_preservation=True,development_panel_screen_only=True,
            model_adopted=False,full_goal_complete=False,inputs={path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for path in paths.values()}),stream,indent=2)
        stream.write('\n')


if __name__=='__main__':
    main()
