"""Eight rendered metrics for PBR isolation or an SS+PBR generated cascade."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .render_panel_statistics import MODELS, paired_panel_statistics


PRIMARY = ('PSNR', 'SSIM', 'LPIPS', 'M_IoU', 'CD', 'FScore', 'DreamSim', 'CLIP')
SUPPLEMENTARY = ('DreamSim_GTcrop', 'CLIP_GTcrop')
DIRECTIONS = dict(PSNR='higher', SSIM='higher', LPIPS='lower', M_IoU='higher',
    CD='lower', FScore='higher', DreamSim='lower', CLIP='higher',
    DreamSim_GTcrop='lower', CLIP_GTcrop='higher')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def thumbnail(value):
    if value.shape != (3, 512, 512) or not np.isfinite(value).all():
        raise ValueError('actual finite native512 RGB required')
    return Image.fromarray((np.clip(value.transpose(1,2,0),0,1)*255).round().astype('uint8')).resize((256,256))


def summarize(plan):
    scope = plan['comparison_scope']
    mode = plan['mode']
    if scope not in {'full-stack', 'PBR-isolation'} or mode not in {'scientific', 'engineering'}:
        raise ValueError('explicit generated full-stack/isolation and scientific/engineering scope required')
    if mode == 'scientific' and (plan['assets'] != 32 or plan['PBR_checkpoint_step'] != 1905):
        raise ValueError('the declared scientific panel is32 assets and equal1905 checkpoints')
    if tuple(plan['models']) != MODELS:
        raise ValueError('explicit ordered vanilla/retargeting/Direct models required')
    ids = [r['sample_id'] for r in plan['models']['v1']]
    if len(ids) != plan['assets'] or len(ids) != len(set(ids)):
        raise ValueError('full distinct declared panel required')
    perceptual = json.loads(Path(plan['perceptual']).read_text())
    if (perceptual['status'] != 'PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION'
            or perceptual['GPU_sanity']['status'] != 'PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY'
            or len(perceptual['per_view']) != 9*len(ids)):
        raise ValueError('actual official GPU scores for every model/asset/view required')
    views = {}
    for row in perceptual['per_view']:
        key = row['arm'], row['sample_id'], row['camera_id']
        if key in views:
            raise ValueError('duplicate perceptual observation')
        views[key] = row
    truth, histories, records, image_audits = {}, {}, {}, {}
    picture = Image.new('RGB', (1024, len(ids)*278), 'white')
    drawing = ImageDraw.Draw(picture)
    expected_arms = dict(v1='vanilla', retargeting='retargeting', direct='direct')
    sampling = None
    for j, model in enumerate(MODELS):
        rows = plan['models'][model]
        if [r['sample_id'] for r in rows] != ids:
            raise ValueError('model panel order or coverage differs')
        assets = []
        for i, item in enumerate(rows):
            sid = item['sample_id']
            generation = json.loads(Path(item['generation']).read_text())
            render = json.loads(Path(item['render']).read_text())
            camera_audit = json.loads(Path(item['render_camera_audit']).read_text())
            geometry = camera_audit['prediction_geometry_postprocess']
            expected = plan['PBR_checkpoints'][model]
            if (generation['status'] != 'PASS_ACTUAL_NATIVE_PBR_REPLAY_ON_GENERATED_SHAPE'
                    or generation['sample_id'] != sid or generation['actual_resolution'] != 1024
                    or generation['PBR_arm'] != expected_arms[model]
                    or generation['PBR_checkpoint'] != expected['path']
                    or generation['PBR_checkpoint_SHA256'] != expected['sha256']
                    or generation['additional_PBR_conditioner'] != (model != 'v1')
                    or not all(generation['geometry_array_byte_identity'].values())
                    or generation['GT_geometry_or_coords_used_in_generation']
                    or generation['GT_sparse_latents_or_projection_cache_used']
                    or generation['per_output_alignment'] or not generation['raw_surface_metrics_unchanged']):
                raise ValueError('actual pinned PBR replay on unchanged generated geometry required')
            if Path(expected['path']).name != f'denoiser_step{plan["PBR_checkpoint_step"]:07d}.pt':
                raise ValueError('declared common PBR checkpoint step differs')
            if (not geometry['same_budget_for_all_predictions']
                    or not geometry['raw_surface_metrics_unchanged']
                    or geometry['GT_geometry_used_for_simplification']
                    or geometry['per_output_alignment']):
                raise ValueError('common native1M-face rendering without GT or alignment required')
            params = generation['sampler_class'], generation['sampler_params'], generation['PBR_replay_seed']
            if sampling is None:
                sampling = params
            elif sampling != params:
                raise ValueError('paired native PBR sampler or seed differs')
            if render['status'] != 'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE':
                raise ValueError('the same archived original-material GT reference is required')
            if len(render['per_view']) != 3 or {r['camera_id'] for r in render['per_view']} != {8,9,10}:
                raise ValueError('all three declared native cameras required')
            metrics = generation['metrics']
            if (metrics['prediction_samples'] != 20000 or metrics['target_samples'] != 20000
                    or metrics['threshold'] != .02 or not generation['source_defined_normalization']):
                raise ValueError('unchanged raw surface sampling and source normalization required')
            row = dict(sample_id=sid, source_cluster_id=item['source_cluster_id'],
                **{m:float(np.mean([r[m] for r in render['per_view']])) for m in PRIMARY[:4]},
                CD=metrics['chamfer_L1'], FScore=metrics['surface_FScore'])
            scores = []
            for cid in [8,9,10]:
                value = views[model, sid, cid]
                archive = Path(item['render']).parent/f'camera{cid:02d}.npz'
                if (value['source_cluster_id'] != item['source_cluster_id']
                        or value['shared_npz_SHA256'] != digest(archive)):
                    raise ValueError('official scores refer to a different rendered prediction')
                with np.load(archive) as z:
                    gt, alpha, prediction = z['GT_shared_renderer'], z['GT_alpha'], z['prediction']
                    identity = (hashlib.sha256(gt.tobytes()).hexdigest(),
                                hashlib.sha256(alpha.tobytes()).hexdigest())
                    if truth.setdefault((sid,cid), identity) != identity:
                        raise ValueError('original-material GT pixels differ across models')
                    audit = value['image_audit']
                    if (identity[0] != audit['original_float_GT_SHA256']
                            or identity[1] != audit['original_GT_alpha_SHA256']):
                        raise ValueError('perceptual GT differs from six-metric GT')
                    crop_identity = (tuple(audit['GT_only_crop_box']),
                        audit['converted_image_SHA256']['reference_full'],
                        audit['converted_image_SHA256']['reference_GTcrop'])
                    if image_audits.setdefault((sid,cid),crop_identity) != crop_identity:
                        raise ValueError('GT-only official image preprocessing differs between arms')
                    if cid == 8:
                        picture.paste(thumbnail(prediction), ((j+1)*256, i*278+22))
                        drawing.text(((j+1)*256+5,i*278+4), f'{model}: case{i+1}', fill='black')
                        if j == 0:
                            picture.paste(thumbnail(gt), (0,i*278+22))
                            drawing.text((5,i*278+4), f'Original-material GT {i+1}', fill='black')
                scores.append(value['scores'])
            for key, metric in [('DreamSim_fullframe','DreamSim'),('CLIP_cosine_fullframe','CLIP'),
                    ('DreamSim_GT_foreground_crop','DreamSim_GTcrop'),('CLIP_cosine_GT_foreground_crop','CLIP_GTcrop')]:
                row[metric] = float(np.mean([r[key] for r in scores]))
            assets.append(row)
            histories.setdefault(sid, []).append(generation)
        records[model] = assets
    if scope == 'PBR-isolation':
        for sid, rows in histories.items():
            for field in ['original_prediction_SHA256','actual_initial_texture_noise_SHA256','geometry_evaluation_SHA256']:
                if len({r[field] for r in rows}) != 1:
                    raise ValueError('PBR-isolation sparse support, actual noise or geometry differs')
            if len({(r['metrics']['chamfer_L1'],r['metrics']['surface_FScore']) for r in rows}) != 1:
                raise ValueError('PBR-isolation must preserve raw geometric metrics exactly')
    statistics = paired_panel_statistics(records, PRIMARY+SUPPLEMENTARY)
    result = dict(status=('PASS_ACTUAL_TRAINED_PBR_GENERATED_PANEL_EIGHT_METRICS' if mode=='scientific'
                          else 'PASS_ACTUAL_PBR_REPLAY_ENGINEERING_PANEL_EIGHT_METRICS'),
        **statistics, metrics=list(PRIMARY), supplementary_GT_only_crop_metrics=list(SUPPLEMENTARY),
        directions=DIRECTIONS, per_asset=records, comparison_scope=scope,
        views_per_asset=3, PBR_checkpoint_step=plan['PBR_checkpoint_step'],
        continuation_training_seed=plan['training_seed'], independent_training_replication=False,
        development_panel=mode=='scientific', engineering_only=mode=='engineering',
        GT_shape_or_sparse_support_used=False, same_generated_shape_across_models=scope=='PBR-isolation',
        fixed_Shape512_Shape1024_weights=True,
        fixed_V1PBR762=mode=='engineering' and plan['PBR_checkpoint_step']==762,
        same_original_material_GT_RGB_alpha_and_GT_only_crops=True,
        raw_generated_surface_metrics_unchanged=True, prediction_render_face_budget=1000000,
        sampler=dict(name=sampling[0],params=sampling[1],PBR_replay_seed=sampling[2]),
        same_actual_noise_field_required=scope=='PBR-isolation',
        source_preservation_metrics=['LPIPS','DreamSim','CLIP'],
        target_alignment_metrics=['PSNR','SSIM','M_IoU','CD','FScore'],
        aggregation='mean3views perdeformation;deformation andequal-source means;pairedsourcecluster bootstrap',
        FID_KID_not_measured=True, heldout_test=False, model_adopted=False, full_goal_complete=False)
    return result, picture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    report, picture = summarize(plan)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    picture.save(output/'all-target8-shared-GT-comparison.png')
    print(json.dumps(dict(status=report['status'],scope=report['comparison_scope'],
        means=report['means'],contrasts=report['contrasts'])),flush=True)


if __name__ == '__main__':
    main()
