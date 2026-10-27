"""Prepare and collect the post-hoc original Pixal3D RGB1 TEST508 table row.

The preceding five-model inference family is intentionally left unchanged.
This script verifies that every added prediction is paired with the exact
original-material GT image used by that family, then reuses the pinned metric
implementations and feature reference bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .inception_distribution_metrics import official_distribution_metrics


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def prior(root):
    native = root.parent/'actual-selected-five-model-TEST508-tenmetrics-v636'
    original = load(native/'native-plans/SELECTED/inception/native.json')
    percept = load(native/'native-plans/SELECTED/perceptual/native.json')
    reference = load(native/'native-plans/reference/inception/native.json')
    return native, original, percept, reference


def prepare(root):
    native, original, percept, reference = prior(root)
    qc_root=root.parent/'actual-frozen-TEST508-GT-alpha-raster-QC-v623'
    qc_summary=load(qc_root/'summary.json')
    if (qc_summary['status']!='PASS_ACTUAL_TEST508_GT_RASTER_ALPHA_DATA_ONLY_AUDIT'
            or qc_summary['assets']!=508 or qc_summary['views']!=1524
            or qc_summary['integer_quantized_one_percent_failures']!=0):
        raise ValueError('complete common GT-only raster QA required')
    with (qc_root/'per-view.jsonl').open() as stream:
        qc={(r['sample_id'],r['camera_id']):r for r in map(json.loads,stream)}
    if len(qc)!=1524:raise ValueError('GT-only QA per-view population differs')
    old_ref = {(x['sample_id'], x['camera_id']): x for x in reference['observations']}
    samples = original['expected_sample_ids']
    if len(samples) != 508 or len(set(samples)) != 508:
        raise ValueError('previous frozen TEST508 population required')
    assignment=load(root/'assignment-all.json')
    if len(assignment)!=508 or {x['row']['sample_id'] for x in assignment}!=set(samples):
        raise ValueError('single-view TEST assignment differs from frozen population')
    snapshot=Path(assignment[0]['official_snapshot'])
    stems={'SS64':'ss_flow_img_dit_1_3B_64_bf16',
        'Shape512':'slat_flow_img2shape_dit_1_3B_512_bf16',
        'Shape1024':'slat_flow_img2shape_dit_1_3B_1024_bf16',
        'PBR1024':'slat_flow_imgshape2tex_dit_1_3B_1024_bf16'}
    blobs={stage:(snapshot/'ckpts'/(stem+'.safetensors')).resolve().name
           for stage,stem in stems.items()}
    if (any(len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in blobs.values())
            or any(x['official_snapshot']!=str(snapshot) for x in assignment)):
        raise ValueError('one immutable official Pixal3D snapshot required')
    observations, inputs, cases = [], [], []
    for sid in samples:
        case = root/'cases'/sid
        audit = load(case/'case-audit.json')
        generation = load(case/'geometry/generation.json')
        render_proof = load(case/'render-device-proof.json')
        shared_proof = load(case/'shared-device-proof.json')
        if (render_proof['physical_GPU'] != shared_proof['physical_GPU']
                or render_proof['physical_GPU'] not in ('6','7')
                or not render_proof['numerical_AST_unchanged']
                or not shared_proof['numerical_AST_unchanged']):
            raise ValueError(f'render/score device proof differs: {sid}')
        if audit['physical_GPU'] != render_proof['physical_GPU']:
            audit['physical_GPU'] = render_proof['physical_GPU']
            audit['device_provenance_reconciled_after_parallel_reuse'] = True
            save(case/'case-audit.json',audit)
        if (audit['status'] != 'PASS_PIXAL3D_SINGLE_VIEW_RENDER_AND_SHARED_GT'
                or audit['sample_id'] != sid or generation['sample_id'] != sid
                or generation['input_states'] != ['M1'] or generation['input_camera_ids'] != [8]
                or generation['deformation_finetuning_updates'] != 0
                or not generation['original_single_view_stage_weights']
                or generation['GT_geometry_or_coords_used_in_generation']):
            raise ValueError(f'not the original single-view RGB1 model: {sid}')
        views = {x['camera_id']: x for x in audit['target_and_source_scores']}
        if set(views) != {8,9,10}:
            raise ValueError(f'camera set changed: {sid}')
        native_views = {x['camera_id']:x for x in load(case/'render/render-evaluation.json')['per_view']}
        if set(native_views)!={8,9,10}:
            raise ValueError(f'native rendered camera set changed: {sid}')
        for camera in (8,9,10):
            old=qc[sid,camera]
            if (not old['integer_quantized_one_percent_pass']
                    or old['GT_alpha_outside_one_pixel_raster']>old['integer_ceil_one_percent_allowance']
                    or abs(native_views[camera]['GT_geometry_RGBA_mask_iou']-old['GT_geometry_RGBA_mask_iou'])>1e-6
                    or abs(native_views[camera]['GT_geometry_RGBA_containment']-old['GT_geometry_RGBA_containment'])>1e-6):
                raise ValueError(f'common GT raster QA differs: {sid} camera{camera}')
        for camera in (8,9,10):
            path = case/'shared'/f'camera{camera:02}.npz'
            with np.load(path,allow_pickle=False) as z:
                pred = z['prediction']
                ref = z['GT_shared_renderer']
                if (pred.shape != (3,512,512) or ref.shape != pred.shape
                        or not np.isfinite(pred).all() or not np.isfinite(ref).all()):
                    raise ValueError(f'bad native render: {sid} camera{camera}')
                pred_sha = hashlib.sha256(pred.tobytes()).hexdigest()
                ref_sha = hashlib.sha256(ref.tobytes()).hexdigest()
            if ref_sha != old_ref[sid,camera]['original_float_RGB_SHA256']:
                raise ValueError(f'GT reference differs from frozen five models: {sid} camera{camera}')
            if generation['source_cluster_id'] != old_ref[sid,camera]['source_cluster_id']:
                raise ValueError(f'source cluster differs from frozen TEST population: {sid}')
            archive_sha = sha(path)
            row = dict(sample_id=sid, source_cluster_id=generation['source_cluster_id'], camera_id=camera,
                       archive=str(path), archive_SHA256=archive_sha, array='prediction',
                       original_float_RGB_SHA256=pred_sha)
            observations.append(row)
            inputs.append(dict(arm='PIXAL3D_SINGLE_VIEW_RGB1', sample_id=sid,
                               source_cluster_id=generation['source_cluster_id'], camera_id=camera,
                               shared_npz=str(path), shared_npz_SHA256=archive_sha))
        cases.append(dict(sample_id=sid, case_audit=str(case/'case-audit.json'),
                          case_audit_SHA256=sha(case/'case-audit.json'),
                          generation_SHA256=sha(case/'geometry/generation.json')))
    plan = {k:v for k,v in original.items() if k != 'observations'}
    plan['observations'] = observations
    plan['model_binding'] = dict(kind='original-Pixal3D-single-view-four-stages',
        input='edited M1 RGB camera8 only', deformation_finetuning_updates=0,
        multi_view_finetuning_used=False, descriptive_post_hoc_baseline=True,
        official_snapshot=str(snapshot),official_pipeline_SHA256=sha(snapshot/'pipeline.json'),
        official_stage_HF_blob_identifiers=blobs,
        projective_DINO_backbone=str(assignment[0]['DINO_local_path']),
        stage_projective_conditioner_config_SHA256={stage:sha(path)
            for stage,path in assignment[0]['stage_configs'].items()})
    percept_plan = {k:v for k,v in percept.items() if k not in ('output','inputs')}
    percept_plan['output'] = str(root/'metrics/perceptual')
    percept_plan['inputs'] = inputs
    save(root/'metrics/plans/inception.json',plan)
    save(root/'metrics/plans/perceptual.json',percept_plan)
    save(root/'metrics/plans/completeness.json',dict(status='PASS_SINGLE_VIEW_RGB1_508x3_IDENTICAL_GT',
        cases=cases, generated_images=1524, reference_feature_bank=str(native/'native/reference/inception/features.npz'),
        frozen_reference_plan_SHA256=sha(native/'native-plans/reference/inception/native.json'),
        common_GT_only_raster_QA_SHA256=sha(qc_root/'per-view.jsonl'),
        original_five_model_family_unchanged=True))
    print(json.dumps(dict(status='PASS_SINGLE_VIEW_RGB1_508x3_IDENTICAL_GT',cases=len(cases),images=len(observations))))


def collect(root):
    native, original, _, reference = prior(root)
    plan = load(root/'metrics/plans/inception.json')
    completeness = load(root/'metrics/plans/completeness.json')
    if completeness['status'] != 'PASS_SINGLE_VIEW_RGB1_508x3_IDENTICAL_GT':
        raise ValueError('complete matching GT evidence required')
    ids = plan['expected_sample_ids']
    pred_path = root/'metrics/inception/features.npz'
    ref_path = native/'native/reference/inception/features.npz'
    per = load(root/'metrics/perceptual/evaluation.json')
    if per['status'] != 'PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION' or len(per['per_view']) != 1524:
        raise ValueError('full official perceptual panel required')
    scores = {(r['sample_id'],r['camera_id']): r for r in per['per_view']}
    if len(scores) != 1524:
        raise ValueError('duplicate perceptual observation')
    if len({x['source_cluster_id'] for x in plan['observations']})!=134:
        raise ValueError('frozen 134-source TEST population required')
    with np.load(pred_path,allow_pickle=False) as p, np.load(ref_path,allow_pickle=False) as q:
        keys_p = list(zip(p['sample_id'].tolist(),p['camera_id'].tolist()))
        keys_q = list(zip(q['sample_id'].tolist(),q['camera_id'].tolist()))
        expected = [(sid,cam) for sid in ids for cam in (8,9,10)]
        if keys_p != expected or keys_q != expected:
            raise ValueError('prediction/reference feature bank order differs')
        sys.path.insert(0,plan['package_root'])
        import torch_fidelity
        if torch_fidelity.__version__ != '0.3.0':
            raise ValueError('pinned torch-fidelity required')
        distribution = official_distribution_metrics(p['features'],q['features'],subset_size=1000,subsets=100,seed=42017)
    per_plan = load(root/'metrics/plans/perceptual.json')
    items = {(r['sample_id'],r['camera_id']):r for r in per_plan['inputs']}
    for key,row in scores.items():
        if key not in items or row['shared_npz_SHA256'] != items[key]['shared_npz_SHA256']:
            raise ValueError('perceptual score differs from planned generated pixels')
    if sha(root/'metrics/plans/inception.json') != load(root/'metrics/inception/audit.json')['input_plan_SHA256']:
        raise ValueError('official feature bank not extracted from exact plan')
    rows = []
    for sid in ids:
        audit = load(root/'cases'/sid/'case-audit.json')
        views = {x['camera_id']:x for x in audit['target_and_source_scores']}
        a,b = scores[sid,9]['scores'],scores[sid,10]['scores']
        rows.append(dict(LPIPS=np.mean([views[c]['LPIPS'] for c in (9,10)]),
            DreamSim=np.mean([a['DreamSim_fullframe'],b['DreamSim_fullframe']]),
            CLIP=np.mean([a['CLIP_cosine_fullframe'],b['CLIP_cosine_fullframe']]),
            PSNR=views[8]['PSNR'],SSIM=views[8]['SSIM'],M_IoU=views[8]['M_IoU'],
            CD=audit['geometry_metrics']['chamfer_L1'],
            FScore=audit['geometry_metrics']['surface_FScore']))
    metric = {k:float(np.mean([row[k] for row in rows])) for k in rows[0]}
    metric['FID']=distribution['frechet_inception_distance']
    metric['KID']=distribution['kernel_inception_distance_mean']
    order=('LPIPS','DreamSim','CLIP','FID','KID','PSNR','SSIM','M_IoU','CD','FScore')
    if not np.isfinite([metric[k] for k in order]).all():
        raise ValueError('all ten metric means must be finite')
    result=dict(status='PASS_POST_HOC_SINGLE_VIEW_RGB1_TEST508_TEN_METRICS',
        sample_count=508,source_clusters=len(set(x['source_cluster_id'] for x in plan['observations'])),
        images=1524,model='Original Pixal3D single-view zero-shot, edited target RGB1',
        table_order=list(order),primary_ten_metrics=metric,distribution=distribution,
        original_five_model_family_unchanged=True,
        input_plans=dict(inception=str(root/'metrics/plans/inception.json'),perceptual=str(root/'metrics/plans/perceptual.json')),
        output_SHA256=dict(perceptual=sha(root/'metrics/perceptual/evaluation.json'),
            inception=sha(pred_path),reference=sha(ref_path)))
    save(root/'metrics/table-row.json',result)
    print(json.dumps(dict(status=result['status'],row=metric),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','collect'])
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    {'prepare':prepare,'collect':collect}[a.phase](a.root)


if __name__=='__main__': main()
