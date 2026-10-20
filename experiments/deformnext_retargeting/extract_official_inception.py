"""Pinned official Inception2048 features from archived native512 renders."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8388608),b''):
            value.update(block)
    return value.hexdigest()


def render_uint8(value):
    if value.shape != (3,512,512) or not np.isfinite(value).all():
        raise ValueError('actual finite native512 CHW RGB render required')
    return (np.clip(value,0,1)*255).round().astype(np.uint8)


def validate_pool(plan):
    if plan['scope'] not in {'engineering','final-test508'}:
        raise ValueError('explicit engineering or prespecified complete test508 scope required')
    expected = {'generated-prediction':'prediction', 'shared-original-material-GT':'GT_shared_renderer'}
    if plan['pool_role'] not in expected:
        raise ValueError('one explicit generated or shared original-material reference pool required')
    if not isinstance(plan['model_binding'],dict) or not plan['model_binding']:
        raise ValueError('explicit reference or model checkpoint provenance required')
    rows = plan['observations']
    ids = [(r['sample_id'],r['camera_id']) for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('distinct explicit sample/camera observations required')
    clusters = {}
    for row in rows:
        if row['array'] != expected[plan['pool_role']] or not row['source_cluster_id']:
            raise ValueError('do not mix prediction and GT images in one pool')
        if clusters.setdefault(row['sample_id'],row['source_cluster_id']) != row['source_cluster_id']:
            raise ValueError('source cluster must be stable across cameras')
    if plan['scope'] == 'final-test508':
        samples = plan['expected_sample_ids']
        if (len(samples)!=508 or len(set(samples))!=508 or len(rows)!=1524
                or set(ids)!={(sid,cid) for sid in samples for cid in (8,9,10)}):
            raise ValueError('all508 prespecified test assets and all three cameras required')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    rows = validate_pool(plan)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if (len(visible)!=1 or visible[0] not in {'0','1','2','3'}
            or visible[0] not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(',')):
        raise ValueError('one explicitly assigned GPU0–3 required')
    weights = Path(plan['weights'])
    if digest(weights)!=plan['weights_SHA256']:
        raise ValueError('official Inception weights changed')
    package = Path(plan['package_root'])
    sys.path.insert(0,str(package))
    import torch
    import torch_fidelity
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    if (torch_fidelity.__version__!='0.3.0'
            or not Path(torch_fidelity.__file__).resolve().is_relative_to(package.resolve())):
        raise ValueError('use the explicitly isolated official torch-fidelity0.3.0 package')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    extractor = FeatureExtractorInceptionV3('inception-v3-compat',['2048'],
        feature_extractor_weights_path=str(weights)).eval().cuda()
    extractor.requires_grad_(False)
    batch = plan['batch_size']
    if not isinstance(batch,int) or isinstance(batch,bool) or batch<1:
        raise ValueError('explicit positive feature-extraction batch required')
    args.output.mkdir(parents=True,exist_ok=False)
    features, identities = [], []
    torch.cuda.reset_peak_memory_stats()
    exact_repeat = None
    with torch.inference_mode():
        for start in range(0,len(rows),batch):
            pixels = []
            for row in rows[start:start+batch]:
                path = Path(row['archive'])
                if digest(path)!=row['archive_SHA256']:
                    raise ValueError('native render archive changed')
                with np.load(path) as z:
                    rgb = z[row['array']]
                    raw_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                    if raw_sha!=row['original_float_RGB_SHA256']:
                        raise ValueError('selected native RGB array identity differs')
                    pixels.append(render_uint8(rgb))
                identities.append(dict(**row,converted_CHW_uint8_SHA256=hashlib.sha256(pixels[-1].tobytes()).hexdigest()))
            image = torch.from_numpy(np.stack(pixels)).cuda()
            value = extractor(image)[0]
            if value.shape!=(len(pixels),2048) or not torch.isfinite(value).all():
                raise ValueError('official2048 feature schema differs')
            if start==0:
                exact_repeat = bool(torch.equal(value,extractor(image)[0]))
                if not exact_repeat:
                    raise ValueError('same-batch frozen Inception extraction must be deterministic')
            features.append(value.cpu().numpy())
            print(json.dumps(dict(images_complete=min(start+batch,len(rows)),total=len(rows))),flush=True)
    values = np.concatenate(features)
    if values.dtype!=np.float32 or not np.isfinite(values).all() or not np.any(values):
        raise ValueError('finite nontrivial FP32 feature bank required')
    np.savez(args.output/'features.npz',features=values,
        sample_id=np.asarray([r['sample_id'] for r in rows]),camera_id=np.asarray([r['camera_id'] for r in rows]),
        source_cluster_id=np.asarray([r['source_cluster_id'] for r in rows]))
    report = dict(status='PASS_ACTUAL_OFFICIAL_INCEPTION2048_FEATURE_POOL',scope=plan['scope'],pool_role=plan['pool_role'],
        images=len(rows),assets=len(Counter(r['sample_id'] for r in rows)),observations=identities,
        input_plan_SHA256=digest(args.plan),weights_SHA256=plan['weights_SHA256'],
        package='torch-fidelity0.3.0',feature_layer='2048',feature_dtype='float32',
        model_binding=plan['model_binding'],
        fullframe=True,GT_crop_or_CLIP_preprocessing_used=False,TF32=False,autocast=False,
        same_batch_repeat_exact=exact_repeat,features_SHA256=digest(args.output/'features.npz'),
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        optimizer_updates=0,distribution_quality_metrics_not_yet_computed=True,
        final_model_quality_gain_not_established=True,full_goal_complete=False)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='observations'}),flush=True)


if __name__=='__main__':
    main()
