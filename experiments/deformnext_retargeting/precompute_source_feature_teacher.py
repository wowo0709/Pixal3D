"""Encode an explicitly bound TRAIN-only source9/10 DINO teacher catalog."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import time

import numpy as np
import torch

from .cascade_view_roles import file_sha256
from .precompute_transport import atomic_npz, load_rgba, pixal_patches
from .source_feature_teacher import validate_teacher_records


def module_sha(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    allowed = os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')
    if not gpu.isdigit() or gpu not in allowed:
        raise ValueError('one exclusively assigned physical GPU required')
    if (plan['schema'] != 'source-feature-teacher-TRAIN-cache-plan-v1'
            or plan['camera_ids'] != [9, 10] or plan['batch_assets'] != 8):
        raise ValueError('explicit two-source teacher plan and16-image encoding batch required')
    for path, digest in plan['input_SHA256'].items():
        if file_sha256(path) != digest:
            raise ValueError(f'Bound teacher input changed: {path}')
    rows = [json.loads(line) for line in Path(plan['train_manifest']).read_text().splitlines()]
    validation = [json.loads(line) for line in Path(plan['validation_manifest']).read_text().splitlines()]
    validate_teacher_records(rows, {row['source_cluster_id'] for row in validation})
    rgb = json.loads(Path(plan['RGB_readiness']).read_text())['records']
    if (len(rows) != 1536 or [r['sample_id'] for r in rows] != [r['sample_id'] for r in rgb]):
        raise ValueError('fixed1536 TRAIN representatives and actual RGB readiness must agree')
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(plan['native_config']).read_text())
    spec = config['trainer']['args']['image_cond_model']['args']
    if spec['image_size'] != 512 or spec['grid_resolution'] != 16 or spec.get('use_naf_upsample', False):
        raise ValueError('unchanged native SS512 DINO encoder required')
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    extractor = DinoV3ProjFeatureExtractor(**spec).cuda().eval()
    assert not any(parameter.requires_grad for parameter in extractor.parameters())
    before = module_sha(extractor)
    start = time.monotonic()
    index = []
    with torch.inference_mode():
        for offset in range(0, len(rows), plan['batch_assets']):
            frames, identities = [], []
            for row, record in zip(rows[offset:offset+8], rgb[offset:offset+8]):
                assert row['source_cluster_id'] == record['source_cluster_id']
                identities.append(record)
                for camera in [9, 10]:
                    entry = record['source_GT_RGB'][str(camera)]
                    path = Path(plan['local_mirror_prefix']) / entry['path'].lstrip('/')
                    if path.stat().st_size != entry['bytes'] or file_sha256(path) != entry['SHA256']:
                        raise ValueError(f'Actual file4 teacher RGB identity mismatch: {path}')
                    frames.append(load_rgba(path, 512))
            images = torch.stack([frame[0] for frame in frames]).cuda()
            # Original transport values use FP32 frozen DINO then FP16 storage.
            # Do not silently introduce a different BF16 teacher distribution.
            with torch.autocast('cuda', enabled=False):
                features = pixal_patches(extractor, images).half().cpu().reshape(-1, 2, 1024, 1024)
            assert torch.isfinite(features).all()
            alpha = torch.stack([frame[1][8::16, 8::16] > .5 for frame in frames]).reshape(-1, 2, 1024)
            for j, record in enumerate(identities):
                row = rows[offset+j]
                name = hashlib.sha256(row['sample_id'].encode()).hexdigest()+'.npz'
                path = root/'features'/name
                atomic_npz(path, schema=np.asarray('source-feature-teacher-TRAIN-v1'),
                    split=np.asarray('train'), not_for_training=np.bool_(False),
                    sample_id=np.asarray(row['sample_id']), source_cluster_id=np.asarray(row['source_cluster_id']),
                    conditioning_input_sha256=np.asarray(row['conditioning_input_sha256']),
                    camera_ids=np.asarray([9, 10], dtype=np.int64),
                    teacher_source9_10_FP16=features[j].numpy(), teacher_alpha9_10=alpha[j].numpy(),
                    RGB_SHA256=np.asarray([record['source_GT_RGB'][str(c)]['SHA256'] for c in [9, 10]]))
                index.append(dict(sample_id=row['sample_id'], source_cluster_id=row['source_cluster_id'],
                    relative_path='features/'+name, SHA256=file_sha256(path), bytes=path.stat().st_size))
            progress = dict(status='LIVE_TRAIN_ONLY_TEACHER_ENCODING', assets_completed=len(index),
                assets_total=1536, elapsed_seconds=time.monotonic()-start, physical_GPU=gpu)
            (root/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
            print(json.dumps(progress), flush=True)
    assert module_sha(extractor) == before
    assert all(file_sha256(p) == digest for p, digest in plan['input_SHA256'].items())
    (root/'index.json').write_text(json.dumps(index, indent=2)+'\n')
    audit = dict(status='PASS_ACTUAL_TRAIN1536_SOURCE_FEATURE_TEACHER_ENCODING',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), assets=1536, source_clusters=1536,
        camera_ids=[9, 10], schema='source-feature-teacher-TRAIN-v1', physical_GPU=gpu,
        validation_source_overlap=0, teacher_sources='TRAIN M1 RGB only',
        native_DINO_FP32_encoding_FP16_storage=True, frozen_DINO_state_SHA256=before,
        encoder_unchanged=True, optimizer_updates=0, SS_or_PBR_training_started=False,
        test_split_accessed=False, inference_GT_input_added=False, full9149_scope_retained=True,
        elapsed_seconds=time.monotonic()-start,
        maximum_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
        maximum_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(), full_goal_complete=False)
    (root/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
