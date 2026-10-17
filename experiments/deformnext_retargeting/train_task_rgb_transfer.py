"""Matched D0/D1 original RGB objective on frozen current-data patch inputs.

This pilot is not a production matcher checkpoint and does not change any
running SS conditioner. Evaluator cue files open after the fixed-final freeze.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from pixal3d.modules.r9_matching import R9Scorer
from .historical_q2c_training import fuse_label_values
from .cache_task_rgb_transfer import SCHEMA
from .reproduce_q2c import checked, read, reference, write_new

ARMS = ('D0-PAIRWISE', 'D1-Q2C')


def load_record(record, *, with_training_values):
    """Never open the independent evaluator file on the training path."""
    with np.load(checked(record['artifact']), allow_pickle=False) as data:
        if (str(data['schema']) != SCHEMA or str(data['sample_id']) != record['sample_id']
                or str(data['role']) != record['role'] or not bool(data['supervision_only'])):
            raise ValueError('current RGB task cache identity differs')
        result = {key: data[key].copy() for key in
                  ['query_L12', 'source_L12', 'candidates', 'raw_scores', 'eligible_queries']}
        if with_training_values:
            if record['role'] != 'train':
                raise ValueError('calibration task values are forbidden in training')
            result.update(source_rgb=data['label_source_rgb'].astype(np.float32),
                          target_rgb=data['label_target_rgb'].astype(np.float32))
    q, source, ids = result['query_L12'], result['source_L12'], result['candidates']
    if (q.shape != (1024, 1024) or source.shape != (3072, 1024) or ids.shape != (1024, 64)
            or np.any((ids < 0) | (ids >= 3072)) or np.any(np.diff(np.sort(ids, -1), axis=-1) == 0)
            or result['eligible_queries'].shape != (1024,)):
        raise ValueError('original RAW64/FRAME1024 patch contract differs')
    if not np.isfinite(result['raw_scores'][result['eligible_queries']]).all():
        raise ValueError('eligible query has invalid fixed RAW candidates')
    if with_training_values:
        for key, count in [('source_rgb', 3072), ('target_rgb', 1024)]:
            values = result[key]
            if values.shape != (16, count, 3) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError('training RGB task value shape/range differs')
    result['query_L12'] = q.astype(np.float32)
    result['source_L12'] = source.astype(np.float32)
    return result


def make_model(arm, seed, device='cuda:0'):
    if arm not in ARMS:
        raise ValueError('only the matched D0/D1 arms are exposed')
    torch.manual_seed(seed)
    model = R9Scorer(decoder_arm=arm).to(device)
    groups = [{'params': list(model.fusion_head.parameters()), 'lr': .001}]
    if arm == 'D1-Q2C':
        groups.append({'params': list(model.decoder.parameters()), 'lr': .0001})
    optimizer = torch.optim.AdamW(groups, weight_decay=.0001)
    return model, optimizer


def sample_batch(records, rng, device='cuda:0'):
    sample_id = sorted(records)[int(rng.integers(len(records)))]
    record = records[sample_id]
    eligible = np.flatnonzero(record['eligible_queries'])
    indices = rng.choice(eligible, size=256, replace=len(eligible) < 256)
    tensor = lambda value: torch.as_tensor(np.array(value, copy=True), device=device)
    candidates = record['candidates'][indices]
    return {'sample_id': sample_id,
            'query': {12: tensor(record['query_L12'][indices])},
            'source': {12: tensor(record['source_L12'])},
            'candidates': tensor(candidates).long(), 'raw_scores': tensor(record['raw_scores'][indices]),
            'source_rgb': tensor(record['source_rgb'][:, candidates]),
            'target_rgb': tensor(record['target_rgb'][:, indices]),
            'sampled_query_sha256': hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()}


def forward(model, batch):
    return model(batch['query'], batch['source'], batch['candidates'], raw_scores=batch['raw_scores'])


def step(model, optimizer, batch, *, check_initial=False):
    optimizer.zero_grad(set_to_none=True)
    result = forward(model, batch)
    if check_initial:
        baseline = copy.deepcopy(model)
        baseline.decoder.blocks = torch.nn.ModuleList()
        with torch.no_grad():
            expected = forward(baseline, batch)['scores']
        if not torch.equal(result['scores'], expected) or not torch.equal(result['scores'], batch['raw_scores']):
            raise ValueError('initial D0/D1/RAW score parity failed')
    prediction = fuse_label_values(result['scores'], batch['source_rgb'], temperature=.005)
    task = (prediction - batch['target_rgb']).abs().mean()
    loss = task + .05 * result['preservation_loss']
    if not torch.isfinite(loss):
        raise ValueError('nonfinite controlled patch RGB loss')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError('missing or nonfinite original RGB task gradient')
    if check_initial and not any(bool(g.abs().sum() > 0) for g in gradients):
        raise ValueError('initial task objective has no connected matching gradient')
    optimizer.step()
    return {'loss': float(loss.detach()), 'task_loss': float(task.detach()),
            'sample_id': batch['sample_id'], 'sampled_query_sha256': batch['sampled_query_sha256'],
            'initial_D0_D1_RAW_scores_exact': True if check_initial else None}


def evaluate(model, records, labels, output, *, allowed_role='calibration'):
    model.eval()
    rows = []
    with torch.no_grad():
        for record in records:
            data = load_record(record, with_training_values=False)
            eligible = np.flatnonzero(data['eligible_queries'])
            if not len(eligible):
                rows.append({'sample_id': record['sample_id'], 'source_cluster_id': record['source_cluster_id'],
                             'queries': 0, 'excluded_reason': 'no observable valid task queries'})
                continue
            with np.load(checked(record['evaluation_artifact']), allow_pickle=False) as bank:
                if (str(bank['schema']) != SCHEMA or str(bank['sample_id']) != record['sample_id']
                        or str(bank['role']) != allowed_role):
                    raise ValueError('held-out cue bank identity differs')
                values = torch.as_tensor(bank['evaluation_source_rgb'].astype(np.float32), device='cuda:0')
                target = torch.as_tensor(bank['evaluation_target_rgb'][:, eligible].astype(np.float32), device='cuda:0')
            source = {12: torch.as_tensor(data['source_L12'], device='cuda:0')}
            scores, candidates, raw = [], [], []
            for selected in np.array_split(eligible, max(1, (len(eligible) + 31) // 32)):
                q = {12: torch.as_tensor(data['query_L12'][selected], device='cuda:0')}
                ids = torch.as_tensor(data['candidates'][selected], device='cuda:0').long()
                base = torch.as_tensor(data['raw_scores'][selected], device='cuda:0')
                scores.append(model(q, source, ids, raw_scores=base)['scores'])
                candidates.append(ids); raw.append(base)
            scores, ids, raw = torch.cat(scores), torch.cat(candidates), torch.cat(raw)
            candidate_rgb = values[:, ids]
            predicted = fuse_label_values(scores, candidate_rgb, temperature=.005)
            raw_prediction = fuse_label_values(raw, candidate_rgb, temperature=.005)
            selected = scores.argmax(-1)
            raw_selected = raw.argmax(-1)
            top1 = candidate_rgb[:, torch.arange(len(ids), device='cuda:0'), selected]
            raw_top1 = candidate_rgb[:, torch.arange(len(ids), device='cuda:0'), raw_selected]
            with np.load(labels / (record['sample_id'] + '.npz'), allow_pickle=False) as teacher:
                if str(teacher['input_sha256']) != record['geometry_input_sha256']:
                    raise ValueError('auxiliary topology evaluation teacher differs')
                positive = torch.as_tensor(teacher['positive_indices'][eligible], device='cuda:0').long()
                visible = torch.as_tensor(teacher['positive_visible'][eligible], device='cuda:0')
            correct = ((ids[..., None] == positive[:, None]) & visible[:, None]).any(-1)
            with np.load(checked(record['descriptor_cache']), allow_pickle=False) as original:
                dino_values = torch.as_tensor(original['source_values'].astype(np.float32), device='cuda:0').flatten(0, 1)
            teacher_value = (dino_values[positive] * visible[..., None]).sum(1) / visible.sum(-1).clamp_min(1)[:, None]
            soft_value = (dino_values[ids] * (scores / .005).softmax(-1)[..., None]).sum(1)
            raw_value = (dino_values[ids] * (raw / .005).softmax(-1)[..., None]).sum(1)
            result = {'sample_id': record['sample_id'], 'source_cluster_id': record['source_cluster_id'],
                      'queries': len(eligible), 'task_rgb_l1': float((predicted-target).abs().mean()),
                      'raw_soft_rgb_l1': float((raw_prediction-target).abs().mean()),
                      'learned_top1_rgb_l1': float((top1-target).abs().mean()),
                      'raw_top1_rgb_l1': float((raw_top1-target).abs().mean()),
                      'cue_errors': (predicted-target).abs().mean((1, 2)).cpu().tolist(),
                      'exact_patch_top1_accuracy': float(correct.gather(1, selected[:, None]).float().mean()),
                      'RAW_exact_patch_top1_accuracy': float(correct.gather(1, raw_selected[:, None]).float().mean()),
                      'proposal_recall': float(correct.any(-1).float().mean())}
            result.update(DINO_teacher_cosine_error=float((1-F.cosine_similarity(soft_value, teacher_value)).mean()),
                          RAW_DINO_teacher_cosine_error=float((1-F.cosine_similarity(raw_value, teacher_value)).mean()))
            rows.append(result)
            if len(rows) <= 3:
                panels = []
                for value in [target[0], predicted[0]]:
                    canvas = torch.ones(1024, 3, device='cuda:0')
                    canvas[torch.as_tensor(eligible, device='cuda:0')] = value
                    panels.append((canvas.reshape(32, 32, 3).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
                Image.fromarray(np.concatenate(panels, axis=1)).resize((512, 256), Image.Resampling.NEAREST).save(
                    output / f'cue-panel-{len(rows):02d}.png')
            print(json.dumps({'calibration_assets_completed': len(rows), **result}), flush=True)
    return rows


def train(args):
    inputs = read(args.inputs / 'summary.json')
    if inputs['schema'] != SCHEMA or inputs['status'] != 'PASS':
        raise ValueError('current controlled-task cache has not passed')
    if not args.preflight_only and (not inputs['full_population'] or len(inputs['records']) != 320):
        raise ValueError('full pilot requires the frozen256/64 population')
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ['summary.json', 'checkpoint-final.pt', 'train-log.jsonl']):
        raise FileExistsError('task head output already exists; preserve the prior attempt')
    training = {r['sample_id']: load_record(r, with_training_values=True)
                for r in inputs['records'] if r['role'] == 'train' and r['eligible_queries'] > 0}
    if not training:
        raise ValueError('no observable training queries')
    model, optimizer = make_model(args.arm, args.seed)
    rng = np.random.default_rng(args.seed)
    steps = 2 if args.preflight_only else 1200
    started = time.monotonic()
    run = None
    if args.wandb_id:
        import wandb
        run = wandb.init(project='pixal3d-deformnext-retargeting', entity='youngwoojeon-unist',
            id=args.wandb_id, resume='never', dir=str(args.output), config={
                'scope': 'current-DeformNeXt-controlled-patch-RGB-pilot', 'arm': args.arm, 'seed': args.seed,
                'steps': steps, 'training_assets_with_observable_queries': len(training),
                'input_summary': reference(args.inputs / 'summary.json'), 'fixed_candidates': 64,
                'query_batch': 256, 'label_cues': 16, 'temperature': .005, 'production_promotion': False})
    history = []
    with (args.output / 'train-log.jsonl').open('x') as stream:
        for update in range(steps):
            batch = sample_batch(training, rng)
            row = step(model, optimizer, batch, check_initial=update == 0)
            row.update(update=update+1, elapsed_seconds=time.monotonic()-started)
            history.append(row); stream.write(json.dumps(row)+'\n')
            if update == 0 or (update+1) % 25 == 0 or args.preflight_only:
                stream.flush(); print(json.dumps(row), flush=True)
                if run:
                    run.log({'train/loss': row['loss'], 'train/update': update+1}, step=update+1)
    checkpoint = args.output / 'checkpoint-final.pt'
    torch.save({'scope': 'unpromoted-current-patch-RGB-pilot', 'arm': args.arm, 'seed': args.seed,
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'completed_updates': steps,
                'input_summary': reference(args.inputs / 'summary.json')}, checkpoint)
    # Deliberately omit the production schema and matcher-contract export.
    write_new(args.output / 'checkpoint-freeze.json', {'checkpoint': reference(checkpoint),
              'learned_calibration_values_opened': False, 'completed_updates': steps})
    evaluation_role = 'train' if args.preflight_only else 'calibration'
    rows = evaluate(model, [r for r in inputs['records'] if r['role'] == evaluation_role],
                    args.labels, args.output, allowed_role=evaluation_role)
    eligible_rows = [r for r in rows if r['queries'] > 0]
    result = {'status': 'PASS_PREFLIGHT_ONLY' if args.preflight_only else 'PASS',
              'scope': 'current-DeformNeXt-controlled-patch-RGB-pilot', 'arm': args.arm, 'seed': args.seed,
              'completed_updates': steps, 'checkpoint': reference(checkpoint),
              'training_assets_with_queries': len(training), 'calibration_assets': len(rows) if not args.preflight_only else 0,
              'evaluated_calibration_assets': len(eligible_rows) if not args.preflight_only else 0,
              'task_rgb_l1_pair_mean': float(np.mean([r['task_rgb_l1'] for r in eligible_rows])) if eligible_rows and not args.preflight_only else None,
              'calibration_rows': rows if not args.preflight_only else [],
              'preflight_training_pair_rows': rows if args.preflight_only else [],
              'initial_D0_D1_RAW_scores_exact': True,
              'calibration_opened_after_final_checkpoint_freeze': not args.preflight_only,
              'elapsed_seconds': time.monotonic()-started,
              'peak_GPU_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
              'production_promotion': False, 'SS_gain_claim': False}
    write_new(args.output / 'summary.json', result)
    if run:
        for path in sorted(args.output.glob('cue-panel-*.png')):
            run.log({'calibration/cue0_GT_Pred': wandb.Image(str(path), caption='Held-out cue0, patch-space GT / transported RGB; not SS geometry')})
        run.summary.update({k: v for k, v in result.items() if isinstance(v, (float, int, bool, str))})
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, choices=[42017, 42018, 42019], default=42017)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--wandb-id')
    args = parser.parse_args()
    torch.set_num_threads(2)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    train(args)


if __name__ == '__main__':
    main()
