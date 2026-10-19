"""Loss-only multi-positive warm-up for the edited-target-patch ray composer.

This trains five utility tensors, never the SS denoiser or retargeter. Its
fixed final head must subsequently be tested with actual SS generation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from pixal3d.modules.task_matching_attention import TaskCandidateAttention
from . import ray_set_head_training as shared


QUERY_UNIT = 'edited-target-patch-ray'


def validate_query_domain(cache_root, rows):
    """Reject the previous pooled-source-ray caches, even with equal shapes."""
    if len(rows) != 544:
        raise ValueError('the declared 512 train plus 32 calibration assets are required')
    train = [row for row in rows if row['role'] == 'train']
    calibration = [row for row in rows if row['role'] == 'calibration']
    if len(train) != 512 or len(calibration) != 32:
        raise ValueError('declared split counts differ')
    if ({row['source_cluster_id'] for row in train}
            & {row['source_cluster_id'] for row in calibration}):
        raise ValueError('training and calibration source clusters overlap')
    for row in rows:
        if row.get('query_unit') != QUERY_UNIT:
            raise ValueError('edited-target-patch ray inputs are required')
        with np.load(cache_root / row['inputs_relative_path'], allow_pickle=False) as archive:
            if str(archive['query_unit']) != QUERY_UNIT:
                raise ValueError('input archive query domain differs')
            patches = archive['query_patch_indices']
            if (patches.shape != (row['queries'],) or (patches < 0).any()
                    or (patches >= 1024).any() or len(np.unique(patches)) != len(patches)):
                raise ValueError('one query per valid edited-target patch required')
    return [row for row in train if row['usable_queries'] > 0], calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['profile', 'train'], required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--head-initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--microqueries', type=int, default=16384)
    parser.add_argument('--queries-per-asset', type=int, default=512)
    parser.add_argument('--cache-receipt', type=Path, required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--profile-proof', type=Path)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (physical not in {'0', '1', '2', '3'}
            or physical not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')):
        raise ValueError('one explicitly assigned physical GPU0-3 is required')
    if args.microqueries <= 0 or args.queries_per_asset != 512 or args.microqueries % 512:
        raise ValueError('whole assets with the declared 512 patch-query draws are required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(101)
    torch.cuda.manual_seed_all(101)
    own_sha, helper_sha = shared.sha(__file__), shared.sha(shared.__file__)
    initial_sha = shared.sha(args.head_initial)

    def guard():
        if shared.sha(__file__) != own_sha or shared.sha(args.head_initial) != initial_sha:
            raise RuntimeError('training source or initializer changed during execution')
        shared.source_guard(args.cache_root, helper_sha)

    guard()
    completed = shared.read(args.cache_root / 'completed.json')
    if (completed['status'] != 'CACHE_COMPLETE' or completed['query_unit'] != QUERY_UNIT
            or not completed['all_actual_head_probabilities_byte_exact']
            or not completed['all_GT_fingerprints_current']
            or completed['head_initializer_SHA256'] != initial_sha):
        raise ValueError('complete actual target-patch cache and its initializer required')
    index = args.cache_root / 'cache-index.jsonl'
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    closure = shared.read(args.cache_receipt)
    if closure.get('status') != 'PASS' or closure.get('local_equals_actual_file4_SHA256') is not True:
        raise ValueError('actual file4 cache closure required before training or profile')
    closed = {row['relative_path']: row for row in closure['files']}
    for path in [index, args.cache_root / 'completed.json', args.head_initial,
                 *[args.cache_root / row[k + '_relative_path'] for row in rows for k in ['inputs', 'labels']]]:
        key = str(path.relative_to(args.campaign_root.parent))
        if key not in closed or shared.sha(path) != closed[key]['sha256']:
            raise ValueError('closed data-cache bytes differ')
    train, calibration = validate_query_domain(args.cache_root, rows)
    if len(train) != 503:
        raise ValueError('expected 503 assets with covered positive sets')
    args.output.mkdir(parents=True, exist_ok=False)
    head = TaskCandidateAttention(content_mode='residual-zero', initial_content_scale=16.).cuda()
    initial = torch.load(args.head_initial, map_location='cpu', weights_only=True)
    head.load_state_dict(initial, strict=True)
    if len(initial) != 5 or len(list(head.parameters())) != 5:
        raise ValueError('exactly five utility tensors required')
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
    common = dict(query_unit=QUERY_UNIT, train_selected_assets=512, train_supervised_assets=len(train),
        calibration_assets=32, queries_per_asset=512, microqueries=args.microqueries, accumulation=1,
        training_source_SHA256=own_sha, shared_helper_SHA256=helper_sha, initializer_SHA256=initial_sha,
        GPU=physical, allocator_fraction=.2, GT_in_forward=False, SS_updates=0, R_updates=0,
        model_adopted=False, full_goal_complete=False)

    def update(batch):
        optimizer.zero_grad(set_to_none=True)
        _, _, logits = shared.forward_head(head, batch)
        loss, _ = shared.ray_set_logit_loss(logits, batch['relation'].cuda(), batch['positive'].cuda())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        if not math.isfinite(float(loss)) or not math.isfinite(float(norm)):
            raise ValueError('nonfinite utility loss or gradient')
        optimizer.step()
        return float(loss.detach()), float(norm)

    per_step = args.microqueries // 512
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if args.mode == 'profile':
        chosen = [train[i % len(train)] for i in range(per_step)]
        loss, norm = update(shared.make_batch(args.cache_root, chosen, 512, np.random.default_rng(101)))
        torch.cuda.synchronize()
        if len(optimizer.state) != 5:
            raise ValueError('actual five-tensor optimizer proof failed')
        result = dict(common, status='PASS_ACTUAL_TARGET_PATCH_HEAD_CAPACITY', loss=loss,
            gradient_norm=norm, optimizer_states=5, temporary_updates=1, profile_model_discarded=True,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            elapsed_seconds=time.monotonic()-started)
        shared.write_new(args.output / 'profile.json', result)
        print(json.dumps(result), flush=True)
        return
    if args.profile_proof is None:
        raise ValueError('same-source actual capacity proof required')
    fit = shared.read(args.profile_proof)
    if (fit['status'] != 'PASS_ACTUAL_TARGET_PATCH_HEAD_CAPACITY'
            or any(fit[key] != common[key] for key in ['training_source_SHA256', 'shared_helper_SHA256',
                'initializer_SHA256', 'microqueries', 'GPU', 'query_unit'])):
        raise ValueError('actual profile does not match this training')
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(entity='youngwoojeon-unist', project='pixal3d-deformnext-retargeting',
            id='ss-anchor-patch-set-head-v1-s101-512-e7', name='Target patch ray utility5 train512 epoch7',
            resume='never', dir=str(args.output), config=dict(common, epochs=7))
    updates, history = 0, []
    for epoch in range(8):
        losses = []
        if epoch:
            rng = np.random.default_rng(101 + epoch)
            order = rng.permutation(len(train))
            head.train()
            for begin in range(0, len(order), per_step):
                batch = shared.make_batch(args.cache_root,
                    [train[i] for i in order[begin:begin+per_step]], 512, rng)
                loss, _ = update(batch)
                updates += 1
                losses.append(loss)
                guard()
                if updates % 8 == 0:
                    progress = dict(status='TRAINING_TARGET_PATCH_UTILITY_ONLY', epoch=epoch,
                        optimizer_updates=updates, elapsed_seconds=time.monotonic()-started,
                        mean_training_loss=float(np.mean(losses)), SS_updates=0, R_updates=0)
                    (args.output / 'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
                    print(json.dumps(progress), flush=True)
                del batch
        metrics = shared.evaluate(head, args.cache_root, calibration)
        row = dict(epoch=epoch, optimizer_updates=updates, calibration=metrics,
            training_loss=float(np.mean(losses)) if losses else None,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            elapsed_seconds=time.monotonic()-started)
        history.append(row)
        with (args.output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        state = {key: value.detach().cpu() for key, value in head.state_dict().items()}
        torch.save(dict(head=state, optimizer=optimizer.state_dict(), epoch=epoch, updates=updates,
            **common), args.output / f'head-epoch{epoch:02d}.pt')
        if epoch == 0 and any(not torch.equal(value, initial[key]) for key, value in state.items()):
            raise ValueError('initialization drift before any training')
        if run:
            run.log({'head/epoch': epoch, 'head/optimizer_updates': updates,
                'head/calibration_loss': metrics['loss'],
                **{f'head/{kind}_{key}': value for kind in ['exact', 'near1']
                    for key, value in metrics[kind].items()},
                **({'head/training_loss': row['training_loss']} if losses else {})}, step=epoch)
        print(json.dumps(row), flush=True)
    changed = [key for key, value in state.items() if not torch.equal(value, initial[key])]
    if not changed or any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('finite learned utility tensors required')
    final = args.output / 'ray-composer-head-epoch07.pt'
    torch.save(state, final)
    guard()
    result = dict(common, status='PASS_FIXED_SEVEN_EPOCH_TARGET_PATCH_UTILITY_TRAINING', epochs=7,
        optimizer_updates=updates, changed_utility_tensors=changed, pure_utility_export=str(final),
        pure_utility_export_SHA256=shared.sha(final), final_calibration=history[-1]['calibration'],
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        elapsed_seconds=time.monotonic()-started,
        scope='Auxiliary target-patch positive-set supervision;not SS improvement or full fine-tuning.')
    shared.write_new(args.output / 'completed.json', result)
    if run:
        run.finish()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
