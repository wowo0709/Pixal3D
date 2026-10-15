#!/usr/bin/env python3
"""Train the exact R9 D1 architecture with an explicit topology-visible teacher.

This is a new DeformNeXt supervision objective, not reproduction of R9's
controlled RGB training loss. No spatial pseudo-label fallback is permitted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from pixal3d.modules.r9_matching import CHECKPOINT_SCHEMA, MATCHER_CONTRACT, frame_raw, frame_topk
from .precompute_matches import remap
from .topology_labels import SCHEMA as LABEL_SCHEMA, fingerprint


def topology_listwise_loss(logits, indices, candidate_valid, positives, visible):
    """Visible-query denominator includes proposal misses; nulls are separate."""
    if positives.shape != visible.shape or positives.shape[0] != logits.shape[0]:
        raise ValueError('topology teacher shape mismatch')
    positive = ((indices[..., None] == positives[:, None]) & visible[:, None]).any(-1) & candidate_valid
    observable = visible.any(-1)
    eligible = positive.any(-1)
    masked = logits.masked_fill(~candidate_valid, -torch.inf)
    if eligible.any():
        selected, mask = masked[eligible], positive[eligible]
        loss = (torch.logsumexp(selected, -1) - torch.logsumexp(selected.masked_fill(~mask, -torch.inf), -1)).mean()
        correct = int(mask.gather(1, selected.argmax(-1, keepdim=True)).sum())
    else:
        loss = logits.masked_fill(~candidate_valid, 0).sum() * 0
        correct = 0
    return loss, {'eligible_queries': int(eligible.sum()), 'top1_correct': correct,
                  'raw_top1_correct': int(positive[:, 0].sum()), 'visible_queries': int(observable.sum()),
                  'proposal_misses': int((observable & ~eligible).sum()), 'null_queries': int((~observable).sum())}


def loss_for_cache(model, path, label_path, *, sample_id, split, topk=64):
    device = next(model.parameters()).device
    with np.load(path) as cache:
        target = frame_raw(torch.from_numpy(cache['target_keys'].copy()).to(device).float())
        source = frame_raw(torch.from_numpy(cache['source_keys'].copy()).to(device).float())
        valid = torch.from_numpy(cache['source_valid'].copy()).to(device).bool()
    with np.load(label_path) as label:
        if (str(label['schema']) != LABEL_SCHEMA or str(label['sample_id']) != sample_id
                or str(label['split']) != split or float(label['visibility_depth_tolerance']) != .005):
            raise ValueError('topology teacher provenance mismatch')
        positives = torch.from_numpy(label['positive_indices'].copy()).to(device).long()
        visible = torch.from_numpy(label['positive_visible'].copy()).to(device).bool()
    flat = source.flatten(0, 1)
    cosine, indices = frame_topk(target[None], source[None], valid[None], topk)
    cosine, indices = cosine[0], indices[0]
    logits = model._q2c_logits(target[None], flat[indices][None], cosine.nan_to_num(-1.)[None])[0]
    return topology_listwise_loss(logits, indices, cosine.isfinite(), positives, visible)


def evaluate(model, rows, mappings, label_root):
    model.eval()
    counts = {k: 0 for k in ('eligible_queries', 'top1_correct', 'raw_top1_correct', 'visible_queries', 'proposal_misses', 'null_queries')}
    loss_sum = 0.
    with torch.no_grad():
        for row in rows:
            loss, metrics = loss_for_cache(model, remap(row['transport_cache'], mappings),
                label_root / f"{row['sample_id']}.npz", sample_id=row['sample_id'], split=row['split'])
            loss_sum += float(loss) * metrics['eligible_queries']
            for key in counts: counts[key] += metrics[key]
    model.train()
    return {**counts, 'samples': len(rows), 'loss': loss_sum / max(counts['eligible_queries'], 1),
            'q2c_top1_accuracy': counts['top1_correct'] / max(counts['visible_queries'], 1),
            'raw_cosine_top1_accuracy': counts['raw_top1_correct'] / max(counts['visible_queries'], 1),
            'topk_recall': counts['eligible_queries'] / max(counts['visible_queries'], 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--validation-manifest', type=Path, required=True)
    parser.add_argument('--label-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--path-map', action='append', default=[])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--topk', type=int, choices=[64], default=64)
    parser.add_argument('--seed', type=int, default=42017)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    mappings = [tuple(v.split('=', 1)) for v in args.path_map]
    resolve = lambda path: remap(path, mappings)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    validation = [json.loads(line) for line in args.validation_manifest.read_text().splitlines() if line]
    if not rows or not validation: raise ValueError('train and validation populations must be nonempty')
    if any(r['split'] != 'train' for r in rows) or any(r['split'] != 'validation' for r in validation):
        raise ValueError('incorrect matcher train/validation split')
    for key in ('sample_id', 'source_uid', 'source_cluster_id'):
        a, b = {r[key] for r in rows if key in r}, {r[key] for r in validation if key in r}
        if a & b: raise ValueError(f'matcher train/validation overlap: {key}')
    input_hashes = []
    for row in rows + validation:
        digest = fingerprint(row, resolve)
        with np.load(args.label_root / f"{row['sample_id']}.npz") as label:
            if str(label['input_sha256']) != digest: raise ValueError('stale topology teacher input hash')
        input_hashes.append(digest)
    model = DeformationAwareConditioner(arm='b2', key_dim=1024, value_dim=1024, query_dim=1024).cuda()
    optimizer = torch.optim.AdamW(model.q2c.parameters(), lr=args.lr, weight_decay=.01)
    rng = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    events = args.output.with_suffix('.jsonl')
    history = [{'epoch': 0, 'validation': evaluate(model, validation, mappings, args.label_root)}]
    print(json.dumps(history[0]), flush=True)
    step = 0
    for epoch in range(1, args.epochs+1):
        for index in rng.permutation(len(rows)):
            row = rows[int(index)]
            loss, counts = loss_for_cache(model, resolve(row['transport_cache']),
                args.label_root / f"{row['sample_id']}.npz", sample_id=row['sample_id'], split='train')
            if not counts['eligible_queries']: continue
            if not torch.isfinite(loss): raise ValueError('nonfinite R9 loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.q2c.parameters(), 1.)
            optimizer.step(); step += 1
        record = {'epoch': epoch, 'validation': evaluate(model, validation, mappings, args.label_root)}
        history.append(record); print(json.dumps(record), flush=True)
        events.write_text(''.join(json.dumps(x)+'\n' for x in history))
    # Fixed final update follows the original checkpoint selection rule. No
    # best-epoch selection on deformation validation metrics.
    tmp = args.output.with_name(f'.{args.output.name}.{os.getpid()}.tmp')
    torch.save({'schema': CHECKPOINT_SCHEMA, 'matcher_contract': MATCHER_CONTRACT,
                'training_objective': 'topology-visible-listwise-v1',
                'teacher_schema': LABEL_SCHEMA, 'teacher_inputs_sha256': hashlib.sha256(''.join(input_hashes).encode()).hexdigest(),
                'q2c': {k: v.detach().cpu() for k, v in model.q2c.state_dict().items()},
                'epochs': args.epochs, 'steps': step, 'seed': args.seed, 'validation': history,
                'train_samples': len(rows), 'validation_samples': len(validation)}, tmp)
    os.replace(tmp, args.output)


if __name__ == '__main__': main()
