"""Warm up only two ray Q/K tensors; all reconstruction weights stay untouched.

The existing 512/32 auxiliary split belongs to SS TRAIN, not reconstruction
validation. Multi-positive GT sets enter loss only. Final epoch7 is fixed in
advance; improved matching cannot establish improved generated geometry.
"""
from collections import OrderedDict
from pathlib import Path
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from pixal3d.modules.ray_prior_anchored_readout import (
    PriorAnchoredRayFeatureReadout, correspondence_probabilities)
from . import ray_set_head_training as shared
from .anchor_patch_head_training import validate_query_domain


class Observations:
    def __init__(self, config, cache_root, rows, closed):
        self.config, self.root = config, cache_root
        self.metadata = {r['sample_id']: r for r in map(json.loads,
            Path(config['data_dir']).read_text().splitlines())}
        self.closed = closed
        self.bound = {}
        self.recent = OrderedDict()
        if any(self.metadata[r['sample_id']]['split'] != 'train' for r in rows):
            raise ValueError('all auxiliary assets must belong to SS TRAIN')

    def load(self, row):
        sample = row['sample_id']
        if sample in self.recent:
            self.recent.move_to_end(sample)
            return self.recent[sample]
        for field in ['inputs', 'labels']:
            path = self.root / row[field + '_relative_path']
            if str(path) not in self.bound:
                digest = shared.sha(path)
                if digest != row[field + '_SHA256'] or self.closed[str(path)] != digest:
                    raise ValueError('closed auxiliary observation/label bytes changed')
                self.bound[str(path)] = digest
        x, y = shared.load_asset(self.root, row)
        with np.load(self.root / row['inputs_relative_path'], allow_pickle=False) as archive:
            patches = archive['query_patch_indices'].astype(np.int64)
        original = self.metadata[sample]
        pathmap = self.config['dataset']['args']['path_map']
        raw = original['transport_cache']
        if not raw.startswith('/file4/') or '/file4' not in pathmap:
            raise ValueError('explicit canonical file4 transport mapping required')
        transport = Path(pathmap['/file4'] + raw[len('/file4'):])
        match = Path(self.config['dataset']['args']['match_root']) / (sample + '.npz')
        for path in [transport, match]:
            if str(path) not in self.bound:
                self.bound[str(path)] = shared.sha(path)
        with np.load(transport, allow_pickle=False) as archive:
            if (str(archive['conditioning_input_sha256']) != original['conditioning_input_sha256']
                    or not np.array_equal(archive['source_values'].reshape(3072, 1024), x['source_value_bank'])):
                raise ValueError('current native RGB transport differs from the closed auxiliary cache')
            x['source_key_bank'] = archive['source_keys'][..., :1024].reshape(3072, 1024).copy()
        with np.load(match, allow_pickle=False) as archive:
            if (not np.array_equal(archive['semantic_indices'][patches], x['candidate_indices'])
                    or not np.array_equal(archive['candidate_valid'][patches], x['candidate_valid'])
                    or not np.array_equal(archive['matching_logits'][patches] * 20., x['scalar_prior_logits'])):
                raise ValueError('current semantic Top64/prior differs from the closed cache')
        self.recent[sample] = (x, y)
        while len(self.recent) > 16:
            self.recent.popitem(last=False)
        return x, y

    def batch(self, rows, rng):
        fields = {name: [] for name in ['native', 'context', 'keys', 'values', 'valid', 'prior']}
        relations, positives, offset = [], [], 0
        for row in rows:
            x, y = self.load(row)
            chosen = rng.choice(y['eligible'], size=512, replace=len(y['eligible']) < 512)
            ids = x['candidate_indices'][chosen].astype(np.int64)
            for name, value in dict(native=x['native_query'][chosen], context=x['context_query'][chosen],
                    keys=x['source_key_bank'][ids], values=x['source_value_bank'][ids],
                    valid=x['candidate_valid'][chosen], prior=x['scalar_prior_logits'][chosen]).items():
                fields[name].append(value)
            lengths = y['pointer'][chosen + 1] - y['pointer'][chosen]
            relations.append(np.repeat(np.arange(512) + offset, lengths))
            positives.append(np.concatenate([y['near1'][y['pointer'][q]:y['pointer'][q+1]] for q in chosen]))
            offset += 512
        args = [torch.from_numpy(np.concatenate(fields[name])).cuda().to(
            dtype=torch.bool if name == 'valid' else torch.float32) for name in fields]
        return args, torch.from_numpy(np.concatenate(relations).astype(np.int64)).cuda(), \
            torch.from_numpy(np.concatenate(positives)).cuda()


def evaluate(head, observations, rows):
    head.eval()
    counts = {kind: dict(mass_sum=0., hit1=0, proposal64_hits=0) for kind in ['exact', 'near1']}
    count, per_asset = 0, []
    with torch.no_grad():
        for row in rows:
            x, y = observations.load(row)
            parts = []
            for begin in range(0, len(x['native_query']), 64):
                take = slice(begin, begin + 64)
                ids = x['candidate_indices'][take].astype(np.int64)
                args = [torch.from_numpy(v).cuda() for v in [x['native_query'][take].astype(np.float32),
                    x['context_query'][take].astype(np.float32), x['source_key_bank'][ids].astype(np.float32),
                    x['source_value_bank'][ids].astype(np.float32), x['candidate_valid'][take],
                    x['scalar_prior_logits'][take].astype(np.float32)]]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    parts.append(correspondence_probabilities(head, *args).mean(1).cpu())
            p = torch.cat(parts) if parts else torch.empty(0, 64)
            relation = torch.from_numpy(y['relation'])
            metrics = {}
            for kind in counts:
                label = torch.from_numpy(y[kind])
                metrics[kind] = dict(mass_sum=float((p[relation] * label).sum()),
                    hit1=int(label.gather(1, p[relation].argmax(-1)[:, None]).sum()),
                    proposal64_hits=int(label.any(-1).sum()))
                for key, value in metrics[kind].items():
                    counts[kind][key] += value
            count += len(relation)
            per_asset.append(dict(sample_id=row['sample_id'], source_cluster_id=row['source_cluster_id'],
                observable_relations=len(relation), metrics=metrics))
    return dict(assets=len(rows), observable_relations=count, per_asset=per_asset,
        **{kind: dict(mean_probability_mass=v['mass_sum']/count, top1_coverage=v['hit1']/count,
            proposal64_coverage=v['proposal64_hits']/count) for kind, v in counts.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['profile', 'train'], required=True)
    for flag in ['config', 'cache-root', 'cache-receipt', 'campaign-root', 'cold-conditioner', 'output']:
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--profile-proof', type=Path)
    p.add_argument('--microqueries', type=int, default=8192)
    p.add_argument('--wandb', action='store_true')
    args = p.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if physical not in {'0', '1', '2', '3'} or physical not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one explicitly assigned GPU0-3 required')
    if args.microqueries < 512 or args.microqueries % 512:
        raise ValueError('whole assets with512 query draws required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(101)
    torch.cuda.manual_seed_all(101)
    from pixal3d.modules import ray_prior_anchored_readout, ray_feature_attention, task_matching_attention
    bound = {str(path.resolve()): shared.sha(path) for path in [Path(__file__), Path(shared.__file__),
        Path(ray_prior_anchored_readout.__file__), Path(ray_feature_attention.__file__),
        Path(task_matching_attention.__file__), Path(__file__).with_name('anchor_patch_head_training.py'),
        args.config, args.cold_conditioner, args.cache_root/'cache-index.jsonl', args.cache_root/'completed.json',
        args.cache_receipt]}
    cfg = shared.read(args.config)
    if cfg['trainer']['args']['deform_conditioner']['score_temperature'] != .05:
        raise ValueError('same declared native score temperature required')
    bound[str(Path(cfg['data_dir']))] = shared.sha(cfg['data_dir'])
    rows = [json.loads(line) for line in (args.cache_root/'cache-index.jsonl').read_text().splitlines()]
    train, calibration = validate_query_domain(args.cache_root, rows)
    if len(train) != 503 or len(calibration) != 32:
        raise ValueError('unchanged declared auxiliary split required')
    receipt = shared.read(args.cache_receipt)
    if receipt.get('status') != 'PASS' or receipt.get('local_equals_actual_file4_SHA256') is not True:
        raise ValueError('actual file4 auxiliary-cache closure required')
    closed = {str(args.campaign_root.parent/r['relative_path']): r['sha256'] for r in receipt['files']}
    if closed[str(args.cache_root/'cache-index.jsonl')] != shared.sha(args.cache_root/'cache-index.jsonl'):
        raise ValueError('closed auxiliary index differs')
    observations = Observations(cfg, args.cache_root, rows, closed)
    args.output.mkdir(parents=True, exist_ok=False)
    state = torch.load(args.cold_conditioner, map_location='cpu', weights_only=True)
    state = {k[len('ray_readout.'):]: v for k, v in state.items() if k.startswith('ray_readout.')}
    if len(state) != 9 or torch.count_nonzero(state['output.weight']):
        raise ValueError('untouched full nine-tensor zero-output readout required')
    head = PriorAnchoredRayFeatureReadout().cuda()
    head.load_state_dict(state, strict=True)
    head.query.weight.data.zero_()
    head.requires_grad_(False)
    head.query.requires_grad_(True)
    head.key.requires_grad_(True)
    initial = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    optimizer = torch.optim.AdamW([head.query.weight, head.key.weight], lr=1e-4, weight_decay=.01)
    common = dict(source_SHA256=bound, query_unit='edited-target-patch-ray', epochs=7,
        train_selected_assets=512, train_supervised_assets=503, calibration_assets=32,
        split='SS-TRAIN;heldoutonlyfromauxiliaryutilityandQKfit', microqueries=args.microqueries,
        queries_per_asset=512, accumulation=1, physical_GPU=physical, allocator_fraction=.2,
        optimizer_tensors=2, query_initialization='zero-weights-not-zero-observations',
        score_selector_frozen=True, value_output_frozen=True, labels_in_forward=False,
        SS_updates=0, R_updates=0, model_adopted=False, quality_gain_claimed=False, full_goal_complete=False)

    def guard(all_inputs=False):
        for path, digest in dict(bound, **(observations.bound if all_inputs else {})).items():
            if shared.sha(path) != digest:
                raise ValueError('frozen source or read-only input changed: ' + path)

    def update(batch):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        values, relation, positives = batch
        with torch.autocast('cuda', dtype=torch.bfloat16):
            probability = correspondence_probabilities(head, *values)
        loss = shared.ray_set_logit_loss(probability.mean(1).clamp_min(1e-30).log(), relation, positives)[0]
        loss.backward()
        gradients = {name: float(parameter.grad.abs().sum()) for name, parameter in head.named_parameters()
                     if parameter.requires_grad}
        norm = torch.nn.utils.clip_grad_norm_([head.query.weight, head.key.weight], 1.)
        if not math.isfinite(float(loss)) or not math.isfinite(float(norm)):
            raise ValueError('finite actual Q/K loss and gradients required')
        optimizer.step()
        return float(loss.detach()), gradients

    guard()
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    per_step = args.microqueries//512
    if args.mode == 'profile':
        losses, gradients = [], []
        for step in range(3):
            batch = observations.batch([train[i % len(train)] for i in range(per_step)], np.random.default_rng(101))
            if step == 0:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    chunk = [v[:64] for v in batch[0]]
                    actual = head(*chunk)[1]
                    fast = correspondence_probabilities(head, *chunk)
                    prior = head.score_selector(chunk[0], chunk[1], chunk[3], chunk[4], chunk[5])[1]
                    if not torch.equal(actual, fast) or not torch.equal(actual, prior[:, None].expand_as(actual)):
                        raise ValueError('native CUDA probabilities must exactly preserve pretrained utility')
            loss, gradient = update(batch)
            losses.append(loss)
            gradients.append(gradient)
            del batch
        torch.cuda.synchronize()
        changed = [k for k, v in head.state_dict().items() if not torch.equal(v.cpu(), initial[k])]
        if set(changed) != {'query.weight', 'key.weight'} or len(optimizer.state) != 2:
            raise ValueError('only actual Q/K tensors may change')
        guard(True)
        shared.write_new(args.output/'profile.json', dict(common,
            status='PASS_ACTUAL_PRIOR_ANCHORED_QK_CAPACITY_AND_NATIVE_PARITY', temporary_updates=3,
            changed_tensors=changed, gradients=gradients, losses=losses, native_probabilities_exact=True,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30, seconds=time.monotonic()-began,
            profile_model_discarded=True))
        print(json.dumps(dict(status='PASS_QK_PROFILE', losses=losses,
            peak_GiB=torch.cuda.max_memory_allocated()/2**30)), flush=True)
        return
    if args.profile_proof is None:
        raise ValueError('same-source actual CUDA profile required')
    proof = shared.read(args.profile_proof)
    if proof['status'] != 'PASS_ACTUAL_PRIOR_ANCHORED_QK_CAPACITY_AND_NATIVE_PARITY' or any(
            proof[key] != common[key] for key in ['source_SHA256', 'microqueries', 'physical_GPU']):
        raise ValueError('actual capacity proof does not match this fixed training')
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(entity='youngwoojeon-unist', project='pixal3d-deformnext-retargeting',
            id='ss-prior-anchored-QK-v1-s101-512-e7', name='Prior anchored ray QK 512 epoch7',
            resume='never', dir=str(args.output), config=common)
    updates, history = 0, []
    for epoch in range(8):
        losses = []
        if epoch:
            rng = np.random.default_rng(101+epoch)
            order = rng.permutation(len(train))
            for begin in range(0, len(order), per_step):
                loss, _ = update(observations.batch([train[i] for i in order[begin:begin+per_step]], rng))
                losses.append(loss)
                updates += 1
                if updates % 8 == 0:
                    guard()
                    progress = dict(status='TRAINING_QK_ONLY', epoch=epoch, optimizer_updates=updates,
                        mean_loss=float(np.mean(losses)), seconds=time.monotonic()-began, SS_updates=0)
                    (args.output/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
                    print(json.dumps(progress), flush=True)
        metrics = evaluate(head, observations, calibration)
        row = dict(epoch=epoch, optimizer_updates=updates, calibration=metrics,
            training_loss=float(np.mean(losses)) if losses else None, seconds=time.monotonic()-began)
        history.append(row)
        with (args.output/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        exported = {k: v.detach().cpu() for k, v in head.state_dict().items()}
        torch.save(dict(head=exported, optimizer=optimizer.state_dict(), epoch=epoch, **common),
            args.output/f'head-epoch{epoch:02d}.pt')
        if run:
            run.log(dict(epoch=epoch, **{f'calibration/{kind}_{key}': value
                for kind in ['exact', 'near1'] for key, value in metrics[kind].items()}), step=epoch)
        print(json.dumps({k: v for k, v in row.items() if k != 'calibration'} |
            dict(calibration={k: metrics[k] for k in ['exact', 'near1']})), flush=True)
    changed = [k for k, v in exported.items() if not torch.equal(v, initial[k])]
    if set(changed) != {'query.weight', 'key.weight'} or torch.count_nonzero(exported['output.weight']):
        raise ValueError('only Q/K may change; exact zero feature output must survive')
    guard(True)
    final = args.output/'ray-readout-QK-epoch07.pt'
    torch.save(exported, final)
    shared.write_new(args.output/'completed.json', dict(common,
        status='PASS_FIXED_SEVEN_EPOCH_PRIOR_ANCHORED_QK', optimizer_updates=updates,
        changed_tensors=changed, final_head=str(final), final_head_SHA256=shared.sha(final),
        final_calibration=history[-1]['calibration'], observed_input_SHA256=observations.bound,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30, seconds=time.monotonic()-began))
    if run:
        run.finish()


if __name__ == '__main__':
    main()
