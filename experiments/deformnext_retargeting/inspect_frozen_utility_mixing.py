"""Actual saved utility reweighting on development RGB/descriptor caches, CPU only."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from pixal3d.modules.r9_matching import PromotedR9Matcher, categorical_weights, frame_raw
from pixal3d.modules.task_matching_attention import TaskCandidateAttention, contextual_query
from .frozen_matcher_utility import validate_profile
from .semantic_topk_search import write_new
from .summarize_frozen_utility_followup import NAME


def read(path):
    return json.loads(path.read_text())


def fingerprint(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def quantiles(value):
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError('nonempty finite descriptive values required')
    result = torch.quantile(value.double().flatten(), torch.tensor([0., .5, .9, .99, 1.], dtype=torch.float64))
    return dict(zip(['min', 'median', 'p90', 'p99', 'max'], result.tolist()))


def distribution_stats(probability):
    if (probability.ndim != 2 or not torch.isfinite(probability).all() or (probability < 0).any()
            or (probability.sum(-1) - 1).abs().max() > 1e-5):
        raise ValueError('finite normalized candidate probabilities required')
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
    return {'mean_max_weight': float(probability.max(-1).values.mean()),
            'mean_entropy_nats': float(entropy.mean()),
            'mean_effective_candidates': float(entropy.exp().mean())}


def contrast(candidate, reference, values):
    left, right = (candidate[..., None] * values).sum(1), (reference[..., None] * values).sum(1)
    return {'argmax_disagreement_fraction': float((candidate.argmax(-1) != reference.argmax(-1)).float().mean()),
            'mean_total_variation': float((candidate - reference).abs().sum(-1).mean() / 2),
            'mean_KL_candidate_vs_reference': float((candidate * (candidate.clamp_min(1e-30).log()
                                                         - reference.clamp_min(1e-30).log())).sum(-1).mean()),
            'value_squared_error_sum': float((left - right).square().sum()),
            'value_reference_squared_sum': float(right.square().sum()),
            'value_mean_cosine': float(F.cosine_similarity(left, right, dim=-1).mean())}


def summarize_contrasts(cases, label):
    rows, counts = [case['contrasts'][label] for case in cases], [case['active_queries'] for case in cases]
    total = sum(counts)
    result = {key: sum(count * row[key] for count, row in zip(counts, rows)) / total
              for key in ['argmax_disagreement_fraction', 'mean_total_variation',
                          'mean_KL_candidate_vs_reference', 'value_mean_cosine']}
    error, reference = sum(row['value_squared_error_sum'] for row in rows), sum(row['value_reference_squared_sum'] for row in rows)
    result['value_aggregate_relative_RMS'] = (error / max(reference, 1e-30)) ** .5
    return result


def inspect(args):
    if args.cases <= 0 or args.cases > 128 or args.step not in [96, 192, 288, 384, 480]:
        raise ValueError('declared development prefix and actual epoch checkpoint required')
    if args.output.exists():
        raise FileExistsError('preserve existing mixing diagnosis')
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    config_path = args.root / 'configs' / (NAME + '.json')
    cfg = read(config_path)
    validate_profile(cfg, args.root)
    checkpoint = args.root / 'runs_ss64' / NAME / 'ckpts' / f'deform_conditioner_step{args.step:07d}.pt'
    closure_path = args.root / 'epoch-audits-v1' / ('train-' + NAME) / f'epoch{args.step // 96:02d}/file4-closure.json'
    closure, stamp = read(closure_path), fingerprint(checkpoint)
    relative = str(checkpoint.relative_to(args.root.parent.parent))
    entries = [item for item in closure['files'] if item['relative_path'] == relative]
    if (closure['status'] != 'PASS' or len(entries) != 1
            or any(entries[0][key] != stamp[key] for key in ['sha256', 'bytes'])):
        raise ValueError('actual saved conditioner differs from canonical epoch receipt')
    spec = cfg['trainer']['args']['deform_conditioner']
    head_path = Path(spec['q2c_checkpoint'])
    matcher = PromotedR9Matcher().eval()
    matcher.load_checkpoint(torch.load(head_path, map_location='cpu', weights_only=False))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if any(not torch.equal(saved['q2c.' + key], value) for key, value in matcher.state_dict().items()):
        raise ValueError('frozen original24 matching tensors changed')
    attention = TaskCandidateAttention(content_mode='residual-zero').eval()
    attention.load_state_dict({key.removeprefix('task_attention.'): value for key, value in saved.items()
                               if key.startswith('task_attention.')}, strict=True)
    zero = copy.deepcopy(attention)
    zero.content_output.weight.zero_()
    manifest = Path(cfg['validation']['roots_file'])
    rows = {row['sample_id']: row for row in (json.loads(line) for line in manifest.read_text().splitlines() if line)}
    ids = read(args.panel)['sample_ids'][:args.cases]
    if len(ids) != args.cases or len(set(ids)) != args.cases:
        raise ValueError('fixed development prefix differs')
    reference_panel = read(args.root.parent / 'evaluations/top1-reference/noise-0/evaluation.json')['quick_panel_ids']
    if read(args.panel)['sample_ids'] != reference_panel:
        raise ValueError('panel is not the original fixed development128')

    def local(value):
        if not value.startswith('/file4/'):
            raise ValueError('explicit canonical file4 cache path required')
        return args.cache_root / value.removeprefix('/')

    cases, distributions, inputs, margins, spans, logits_abs = [], {}, [], [], [], []
    zero_max_error, score_max_error, categorical_max_error = 0., 0., 0.
    for sid in ids:
        row = rows[sid]
        transport, match, image = local(row['transport_cache']), Path(cfg['validation']['dataset_args']['match_root']) / (sid + '.npz'), local(row['target_anchor'])
        with np.load(transport) as z:
            if str(z['conditioning_input_sha256']) != row['conditioning_input_sha256']:
                raise ValueError('RGB transport conditioning identity differs')
            query = frame_raw(torch.from_numpy(z['target_keys'].copy())).float()
            native = torch.from_numpy(z['target_query'].copy()).float()
            keymap = frame_raw(torch.from_numpy(z['source_keys'].copy())).float().flatten(0, 1)
            valuemap = torch.from_numpy(z['source_values'].copy()).float().flatten(0, 1)
            source_valid = torch.from_numpy(z['source_valid'].copy()).bool().flatten()
        with np.load(match) as z:
            if str(z['conditioning_input_sha256']) != row['conditioning_input_sha256']:
                raise ValueError('semantic proposal conditioning identity differs')
            indices, valid = torch.from_numpy(z['semantic_indices'].copy()).long(), torch.from_numpy(z['candidate_valid'].copy()).bool()
            cached = torch.from_numpy(z['matching_logits'].copy()).float()
        with Image.open(image) as rgba:
            alpha = np.asarray(rgba.convert('RGBA').resize((512, 512), Image.Resampling.NEAREST))[..., 3]
        target_valid = torch.from_numpy((alpha[8::16, 8::16] > 127).copy()).flatten()
        valid &= source_valid[indices] & target_valid[:, None] & (source_valid.sum() >= 64)
        active = valid.any(-1)
        query, native, indices, valid, cached = [value[active] for value in [query, native, indices, valid, cached]]
        if not len(query) or not bool(valid.all()):
            raise ValueError('this prefix requires explicit full64 active support')
        keys, values = keymap[indices], valuemap[indices]
        context = contextual_query(matcher.decoder, query, keys, valid)
        nq, nk = F.normalize(context, dim=-1), F.normalize(keys, dim=-1)
        cosine = (nq[:, None] * nk).sum(-1)
        score = matcher.fusion_head(nq, nk, cosine[..., None], cosine)
        raw = (F.normalize(query, dim=-1)[:, None] * F.normalize(keys, dim=-1)).sum(-1)
        original_score = matcher.score_candidates(query, keys, raw)
        score_max_error = max(score_max_error, float((score - original_score).abs().max()))
        prior = score / spec['score_temperature']
        categorical_probability = categorical_weights(score, valid, spec['score_temperature'])
        # The task path scales first, then subtracts the maximum. Reuse
        # that exact FP32 operation order for the utility-only comparison.
        prior_probability = categorical_weights(prior, valid, 1.)
        categorical_max_error = max(categorical_max_error,
                                    float((prior_probability - categorical_probability).abs().max()))
        cached_probability = categorical_weights(cached, valid, spec['score_temperature'])
        _, probability, total_logits = attention(native, context, values, valid, prior)
        _, utility_probability, utility_logits = attention(native, context, values, valid, None)
        _, zero_probability, _ = zero(native, context, values, valid, prior)
        zero_max_error = max(zero_max_error, float((zero_probability - prior_probability).abs().max()))
        torch.testing.assert_close(total_logits, prior + utility_logits, rtol=0, atol=1e-6)
        torch.testing.assert_close(zero_probability, prior_probability, rtol=0, atol=0)
        top = prior.topk(2, dim=-1).values
        margin, span = top[:, 0] - top[:, 1], utility_logits.max(-1).values - utility_logits.min(-1).values
        margins.append(margin)
        spans.append(span)
        logits_abs.append(utility_logits.abs().max(-1).values)
        hard = torch.zeros_like(prior_probability).scatter(1, prior.argmax(-1, keepdim=True), 1.)
        probabilities = {'cached_prior': cached_probability, 'fresh_frozen_prior': prior_probability,
                         'fresh_original_categorical_prior': categorical_probability,
                         'trained_total': probability, 'utility_without_prior': utility_probability,
                         'hard_Top1': hard}
        for name, value in probabilities.items():
            distributions.setdefault(name, []).append(value)
        contrasts = {name: contrast(probabilities[candidate], probabilities[reference], values)
                     for name, candidate, reference in [
                         ('trained-vs-prior', 'trained_total', 'fresh_frozen_prior'),
                         ('fresh-vs-cached-prior', 'fresh_frozen_prior', 'cached_prior'),
                         ('trained-vs-hard-Top1', 'trained_total', 'hard_Top1')]}
        cases.append({'sample_id': sid, 'source_cluster_id': row['source_cluster_id'], 'active_queries': len(query),
                      'prior_top1_margin': quantiles(margin), 'utility_logit_span': quantiles(span), 'contrasts': contrasts})
        inputs.append({'sample_id': sid, 'conditioning_input_sha256': row['conditioning_input_sha256'],
                       'files': [fingerprint(path) for path in [transport, match, image]]})
        print(json.dumps({'case': len(cases), 'active_queries': len(query),
                          'utility_vs_prior': contrasts['trained-vs-prior']}), flush=True)
    margin, span = torch.cat(margins), torch.cat(spans)
    repo = Path(__file__).resolve().parents[2]
    result = {'status': 'PASS_ACTUAL_SAVED_FROZEN_UTILITY_MIXING_DIAGNOSIS',
              'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'step': args.step,
              'device': 'CPU_FP32', 'torch_threads': 2, 'GPU_used': False, 'optimizer_updates': 0,
              'cases': len(cases), 'active_queries': sum(case['active_queries'] for case in cases),
              'source_clusters': len({case['source_cluster_id'] for case in cases}),
              'actual_frozen24_matching_tensors_exact': True,
              'actual_conditioner': stamp, 'actual_epoch_file4_closure': fingerprint(closure_path),
              'original_head': fingerprint(head_path), 'recipe': fingerprint(config_path),
              'panel': fingerprint(args.panel), 'manifest': fingerprint(manifest), 'inputs': inputs,
              'driver': fingerprint(Path(__file__)),
              'runtime_sources': [fingerprint(repo / 'pixal3d/modules' / name)
                                  for name in ['r9_matching.py', 'task_matching_attention.py']],
              'effective_content_scale': float(attention.log_content_scale.exp().clamp(max=20.)),
              'uncapped_content_scale': float(attention.log_content_scale.exp()),
              'zero_utility_replay_max_probability_error': zero_max_error,
              'task_vs_original_categorical_max_probability_error': categorical_max_error,
              'original_scorer_vs_task_prior_max_score_error': score_max_error,
              'prior_top1_logit_margin': quantiles(margin), 'utility_logit_span': quantiles(span),
              'utility_max_absolute_logit': quantiles(torch.cat(logits_abs)),
              'fraction_prior_top1_margin_exceeds_utility_span': float((margin > span).float().mean()),
              'attention_distributions': {name: distribution_stats(torch.cat(values)) for name, values in distributions.items()},
              'contrasts': {name: summarize_contrasts(cases, name) for name in cases[0]['contrasts']},
              'per_case': cases,
              'scope': 'Original development prefix in fixed order; actual saved frozen decoder/scalar head and learned utility, '
                       'all head/value arithmetic FP32 after the stored FP16 boundary. No hidden depth/GT matching labels, '
                       'SS generation or reserved121 outcomes. Argmax disagreement is not correspondence accuracy. '
                       'Prior dominance bounds possible argmax changes on these queries;not a quality cause or adoption gate.',
              'model_adopted': False, 'full_goal_complete': False}
    write_new(args.output / 'summary.json', result)
    np.savez_compressed(args.output / 'query-logit-ranges.npz', prior_margin=margin.numpy(), utility_span=span.numpy())
    print(json.dumps({'status': result['status'], 'distributions': result['attention_distributions'],
                      'contrasts': result['contrasts'], 'scale': result['effective_content_scale'],
                      'margin_exceeds_span': result['fraction_prior_top1_margin_exceeds_utility_span']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--cases', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    inspect(parser.parse_args())


if __name__ == '__main__':
    main()
