"""Measure the two declared score priors on unchanged development caches, CPU only."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def fingerprint(path):
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('input changed during read-only diagnosis')
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': stat.st_size}


def probability(logits, valid, temperature):
    if (logits.ndim != 2 or logits.shape != valid.shape or valid.dtype != np.bool_
            or not np.isfinite(logits[valid]).all() or not valid.any(1).all()
            or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError('finite active logits and positive temperature required')
    value = np.where(valid, logits.astype(np.float32) / temperature, -np.inf)
    value = value - value.max(1, keepdims=True)
    weight = np.where(valid, np.exp(value), 0.)
    return weight / weight.sum(1, keepdims=True)


def query_statistics(weights, view_index, views=3):
    if (weights.ndim != 2 or weights.shape != view_index.shape or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.allclose(weights.sum(1), 1., rtol=0, atol=1e-6)
            or (view_index < 0).any() or (view_index >= views).any()):
        raise ValueError('normalized finite candidate weights and declared view indices required')
    positive = np.maximum(weights, 1e-30)
    entropy = -(weights * np.log(positive)).sum(1)
    view_mass = np.stack([np.where(view_index == view, weights, 0.).sum(1) for view in range(views)], 1)
    view_entropy = -(view_mass * np.log(np.maximum(view_mass, 1e-30))).sum(1)
    ordered = np.sort(weights, axis=1)[:, ::-1]
    return {'maximum_weight': ordered[:, 0], 'entropy_nats': entropy,
        'effective_candidates': np.exp(entropy), 'effective_views': np.exp(view_entropy),
        'largest_view_mass': view_mass.max(1),
        'top4_mass': ordered[:, :4].sum(1), 'top8_mass': ordered[:, :8].sum(1)}


def summarize(values):
    return {key: {'mean': float(value.mean()),
        'median': float(np.median(value)), 'p90': float(np.quantile(value, .9))}
        for key, value in values.items()}


def inspect(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('diagnosis requires an explicitly CPU-only process')
    if args.output.exists():
        raise FileExistsError('preserve completed score-prior diagnosis')
    configs = [read(path) for path in [args.reference_config, args.soft_config]]
    specs = [cfg['trainer']['args']['deform_conditioner'] for cfg in configs]
    if ([spec['score_temperature'] for spec in specs] != [.005, .05]
            or any(spec['implementation'] != 'r9-native-v5' or spec['arm'] != 'c1' for spec in specs)
            or any(spec.get('correspondence_mode', 'semantic') != 'semantic' for spec in specs)):
        raise ValueError('the two declared frozen semantic score priors required')
    left, right = [dict(spec) for spec in specs]
    for spec in [left, right]: spec.pop('score_temperature')
    if left != right:
        raise ValueError('the two patch-query profiles differ beyond score temperature')
    if any(spec['task_attention'].get(key) is not False for spec in specs for key in ['train_decoder', 'train_score']):
        raise ValueError('both candidate priors must retain the frozen original head')
    for key in ['roots_file', 'dataset_args']:
        if configs[0]['validation'][key] != configs[1]['validation'][key]:
            raise ValueError('validation data/cache contract differs')
    head = fingerprint(Path(specs[0]['q2c_checkpoint']))
    if any(cfg['experiment']['head_sha256'] != head['sha256'] for cfg in configs):
        raise ValueError('actual selected matching head differs')
    manifest = Path(configs[0]['validation']['roots_file'])
    rows = {row['sample_id']: row for row in (json.loads(line) for line in manifest.read_text().splitlines() if line)}
    panel = read(args.panel)['sample_ids']
    if len(panel) != len(set(panel)) or len(panel) != 128:
        raise ValueError('original development128 required')
    reference = read(args.reference_evaluation)
    if panel != reference['quick_panel_ids'] or len({row['source_cluster_id'] for row in reference['per_asset']}) != 93:
        raise ValueError('original development128/93source reference required')
    expected = {row['sample_id']: row['source_cluster_id'] for row in reference['per_asset']}
    if any(rows[sid]['source_cluster_id'] != expected[sid] for sid in panel):
        raise ValueError('development source identity changed')

    def local(path):
        if not path.startswith('/file4/'):
            raise ValueError('canonical cache path required')
        return args.cache_root / path.removeprefix('/')

    summaries, inputs, per_asset = {str(t): {} for t in [.005, .05]}, [], []
    for sid in panel:
        row = rows[sid]
        match = Path(configs[0]['validation']['dataset_args']['match_root']) / (sid + '.npz')
        transport = local(row['transport_cache'])
        image = local(row['target_anchor'])
        with np.load(transport) as cache:
            if str(cache['conditioning_input_sha256']) != row['conditioning_input_sha256']:
                raise ValueError('transport RGB identity differs')
            source_valid = cache['source_valid'].astype(bool)
        with np.load(match) as cache:
            if (str(cache['conditioning_input_sha256']) != row['conditioning_input_sha256']
                    or str(cache['q2c_sha256']) != head['sha256'] or int(cache['topk']) != 64):
                raise ValueError('matching cache RGB/head/top64 identity differs')
            ids, logits, valid = cache['semantic_indices'].astype(np.int64), cache['matching_logits'], cache['candidate_valid'].copy()
        if source_valid.shape != (3, 1024) or ids.shape != logits.shape or ids.shape != (1024, 64):
            raise ValueError('original three-view/32x32 semantic64 cache required')
        with Image.open(image) as rgba:
            alpha = np.asarray(rgba.convert('RGBA').resize((512, 512), Image.Resampling.NEAREST))[..., 3]
        valid &= source_valid.flatten()[ids] & (alpha[8::16, 8::16] > 127).flatten()[:, None] & (source_valid.sum() >= 64)
        active = valid.any(1)
        if not active.any():
            raise ValueError('each original case must retain active target queries')
        ids, logits, valid = ids[active], logits[active], valid[active]
        distributions = {str(t): probability(logits, valid, t) for t in [.005, .05]}
        if not np.array_equal(distributions['0.005'].argmax(1), distributions['0.05'].argmax(1)):
            raise ValueError('positive temperature scaling changed hard Top1')
        case = {'sample_id': sid, 'source_cluster_id': row['source_cluster_id'], 'active_queries': int(active.sum()), 'distributions': {}}
        for temperature, weights in distributions.items():
            values = query_statistics(weights, ids // 1024)
            case['distributions'][temperature] = summarize(values)
            for key, value in values.items(): summaries[temperature].setdefault(key, []).append(value)
        per_asset.append(case)
        inputs.append({'sample_id': sid, 'conditioning_input_sha256': row['conditioning_input_sha256'],
            'files': [fingerprint(path) for path in [match, transport, image]]})
    result = {'status': 'PASS_ACTUAL_CACHED_SCORE_PRIOR_SUPPORT_DIAGNOSIS',
        'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'cases': len(panel), 'source_clusters': 93, 'active_queries': sum(row['active_queries'] for row in per_asset),
        'GPU_used': False, 'optimizer_updates': 0, 'reserved121_accessed': False,
        'same_semantic64_candidate_ids_and_support': True, 'hard_Top1_identity_unchanged': True,
        'query_weighted_distributions': {temperature: summarize({key: np.concatenate(values) for key, values in fields.items()})
                                         for temperature, fields in summaries.items()},
        'per_asset': per_asset, 'inputs': inputs, 'original_head': head,
        'bindings': [fingerprint(path) for path in [args.reference_config, args.soft_config, args.panel,
                     args.reference_evaluation, manifest, Path(__file__)]],
        'scope': 'Descriptive cached FP32 matching-score priors on original development128/93sources. '
                 'Temperature0.05 was already declared before this diagnosis. No utility computation, '
                 'fresh decoder/GPU-AMP replay,SS generation,GT correspondence accuracy or model-quality/adoption claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ['status', 'cases', 'active_queries', 'query_weighted_distributions']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['reference-config', 'soft-config', 'panel', 'reference-evaluation', 'cache-root', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    inspect(parser.parse_args())


if __name__ == '__main__':
    main()
