"""Read-only matching evaluation on actual SS validation, not SS generation."""
from pathlib import Path
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from pixal3d.modules.ray_feature_attention import RayFeatureReadout
from . import ray_set_head_training as shared
from .validation_ray_matching_v2 import evaluate_native_partition
from .validation_ray_matching import ValidationObservations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cache-root', 'cache-receipt', 'campaign-root', 'cold-conditioner', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--warm-head', type=Path)
    p.add_argument('--warm-receipt', type=Path)
    p.add_argument('--entry-probe', action='store_true')
    args = p.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if physical not in {'0', '1', '2', '3'} or physical not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one explicitly assigned GPU0-3 required')
    if args.entry_probe == bool(args.warm_head):
        raise ValueError('declare exactly one initial entry probe or actual fixed7 head')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.1, 0)
    torch.use_deterministic_algorithms(True)
    from . import train_prior_anchored_ray_qk, validation_ray_matching, validation_ray_matching_v2
    from pixal3d.modules import ray_feature_attention, ray_prior_anchored_readout, task_matching_attention
    paths = [Path(__file__), Path(shared.__file__), Path(train_prior_anchored_ray_qk.__file__),
        Path(validation_ray_matching.__file__), Path(validation_ray_matching_v2.__file__), Path(ray_feature_attention.__file__),
        Path(ray_prior_anchored_readout.__file__), Path(task_matching_attention.__file__),
        args.cold_conditioner, args.cache_root/'cache-index.jsonl', args.cache_root/'completed.json',
        args.cache_receipt]
    bound = {str(path.resolve()): shared.sha(path) for path in paths}

    def closed(path, receipt):
        record = shared.read(receipt)
        relative = str(path.relative_to(args.campaign_root.parent))
        if (record.get('status') != 'PASS' or record.get('local_equals_actual_file4_SHA256') is not True
                or not any(r['relative_path'] == relative and r['sha256'] == shared.sha(path)
                           for r in record['files'])):
            raise ValueError('actual file4 artifact closure required: '+str(path))

    for path in [args.cache_root/'cache-index.jsonl', args.cache_root/'completed.json']:
        closed(path, args.cache_receipt)
    cache_done = shared.read(args.cache_root/'completed.json')
    if (cache_done['status'] != 'PASS_ACTUAL_NATIVE_SS_VALIDATION32_MATCHING_CACHE'
            or cache_done.get('feature_chunk_size') != 256
            or cache_done['auxiliary_fit_or_calibration_overlap']
            or not cache_done['all_native_probability_replays_exact']):
        raise ValueError('actual native source-disjoint validation cache required')
    rows = list(map(json.loads, (args.cache_root/'cache-index.jsonl').read_text().splitlines()))
    if len(rows) != 32 or len({r['source_cluster_id'] for r in rows}) != 32:
        raise ValueError('fixed32 validation source clusters required')
    for row in rows:
        for kind in ['inputs', 'labels']:
            path = args.cache_root/row[kind+'_relative_path']
            closed(path, args.cache_receipt)
            bound[str(path)] = row[kind+'_SHA256']
    source = torch.load(args.cold_conditioner, map_location='cpu', weights_only=True)
    state = {k[len('ray_readout.'):]: v for k, v in source.items() if k.startswith('ray_readout.')}
    cold = RayFeatureReadout().cuda().eval().requires_grad_(False)
    cold.load_state_dict(state, strict=True)
    if len(state) != 9 or torch.count_nonzero(cold.output.weight):
        raise ValueError('untouched cold9 readout required')
    anchored = copy.deepcopy(cold)
    anchored.query.weight.zero_()
    candidate = copy.deepcopy(anchored)
    if not args.entry_probe:
        if args.warm_receipt is None or args.warm_head.name != 'ray-readout-QK-epoch07.pt':
            raise ValueError('fixed epoch7 warmup and actual file4 receipt required')
        done_path = args.warm_head.parent/'completed.json'
        for path in [args.warm_head, done_path, args.warm_head.parent/'terminal.json']:
            closed(path, args.warm_receipt)
            bound[str(path)] = shared.sha(path)
        done = shared.read(done_path)
        if (done['status'] != 'PASS_FIXED_SEVEN_EPOCH_PRIOR_ANCHORED_QK'
                or done['epochs'] != 7 or done['SS_updates'] != 0
                or shared.read(args.warm_head.parent/'terminal.json')['observed_terminal_exit_code'] != 0
                or done['final_head_SHA256'] != shared.sha(args.warm_head)):
            raise ValueError('actual completed fixed7 QK-only training required')
        candidate.load_state_dict(torch.load(args.warm_head, map_location='cpu', weights_only=True), strict=True)
        changed = {k for k, v in candidate.state_dict().items() if not torch.equal(v, anchored.state_dict()[k])}
        if changed != {'query.weight', 'key.weight'} or torch.count_nonzero(candidate.output.weight):
            raise ValueError('only two Q/K tensors may differ from the anchored native reference')
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    observations = ValidationObservations(args.cache_root)
    candidate_name = 'entry-anchored-initial' if args.entry_probe else 'fixed7-prior-anchored'
    metrics = {name: evaluate_native_partition(head, observations, rows,
        verify_native_reference=name=='cold-random-QK') for name, head in [
        ('cold-random-QK', cold), ('pretrained-utility7', anchored), (candidate_name, candidate)]}
    reference = {r['sample_id']: r for r in metrics['pretrained-utility7']['per_asset']}
    reference_rows = metrics['pretrained-utility7']['per_asset']
    denominator = np.array([r['observable_relations'] for r in reference_rows])
    rng = np.random.default_rng(20261020)
    take = rng.integers(0, 32, size=(10000, 32))
    bootstrap_denominator = denominator[take].sum(1)
    if not (bootstrap_denominator > 0).all():
        raise ValueError('observable validation relations required')
    contrasts = {}
    for name in ['cold-random-QK', candidate_name]:
        ordered = {r['sample_id']: r for r in metrics[name]['per_asset']}
        result = {}
        for kind in ['exact', 'near1']:
            delta = []
            for old in reference_rows:
                row = ordered[old['sample_id']]
                if row['observable_relations'] != old['observable_relations'] or row['source_cluster_id'] != old['source_cluster_id']:
                    raise ValueError('paired native ray relations changed')
                delta.append(row['metrics'][kind]['mass_sum'] - old['metrics'][kind]['mass_sum'])
            delta = np.array(delta)
            result[kind] = dict(probability_mass_difference=float(delta.sum()/denominator.sum()),
                source_cluster_CI95=np.quantile(delta[take].sum(1)/bootstrap_denominator, [.025, .975]).tolist())
        contrasts[name+' minus pretrained-utility7'] = result
    if args.entry_probe and any(metrics[candidate_name][k] != metrics['pretrained-utility7'][k]
                                for k in ['exact', 'near1', 'per_asset']):
        raise ValueError('native entry probe must exactly reproduce utility7 on all32 validation assets')
    if any(shared.sha(path) != digest for path, digest in bound.items()):
        raise ValueError('read-only validation evaluation source/cache/head changed')
    result = dict(status='PASS_ACTUAL_SS_VAL32_QK_MATCHING_EVALUATION', split='validation', assets=32,
        source_clusters=32, feature_chunk_size=256, all_cold_native_reference_probabilities_exact=True,
        auxiliary_training_or_calibration_overlap=False, entry_probe=args.entry_probe,
        metrics=metrics, contrasts=contrasts, bootstrap_draws=10000, bootstrap_seed=20261020,
        source_SHA256=bound, physical_GPU=physical, peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        seconds=time.monotonic()-began, optimizer_updates=0, labels_in_model_forward=False,
        SS_generation_or_quality_gain_claimed=False, independent_publication_test=False,
        model_adopted=False, full_goal_complete=False)
    shared.write_new(args.output/'evaluation.json', result)
    print(json.dumps(dict(status=result['status'], entry_probe=args.entry_probe, contrasts=contrasts,
        peak_GiB=result['peak_allocated_GiB'], seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
