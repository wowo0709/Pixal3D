"""Diagnose matching drift in actual SS192/288 heads on native VAL32 rays.

Saved-weight diagnostic only: matching probabilities do not establish a
reconstruction gain. Q-zero changes projection weights, not observations.
The candidate/proposal/score bank and actual native256 query partition stay
fixed. No denoiser, rendering or optimizer is constructed.
"""
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
from .validation_ray_matching import ValidationObservations
from .validation_ray_matching_v2 import evaluate_native_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache-root', 'cache-receipt', 'campaign-root', 'cold-conditioner', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--conditioner-checkpoint', type=Path, action='append', required=True)
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if physical not in {'0', '1', '2', '3'} or physical not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one explicitly assigned physical GPU0-3 required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.1, 0)
    torch.use_deterministic_algorithms(True)
    from . import validation_ray_matching, validation_ray_matching_v2
    from pixal3d.modules import ray_feature_attention, ray_prior_anchored_readout, task_matching_attention
    sources = [Path(__file__), Path(shared.__file__), Path(validation_ray_matching.__file__),
        Path(validation_ray_matching_v2.__file__), Path(ray_feature_attention.__file__),
        Path(ray_prior_anchored_readout.__file__), Path(task_matching_attention.__file__),
        args.cold_conditioner, args.cache_root/'cache-index.jsonl', args.cache_root/'completed.json',
        args.cache_receipt, *args.conditioner_checkpoint]
    bound = {str(path.resolve()): shared.sha(path) for path in sources}
    receipt = shared.read(args.cache_receipt)
    if receipt.get('status') != 'PASS' or receipt.get('local_equals_actual_file4_SHA256') is not True:
        raise ValueError('actual native validation cache closure required')
    closed = {row['relative_path']: row['sha256'] for row in receipt['files']}
    rows = list(map(json.loads, (args.cache_root/'cache-index.jsonl').read_text().splitlines()))
    completed = shared.read(args.cache_root/'completed.json')
    if (len(rows) != 32 or len({r['source_cluster_id'] for r in rows}) != 32
            or completed['status'] != 'PASS_ACTUAL_NATIVE_SS_VALIDATION32_MATCHING_CACHE'
            or completed['feature_chunk_size'] != 256 or completed['split'] != 'validation'):
        raise ValueError('same native256 actual validation32 panel required')
    for path in [args.cache_root/'cache-index.jsonl', args.cache_root/'completed.json',
                 *[args.cache_root/row[kind+'_relative_path'] for row in rows for kind in ['inputs', 'labels']]]:
        if closed[str(path.relative_to(args.campaign_root.parent))] != shared.sha(path):
            raise ValueError('closed validation cache bytes differ')
        bound[str(path)] = shared.sha(path)
    if sorted(path.name for path in args.conditioner_checkpoint) != [
            'deform_conditioner_step0000192.pt', 'deform_conditioner_step0000288.pt']:
        raise ValueError('this declared audit needs exactly Direct192 and Direct288')
    if len({path.parent for path in args.conditioner_checkpoint}) != 1:
        raise ValueError('same Direct scientific run required')

    def head(path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        state = {key[len('ray_readout.'):]: value for key, value in state.items() if key.startswith('ray_readout.')}
        if len(state) != 9 or any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError('complete finite native9 readout checkpoint required')
        module = RayFeatureReadout().cuda().eval().requires_grad_(False)
        module.load_state_dict(state, strict=True)
        return module

    args.output.mkdir(parents=True, exist_ok=False)
    cold = head(args.cold_conditioner)
    utility = copy.deepcopy(cold)
    utility.query.weight.zero_()
    models = [('cold-random-QK', cold), ('pretrained-utility7', utility)]
    for path in sorted(args.conditioner_checkpoint):
        step = int(path.stem.split('step')[1])
        learned = head(path)
        anchored = copy.deepcopy(learned)
        anchored.query.weight.zero_()
        models.extend([(f'Direct{step}', learned), (f'Direct{step}-Qzero', anchored)])
    observations = ValidationObservations(args.cache_root)
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    metrics = {}
    for name, model in models:
        metrics[name] = evaluate_native_partition(model, observations, rows,
            verify_native_reference=name == 'cold-random-QK')
        (args.output/(name+'.json')).write_text(json.dumps(metrics[name], indent=2)+'\n')
        print(json.dumps(dict(model=name, near1=metrics[name]['near1'],
            seconds=time.monotonic()-began)), flush=True)
    reference = metrics['pretrained-utility7']['per_asset']
    count = np.array([row['observable_relations'] for row in reference])
    random = np.random.default_rng(20261020)
    indices = random.integers(0, 32, size=(10000, 32))
    denominator = count[indices].sum(1)
    if not (denominator > 0).all():
        raise ValueError('observable source-cluster relations required')
    contrasts = {}
    for candidate, baseline in [('Direct192', 'pretrained-utility7'), ('Direct288', 'pretrained-utility7'),
            ('Direct288', 'Direct192'), ('Direct192-Qzero', 'Direct192'), ('Direct288-Qzero', 'Direct288')]:
        candidate_rows = {row['sample_id']: row for row in metrics[candidate]['per_asset']}
        baseline_rows = {row['sample_id']: row for row in metrics[baseline]['per_asset']}
        comparisons = {}
        for kind in ['exact', 'near1']:
            values = []
            for item in reference:
                c, b = candidate_rows[item['sample_id']], baseline_rows[item['sample_id']]
                if (c['source_cluster_id'] != b['source_cluster_id']
                        or c['observable_relations'] != b['observable_relations']):
                    raise ValueError('paired matching relations differ')
                values.append(c['metrics'][kind]['mass_sum']-b['metrics'][kind]['mass_sum'])
            values = np.array(values)
            comparisons[kind] = dict(probability_mass_difference=float(values.sum()/count.sum()),
                source_cluster_CI95=np.quantile(values[indices].sum(1)/denominator, [.025, .975]).tolist())
        contrasts[candidate+' minus '+baseline] = comparisons
    if any(shared.sha(path) != digest for path, digest in bound.items()):
        raise ValueError('read-only audit source, immutable checkpoint or cache changed')
    result = dict(status='PASS_ACTUAL_SS_MATCHING_DRIFT_NATIVE_VAL32', assets=32, source_clusters=32,
        observable_relations=int(count.sum()), split='actual SS validation', feature_chunk_size=256,
        metrics={name: {key: data[key] for key in ['exact', 'near1']} for name, data in metrics.items()},
        contrasts=contrasts, source_SHA256=bound, bootstrap_draws=10000, bootstrap_seed=20261020,
        optimizer_updates=0, labels_in_forward=False, whole_decoder_ablated=False,
        Qzero_is_weight_intervention_not_observation_zeroing=True, proposal_bank_fixed=True,
        SS_reconstruction_gain_or_causal_module_effect_claimed=False, diagnostic_only=True,
        physical_GPU=physical, peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        seconds=time.monotonic()-began, model_adopted=False, full_goal_complete=False)
    shared.write_new(args.output/'summary.json', result)
    print(json.dumps(dict(status=result['status'], contrasts=contrasts)), flush=True)


if __name__ == '__main__':
    main()
