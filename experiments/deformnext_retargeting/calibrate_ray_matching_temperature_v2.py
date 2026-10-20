"""Select confidence calibration on auxiliary TRAIN only, then evaluate VAL.

The fixed epoch7 head is never fitted again. Native per-head probabilities
are replayed exactly. Reconstruction weights and running experiments are
untouched. This diagnostic cannot establish an SS quality improvement.
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
from pixal3d.modules.ray_prior_anchored_readout import correspondence_probabilities
from pixal3d.modules.ray_probability_temperature import (
    matching_logits, native_probability, pooled_temperature)
from . import ray_set_head_training as shared
from .train_prior_anchored_ray_qk import Observations
from .validation_ray_matching import ValidationObservations


def asset_metrics(row, y, probability):
    relation = torch.from_numpy(y['relation'])
    queries = np.unique(y['relation'])
    records = {}
    for kind in ['exact', 'near1']:
        label = torch.from_numpy(y[kind])
        rp = probability[relation]
        relation_stats = dict(mass_sum=float((rp*label).sum()),
            hit1=int(label.gather(1, rp.argmax(-1)[:, None]).sum()),
            proposal64_hits=int(label.any(-1).sum()))
        union = np.zeros((len(probability), 64), bool)
        np.logical_or.at(union, y['relation'], y[kind])
        label_union = torch.from_numpy(union[queries])
        rays = probability[queries]
        mass = (rays*label_union).sum(-1)
        covered = label_union.any(-1)
        ray_stats = dict(mass_sum=float(mass.sum()),
            hit1=int(label_union.gather(1, rays.argmax(-1)[:, None]).sum()),
            proposal64_hits=int(covered.sum()),
            covered_NLL_sum=float(-mass[covered].clamp_min(1e-30).log().sum()))
        records[kind] = dict(relations=relation_stats, rays=ray_stats)
    entropy = -(probability*probability.clamp_min(1e-30).log()).sum(-1)
    return dict(sample_id=row['sample_id'], source_cluster_id=row['source_cluster_id'],
        observable_relations=len(relation), observable_rays=len(queries), metrics=records,
        native_queries=len(probability), entropy_sum=float(entropy.sum()),
        effective_K_sum=float(entropy.exp().sum()), maximum_probability_sum=float(probability.amax(-1).sum()))


def aggregate(rows):
    out = dict(assets=len(rows), assets_without_native_queries=sum(r['native_queries']==0 for r in rows), observable_relations=sum(r['observable_relations'] for r in rows),
        observable_rays=sum(r['observable_rays'] for r in rows))
    queries = sum(r['native_queries'] for r in rows)
    out['readout'] = dict(native_queries=queries,
        mean_entropy=sum(r['entropy_sum'] for r in rows)/queries,
        mean_effective_K=sum(r['effective_K_sum'] for r in rows)/queries,
        mean_maximum_probability=sum(r['maximum_probability_sum'] for r in rows)/queries)
    for kind in ['exact', 'near1']:
        out[kind] = {}
        for axis in ['relations', 'rays']:
            denominator = out['observable_'+axis]
            totals = {key: sum(r['metrics'][kind][axis][key] for r in rows)
                      for key in ['mass_sum', 'hit1', 'proposal64_hits']}
            out[kind][axis] = dict(mean_probability_mass=totals['mass_sum']/denominator,
                top1_coverage=totals['hit1']/denominator,
                proposal64_coverage=totals['proposal64_hits']/denominator)
            if axis == 'rays':
                out[kind][axis]['covered_mean_NLL'] = sum(
                    r['metrics'][kind][axis]['covered_NLL_sum'] for r in rows)/totals['proposal64_hits']
    return out


def compare(left, right):
    by_id = {r['sample_id']: r for r in left}
    assert len(left) == len(right) == 32
    assert len({r['source_cluster_id'] for r in right}) == 32
    ix = np.random.default_rng(20261020).integers(0, 32, (10000, 32))
    result = {}
    for axis in ['relations', 'rays']:
        count = np.array([r['observable_'+axis] for r in right])
        for r in right:
            old = by_id[r['sample_id']]
            assert r['source_cluster_id'] == old['source_cluster_id']
            assert r['observable_'+axis] == old['observable_'+axis]
        for kind in ['exact', 'near1']:
            for metric in ['mass_sum', 'hit1']:
                delta = np.array([r['metrics'][kind][axis][metric]
                    - by_id[r['sample_id']]['metrics'][kind][axis][metric] for r in right])
                distribution = delta[ix].sum(1)/count[ix].sum(1)
                result[axis+'_'+kind+'_'+metric] = dict(difference_pp=float(100*delta.sum()/count.sum()),
                    paired_source_cluster_CI95_pp=(100*np.quantile(distribution, [.025, .975])).tolist())
    return result


def join_probabilities(parts):
    return torch.cat(parts) if parts else torch.empty(0, 64, dtype=torch.float32)


def evaluate(head, observations, rows, temperatures, chunk, phase):
    records = {str(t): [] for t in temperatures}
    ranking_changes = {str(t): 0 for t in temperatures}
    with torch.no_grad():
        for row in rows:
            x, y = observations.load(row)
            parts = {str(t): [] for t in temperatures}
            for begin in range(0, len(x['native_query']), chunk):
                take = slice(begin, begin+chunk)
                ids = x['candidate_indices'][take].astype(np.int64)
                args = [torch.from_numpy(v).cuda() for v in [x['native_query'][take].astype(np.float32),
                    x['context_query'][take].astype(np.float32), x['source_key_bank'][ids].astype(np.float32),
                    x['source_value_bank'][ids].astype(np.float32), x['candidate_valid'][take],
                    x['scalar_prior_logits'][take].astype(np.float32)]]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    reference = correspondence_probabilities(head, *args)
                    logits = matching_logits(head, *args)
                    actual = native_probability(logits, args[4])
                if not torch.equal(reference, actual):
                    raise ValueError('native per-head probability replay differs')
                # Preserve each existing diagnostic's original reduction device
                # at T1: TRAIN calibration averages on GPU; VAL32 on CPU.
                baseline = reference.mean(1).cpu() if phase == 'calibration' else reference.cpu().mean(1)
                for temperature in temperatures:
                    calibrated = baseline if temperature == 1. else pooled_temperature(
                        logits, args[4], temperature).cpu()
                    ranking_changes[str(temperature)] += int((calibrated.argmax(-1) != baseline.argmax(-1)).sum())
                    parts[str(temperature)].append(calibrated)
            for temperature in temperatures:
                probability = join_probabilities(parts[str(temperature)])
                records[str(temperature)].append(asset_metrics(row, y, probability))
    return {t: dict(summary=aggregate(r), per_asset=r,
        ranking_changes=ranking_changes[t]) for t, r in records.items()}


def verify_baseline(actual, expected):
    for kind in ['exact', 'near1']:
        assert actual['summary'][kind]['relations'] == expected[kind]
    by_id = {r['sample_id']: r for r in expected['per_asset']}
    for row in actual['per_asset']:
        original = by_id[row['sample_id']]
        assert row['source_cluster_id'] == original['source_cluster_id']
        assert row['observable_relations'] == original['observable_relations']
        for kind in ['exact', 'near1']:
            assert row['metrics'][kind]['relations'] == original['metrics'][kind]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', choices=['calibration', 'validation'], required=True)
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if physical not in {'0', '1', '2', '3'} or physical not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one explicitly assigned GPU0-3 required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.1, 0)
    torch.use_deterministic_algorithms(True)
    plan = shared.read(args.plan)
    root = Path(plan['output_root'])
    archive = Path(plan['campaign_root']).parent
    assert plan['temperature_grid'] == [.125, .25, .5, .75, 1., 1.5, 2., 4.]
    bound = dict(plan['source_SHA256'])
    bound[str(args.plan)] = shared.sha(args.plan)

    def closed(path, receipt):
        path, receipt = Path(path), Path(receipt)
        record = shared.read(receipt)
        assert record['status'] == 'PASS' and record['local_equals_actual_file4_SHA256']
        assert any(r['relative_path'] == str(path.relative_to(archive))
                   and r['sha256'] == shared.sha(path) for r in record['files'])

    def guard():
        assert all(shared.sha(p) == digest for p, digest in bound.items())

    guard()
    for path, receipt in plan['closed_inputs'].items():
        closed(path, receipt)
    raw = torch.load(plan['cold_conditioner'], map_location='cpu', weights_only=True)
    initial = {k[len('ray_readout.'):]: v for k, v in raw.items() if k.startswith('ray_readout.')}
    utility = RayFeatureReadout().cuda().eval().requires_grad_(False)
    utility.load_state_dict(initial, strict=True)
    utility.query.weight.zero_()
    candidate = copy.deepcopy(utility)
    warm = Path(plan['warm_head'])
    done = shared.read(warm.parent/'completed.json')
    assert done['epochs'] == 7 and done['SS_updates'] == 0 and done['final_head_SHA256'] == shared.sha(warm)
    candidate.load_state_dict(torch.load(warm, map_location='cpu', weights_only=True), strict=True)
    assert {k for k, v in candidate.state_dict().items()
            if not torch.equal(v, utility.state_dict()[k])} == {'query.weight', 'key.weight'}
    aux_root, val_root = Path(plan['aux_cache']), Path(plan['val_cache'])
    auxiliary = list(map(json.loads, (aux_root/'cache-index.jsonl').read_text().splitlines()))
    calibration = [r for r in auxiliary if r['role'] == 'calibration']
    validation = list(map(json.loads, (val_root/'cache-index.jsonl').read_text().splitlines()))
    assert len(calibration) == len(validation) == 32
    assert not ({r['source_cluster_id'] for r in auxiliary} & {r['source_cluster_id'] for r in validation})
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    out = root/(args.phase+'-v1')
    out.mkdir(parents=True, exist_ok=False)
    metrics = {}
    selected = None
    if args.phase == 'calibration':
        cfg = shared.read(Path(plan['aux_config']))
        receipt = shared.read(Path(plan['aux_receipt']))
        closed_aux = {str(archive/r['relative_path']): r['sha256'] for r in receipt['files']}
        observations = Observations(cfg, aux_root, calibration, closed_aux)
        history = list(map(json.loads, (warm.parent/'history.jsonl').read_text().splitlines()))
        assert [r['epoch'] for r in history] == list(range(8))
        for name, head, baseline in [('utility', utility, history[0]['calibration']),
                                     ('fixed7', candidate, history[-1]['calibration'])]:
            result = evaluate(head, observations, calibration, plan['temperature_grid'], 64, args.phase)
            verify_baseline(result['1.0'], baseline)
            metrics[name] = result
        selected = {}
        for name in metrics:
            table = metrics[name]
            penalty = lambda t: (abs(np.log(float(t))), float(t))
            selected[name] = dict(
                near1_union_NLL=float(min(table, key=lambda t: (
                    table[t]['summary']['near1']['rays']['covered_mean_NLL'], *penalty(t)))),
                near1_union_mass=float(min(table, key=lambda t: (
                    -table[t]['summary']['near1']['rays']['mean_probability_mass'], *penalty(t)))))
        selection = dict(status='PASS_TRAIN_CALIBRATION_ONLY_TEMPERATURE_SELECTION',
            temperatures=selected, grid=plan['temperature_grid'], SS_split='train',
            held_out_from_auxiliary_head_fit=True, assets=32, validation_metrics_read=False,
            optimization='finite scalar grid only;no optimizer and no neural weight updates',
            source_plan_SHA256=shared.sha(args.plan), trained_head_epoch=7,
            query_partition=64, NoScore_ablation=False)
        shared.write_new(out/'selection.json', selection)
        assert all(shared.sha(p) == d for p, d in observations.bound.items())
    else:
        selection_path = root/'calibration-v1/selection.json'
        closed(selection_path, root/'calibration-v1/file4-closure.json')
        selection = shared.read(selection_path)
        assert selection['validation_metrics_read'] is False and selection['source_plan_SHA256'] == shared.sha(args.plan)
        selected = selection['temperatures']
        bound[str(selection_path)] = shared.sha(selection_path)
        observations = ValidationObservations(val_root)
        for row in validation:
            for kind in ['inputs', 'labels']:
                path = val_root/row[kind+'_relative_path']
                closed(path, Path(plan['val_receipt']))
                bound[str(path)] = row[kind+'_SHA256']
        original = shared.read(Path(plan['native_val_evaluation']))['metrics']
        for name, head, baseline_name in [('utility', utility, 'pretrained-utility7'),
                                           ('fixed7', candidate, 'fixed7-prior-anchored')]:
            temperatures = sorted({1., *selected[name].values()})
            result = evaluate(head, observations, validation, temperatures, 256, args.phase)
            verify_baseline(result['1.0'], original[baseline_name])
            metrics[name] = result
    contrasts = {}
    if args.phase == 'validation':
        for objective in ['near1_union_NLL', 'near1_union_mass']:
            chosen = metrics['fixed7'][str(selected['fixed7'][objective])]['per_asset']
            for baseline, rows in [('fixed7_T1', metrics['fixed7']['1.0']['per_asset']),
                    ('utility_T1', metrics['utility']['1.0']['per_asset']),
                    ('calibrated_utility', metrics['utility'][str(selected['utility'][objective])]['per_asset'])]:
                contrasts[objective+' minus '+baseline] = compare(rows, chosen)
    guard()
    result = dict(zero_query_assets_retained=True, version='v2-empty-queries-preserved', status='PASS_POOLED_CORRESPONDENCE_TEMPERATURE_'+args.phase.upper(),
        phase=args.phase, assets=32, selected_temperatures=selected, metrics=metrics, contrasts=contrasts,
        all_native_per_head_probabilities_exact=True, original_T1_all32_relation_metrics_exact=True,
        grid_selected_on_auxiliary_TRAIN_only=True, auxiliary_query_partition=64,
        actual_VAL_query_partition=256, source_clusters_disjoint=True,
        calibration_targets='near1 ray-union mass and covered-ray NLL;visible-view relations reported separately',
        probability_temperature_after_head_pooling=True, per_head_logits_unchanged=True,
        labels_in_neural_forward=False, optimizer_updates=0, NN_R_updates=0,
        weights_or_temperatures_applied_to_live_SS=False, GT_metric_computation_CPU_only=True,
        physical_GPU=physical, allocator_fraction=.1,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30, seconds=time.monotonic()-began,
        source_SHA256=bound, independent_publication_test=False, model_adopted=False, full_goal_complete=False)
    shared.write_new(out/'evaluation.json', result)
    print(json.dumps(dict(status=result['status'], selected_temperatures=selected,
        seconds=result['seconds'], peak_GiB=result['peak_allocated_GiB'], contrasts=contrasts)), flush=True)


if __name__ == '__main__':
    main()
