"""Measure the learned geometry route at an actual saved SS checkpoint.

Only the conditioner executes. NN generation/training are not performed.
Native VAL32 observations must equal the closed cache before GT labels are
used for matching metrics. Projection-off and wrong RAW address results
are saved-weight sensitivity diagnostics, not retrained ablations.
"""
from pathlib import Path
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from pixal3d.datasets.deformnext_deformation_query import DeformNeXtDeformationQuerySS64
from pixal3d.modules.ray_geometry_retargeting import RayGeometryRetargetingConditioner
from pixal3d.modules.ray_probability_temperature import matching_logits, native_probability
from . import ray_set_head_training as shared
from .calibrate_ray_matching_temperature_v2 import asset_metrics, aggregate, compare
from .validation_ray_matching import ValidationObservations


def rms(value):
    return float(value.float().square().mean().sqrt()) if value.numel() else 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if physical not in {'0', '1', '2', '3'} or physical not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one assigned GPU0-3 required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.1, 0)
    torch.use_deterministic_algorithms(True)
    plan = shared.read(args.plan)
    archive = Path(plan['campaign_root']).parent
    bound = dict(plan['source_SHA256'])
    bound[str(args.plan)] = shared.sha(args.plan)

    def closed(path, receipt):
        path = Path(path)
        r = shared.read(Path(receipt))
        assert r['status'] == 'PASS' and r['local_equals_actual_file4_SHA256']
        assert any(x['relative_path'] == str(path.relative_to(archive))
                   and x['sha256'] == shared.sha(path) for x in r['files'])

    assert all(shared.sha(p) == d for p, d in bound.items())
    for path, receipt in plan['closed_inputs'].items():
        closed(path, receipt)
    cfg = shared.read(Path(plan['config']))
    val = cfg['validation']
    dataset = DeformNeXtDeformationQuerySS64(val['roots_file'], **val['dataset_args'])
    by_id = {row['sample_id']: i for i, row in enumerate(dataset.rows)}
    model = RayGeometryRetargetingConditioner(**cfg['trainer']['args']['deform_conditioner']).cuda().eval()
    model.load_state_dict(torch.load(plan['weights'], map_location='cpu', weights_only=True), strict=True)
    model.requires_grad_(False)
    assert torch.count_nonzero(model.retargeter.deformation_projection.weight) > 0
    assert not hasattr(model.ray_readout, 'value') and not hasattr(model.ray_readout, 'output')
    current_qzero = copy.deepcopy(model.ray_readout)
    current_qzero.query.weight.zero_()
    utility = copy.deepcopy(current_qzero)
    cold = torch.load(plan['cold_conditioner'], map_location='cpu', weights_only=True)
    utility.score_selector.load_state_dict({k[len('ray_readout.score_selector.'):]: v
        for k, v in cold.items() if k.startswith('ray_readout.score_selector.')}, strict=True)
    root = Path(plan['val_cache'])
    index = list(map(json.loads, (root/'cache-index.jsonl').read_text().splitlines()))
    assert len(index) == 32 and len({r['source_cluster_id'] for r in index}) == 32
    observations = ValidationObservations(root)
    metrics = {n: [] for n in ['current', 'utility', 'current-Qzero']}
    activations = []
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for row in index:
            pack = dataset[by_id[row['sample_id']]]
            transport = {key[len('transport_'):]: value[None].cuda() for key, value in pack.items()
                if key.startswith('transport_') and key not in ['transport_patch_grid', 'transport_patch_ids']}
            assert not {'deformation', 'spatial_indices', 'positive_indices', 'positive_visible'} & transport.keys()
            native, appearance_hidden, deformation_hidden, pre_tanh, retarget = [], [], [], [], []
            handles = [model.ray_readout.register_forward_hook(lambda m, ins, out:
                native.append((tuple(x.detach() for x in ins), out[1].detach()))),
                model.retargeter.net[0].register_forward_hook(lambda m, ins, out: appearance_hidden.append(out.detach())),
                model.retargeter.deformation_projection.register_forward_hook(lambda m, ins, out: deformation_hidden.append(out.detach())),
                model.retargeter.net[2].register_forward_hook(lambda m, ins, out: pre_tanh.append(out.detach())),
                model.retargeter.register_forward_hook(lambda m, ins, out: retarget.append((ins[0].detach(), out.detach())))]
            try:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = model(**transport)
            finally:
                for handle in handles:
                    handle.remove()
            assert len(retarget) == len(appearance_hidden) == len(deformation_hidden) == len(pre_tanh) == 1
            active = result.query_valid
            patches = active[0].nonzero().flatten()
            # Labels are opened only after the complete actual conditioner forward.
            x, y = observations.load(row)
            with np.load(root/row['inputs_relative_path'], allow_pickle=False) as saved:
                assert np.array_equal(patches.cpu().numpy(), saved['query_patch_indices'])
            ids = x['candidate_indices'].astype(np.int64)
            expected_inputs = [x['native_query'], x['context_query'], x['source_key_bank'][ids],
                x['source_value_bank'][ids], x['candidate_valid'], x['scalar_prior_logits']]
            dimensions = [(0, 1024), (0, 1024), (0, 64, 1024), (0, 64, 1024), (0, 64), (0, 64)]
            for i, expected in enumerate(expected_inputs):
                actual = torch.cat([part[0][i].cpu() for part in native]).numpy() if native else np.empty(dimensions[i], dtype=expected.dtype)
                assert np.array_equal(actual, expected), ('native input differs', row['sample_id'], i)
            probabilities = {n: [] for n in metrics}
            for inputs, actual in native:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    replay = native_probability(matching_logits(model.ray_readout, *inputs), inputs[4])
                    assert torch.equal(actual, replay)
                    probabilities['current'].append(actual.cpu())
                    for name, head in [('utility', utility), ('current-Qzero', current_qzero)]:
                        probabilities[name].append(native_probability(matching_logits(head, *inputs), inputs[4]).cpu())
            for name in metrics:
                p = torch.cat(probabilities[name]).mean(1) if probabilities[name] else torch.empty(0, 64)
                metrics[name].append(asset_metrics(row, y, p))
            inputs, full = retarget[0]
            ha, hd = appearance_hidden[0], deformation_hidden[0]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                base = model.g_max*torch.tanh(model.retargeter.gate_logit)*torch.tanh(
                    model.retargeter.net[2](model.retargeter.net[1](ha)))
            actual_delta = full-base
            wrong_difference = torch.zeros_like(full)
            if len(patches):
                valid_addresses = transport['source_valid'].flatten().nonzero().flatten()
                permutation = torch.arange(3072, device='cuda:0')
                permutation[valid_addresses] = valid_addresses.roll(max(1, len(valid_addresses)//2))
                wrong_ids = permutation[result.candidate_indices[0, patches]]
                probability = torch.cat([p for _, p in native]).mean(1)
                raw = transport['source_context'].flatten(1, 2)[0].float()
                wrong_dq = transport['target_context'][0, patches].float()-(
                    probability[..., None]*raw[wrong_ids]).sum(1)
                wrong_input = inputs.clone()
                wrong_input[0, patches, -2048:] = wrong_dq
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    wrong_difference = model.retargeter(wrong_input, model.g_max)-full
            item = dict(sample_id=row['sample_id'], source_cluster_id=row['source_cluster_id'],
                active_rays=int(active.sum()), appearance_input_RMS=rms(inputs[..., :2048][active]),
                raw_DQ_RMS=rms(inputs[..., -2048:][active]),
                raw_DQ_FRAME_RMS=rms(inputs[..., -2048:-1024][active]),
                raw_DQ_GLOBAL_RMS=rms(inputs[..., -1024:][active]),
                appearance_hidden_RMS=rms(ha[active]), DQ_projection_hidden_RMS=rms(hd[active]),
                effective_hidden_addition_RMS=rms(((ha+hd).float()-ha.float())[active]),
                hidden_addition_absorbed_fraction=float(((ha+hd)==ha)[active].float().mean()) if active.any() else 0.,
                R_full_RMS=rms(full[active]), R_without_DQ_RMS=rms(base[active]),
                R_actual_DQ_contribution_RMS=rms(actual_delta[active]),
                R_wrong_RAW_address_change_RMS=rms(wrong_difference[active]),
                R_pre_tanh_saturated_fraction=float((pre_tanh[0].abs()>2)[active].float().mean()) if active.any() else 0.)
            activations.append(item)
            del pack, transport, native, retarget, appearance_hidden, deformation_hidden, pre_tanh, result
    original = shared.read(Path(plan['native_val_evaluation']))['metrics']['pretrained-utility7']
    reference = aggregate(metrics['utility'])
    for kind in ['exact', 'near1']:
        assert reference[kind]['relations'] == original[kind]
    total = sum(r['active_rays'] for r in activations)
    pooled = {name: float(np.sqrt(sum(r['active_rays']*r[name]**2 for r in activations)/total))
        for name in activations[0] if name.endswith('_RMS')}
    ratio = lambda numerator, denominator: numerator/denominator if denominator > 0 else None
    pooled.update(DQ_hidden_to_appearance_ratio=ratio(pooled['DQ_projection_hidden_RMS'],pooled['appearance_hidden_RMS']),
        DQ_contribution_to_full_R_ratio=ratio(pooled['R_actual_DQ_contribution_RMS'],pooled['R_full_RMS']),
        wrong_address_change_to_DQ_contribution_ratio=ratio(pooled['R_wrong_RAW_address_change_RMS'],pooled['R_actual_DQ_contribution_RMS']),
        native_projected_DQ_contribution_RMS=.75*pooled['R_actual_DQ_contribution_RMS'])
    result = dict(status='PASS_ACTUAL_NATIVE_GEOMETRY_RETARGETING_USAGE',step=plan['step'], assets=32,
        active_rays=total, pooled_activations=pooled, per_asset_activations=activations,
        matching_metrics={n: aggregate(r) for n, r in metrics.items()},
        matching_contrasts={n+' minus utility': compare(metrics['utility'], metrics[n]) for n in ['current', 'current-Qzero']},
        all_six_native_observations_exact_to_closed_VAL_cache=True, actual_current_head_replay_exact=True,
        original_utility32_relation_metrics_exact=True, appearance_VO_physically_absent=True,
        DQ_observations_not_zeroed=True, projection_off_is_saved_weight_sensitivity_not_trained_NoDQ=True,
        wrong_address_only_changes_RAW_value_input_to_R=True, GT_in_model_forward=False,
        NN_generation=False, optimizer_updates=0, model_adopted=False,full_goal_complete=False,
        WP_Frobenius=float(model.retargeter.deformation_projection.weight.float().norm()),
        retargeter_gate=float(torch.tanh(model.retargeter.gate_logit)),g_max=model.g_max,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        seconds=time.monotonic()-began,source_SHA256=bound)
    assert all(shared.sha(p) == d for p, d in bound.items())
    shared.write_new(output/'audit.json',result)
    print(json.dumps(dict(status=result['status'], pooled_activations=pooled,
        seconds=result['seconds'], peak_GiB=result['peak_allocated_GiB'])),flush=True)


if __name__ == '__main__':
    main()
