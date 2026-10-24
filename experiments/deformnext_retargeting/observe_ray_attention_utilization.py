"""Observe trained native conditioning on the existing DEV32; no sampling/training."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path

from .cascade_view_roles import ClosedInputs, file_sha256
from .dev128_pipeline_contract import validate_archive128
from .ray_attention_utilization import utilization, summarize_counts, mass_intervals, validate_native_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (plan['schema'] != 'actual-native-ray-utilization-DEV32-v1'
            or gpu not in plan['assigned_GPUs']
            or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')
            or list(plan['models']) != ['E_FIXED_RAW', 'E_LEARNED_RAW']):
        raise ValueError('assigned single GPU and existing fixed/learned DEV32 only')
    os.sched_setaffinity(0, set(plan['CPU_affinity_per_GPU'][gpu]))
    check = ClosedInputs(plan['canonical_local_root'])
    check.check(args.plan, args.plan.parent / 'file4-closure.json')

    def read(spec):
        check.check(spec['path'], spec['receipt'])
        if file_sha256(spec['path']) != spec['SHA256']:
            raise ValueError('closed input differs from its explicit identity')
        return json.loads(Path(spec['path']).read_text())

    def guard():
        for path, digest in {**plan['source_SHA256'], **plan['input_SHA256']}.items():
            if file_sha256(path) != digest:
                raise ValueError('bound diagnostic source/input changed: ' + path)

    guard()
    index = read(plan['index'])
    ids = index['sample_ids'][:32]
    if (plan['sample_ids'] != ids or len(set(ids)) != 32
            or len({index['source_clusters_by_sample'][sid] for sid in ids}) != 32):
        raise ValueError('original cascade/matching DEV32 source-disjoint panel required')
    reference = read(plan['SS_reference'])
    matching_reference = read(plan['matching_reference'])
    if [r['sample_id'] for r in matching_reference['records']] != ids:
        raise ValueError('existing correspondence diagnostic is a different panel')
    out = Path(plan['output']); out.mkdir(parents=True, exist_ok=False)

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)

    import numpy as np
    import torch
    from pixal3d import datasets, models
    from pixal3d.utils.data_utils import recursive_to_device
    from train import setup_rng
    from .consistent_evidence_native_runtime import selected_trainer, parameter_groups, state_sha, tensor_sha
    from .evaluate_consistent_evidence_checkpoint import validate_bundle
    from .topology_labels import fingerprint
    from pixal3d.modules.bounded_member_feature_source_ray import masked_probabilities
    torch.cuda.set_device(0); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    observations = {}; model_proofs = {}; all_forwards_done = False
    for arm, spec in plan['models'].items():
        archive = read(spec['SS_evaluation'])
        validate_archive128(archive, arm, reference, binding=spec['binding'])
        bundle = read(spec['bundle'])
        for field in ['native_config', 'checkpoint', 'conditioner_checkpoint']:
            check.check(bundle[field], spec['bundle']['receipt'])
        cfg = json.loads(Path(bundle['native_config']).read_text())
        if validate_bundle(bundle, cfg) != arm or bundle['step'] != 480:
            raise ValueError('actual own trained480 source/LoRA pair required')
        setup_rng(0, cfg['training_seed'])
        validation = cfg['validation']
        dataset = getattr(datasets, validation['dataset_name'])(validation['roots_file'], **validation['dataset_args'])
        if len(dataset) != 508 or any(row['split'] != 'validation' for row in dataset.rows):
            raise ValueError('validation508 only; no TEST observations')
        byid = {row['sample_id']: i for i, row in enumerate(dataset.rows)}
        with torch.device('meta'):
            md = {name: getattr(models, item['name'])(**item['args']) for name, item in cfg['models'].items()}
        kw = copy.deepcopy(cfg['trainer']['args'])
        kw.update(continuation_ckpt={'denoiser': bundle['checkpoint']}, batch_size_per_gpu=1,
                  batch_split=1, num_workers=0, prefetch_data=False, max_steps=1)
        trainer = selected_trainer(arm)(md, dataset, **kw, output_dir=str(out / arm / 'loader'),
                                       load_dir=None, step=None, wandb_run=None)
        trainer.deform_conditioner.load_state_dict(torch.load(bundle['conditioner_checkpoint'],
                                                   map_location='cpu', weights_only=True), strict=True)
        lora, source = parameter_groups(trainer, arm)
        if (any(p.device.type != 'cuda' for p in trainer.deform_conditioner.parameters())
                or any(p.device.type != 'cuda' for p in trainer.image_cond_model.parameters())):
            raise ValueError('all matching/conditioning NN forwards must use the assigned GPU')
        before = state_sha(list(lora.items()) + list(source.items()))
        for model in trainer.models.values(): model.eval()
        trainer.p_uncond = 0.
        observations[arm] = []
        batch_captures = []

        def observe(module, positional, inputs, output):
            if positional or any(t.device.type != 'cuda' for t in inputs.values() if isinstance(t, torch.Tensor)):
                raise ValueError('actual keyword-only GPU conditioner forward required')
            attention = output.attention
            candidates = attention.candidate_indices
            valid = inputs['source_valid'].flatten(1).gather(1, candidates.flatten(1)).reshape_as(candidates)
            valid = valid & attention.query_valid[..., None]
            logits = attention.matching_logits.flatten(0, 1)
            with torch.autocast('cuda', enabled=False):
                prior = masked_probabilities(logits[:, None].expand(-1, 4, -1), valid.flatten(0, 1)).mean(1).reshape_as(attention.candidate_weights)
                native = inputs['source_values'].flatten(1, 2).float()
                correction = (output.source_numerator.flatten(1, 2).float()
                              - output.semantic_weight.flatten(1, 2).float() * native)
                rms = correction.square().mean(-1).sqrt()
                relative = rms / native.square().mean(-1).sqrt().clamp_min(1e-12)
            if arm == 'E_FIXED_RAW' and not torch.equal(prior, attention.candidate_weights):
                raise ValueError('actual fixed aggregation differs from reconstructed frozen prior')
            for b in range(len(candidates)):
                observed_rays = torch.where(inputs['observed_query'][b, None], inputs['ray_ids'][b], -1)
                arrays = dict(ray_ids=observed_rays, patch_ids=inputs['target_patch_ids'][b],
                              candidates=candidates[b], candidate_valid=valid[b], prior=prior[b],
                              weights=attention.candidate_weights[b], source_valid=inputs['source_valid'][b].flatten(),
                              use_probability=output.use_probability[b], correction_rms=rms[b],
                              relative_correction_rms=relative[b])
                hashes = {key: tensor_sha(inputs[key][b]) for key in
                          ['native_voxels', 'target_native_voxels', 'frame_target_keys', 'source_keys',
                           'source_values', 'ray_ids', 'target_patch_ids', 'observed_query', 'source_valid']}
                batch_captures.append((
                    {key: (value.detach().float() if value.is_floating_point() else value.detach()).cpu().numpy().copy()
                     for key, value in arrays.items()}, hashes))
            # Returning None is essential: the hook never replaces the native output.

        def same(left, right):
            if isinstance(left, torch.Tensor): return torch.equal(left, right)
            if isinstance(left, dict): return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
            if isinstance(left, (tuple, list)): return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
            return left == right

        with torch.no_grad(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
            for start in range(0, 32, 8):
                batchids = ids[start:start + 8]
                pack = dataset.collate_fn([dataset[byid[sid]] for sid in batchids])
                pack = recursive_to_device(pack if isinstance(pack, dict) else pack[0], trainer.device)
                pack.pop('_dataset_name', None); pack.pop('_sha256', None); pack.pop('x_0')
                baseline = trainer.get_inference_cond(**pack) if start == 0 else None
                handle = trainer.deform_conditioner.register_forward_hook(observe, with_kwargs=True)
                try: actual = trainer.get_inference_cond(**pack)
                finally: handle.remove()
                if start == 0 and not same(actual, baseline):
                    raise ValueError('observer changed actual conditioning tensors')
                if len(batch_captures) != 8:
                    raise ValueError('exactly one native batch8 observation per asset required')
                for sid, (arrays, hashes) in zip(batchids, batch_captures):
                    path = out / arm / 'observations' / (sid + '.npz'); path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(path, **arrays)
                    observations[arm].append(dict(sample_id=sid,source_cluster_id=index['source_clusters_by_sample'][sid],
                                                  path=str(path), SHA256=file_sha256(path), input_tensor_SHA256=hashes))
                batch_captures.clear()
                print(json.dumps(dict(phase='GPU_native_condition_observation', arm=arm, current=start + 8,total=32)), flush=True)
        if trainer.optimizer.state or trainer.step != 0 or before != state_sha(list(lora.items()) + list(source.items())):
            raise ValueError('observer changed learned weights or optimizer state')
        model_proofs[arm] = dict(actual_loaded_trainable_SHA256=before, observer_first_batch_conditioning_exact=True,
                                optimizer_updates=0,denoiser_sampling_calls=0, native_batch_size=8)
        del actual, baseline, pack, lora, source, trainer, md
        gc.collect(); torch.cuda.empty_cache()
    all_forwards_done = True
    # Only now open topology GT. It cannot affect any candidate, weight or correction.
    records = {arm: [] for arm in plan['models']}
    fixed, learned = observations['E_FIXED_RAW'], observations['E_LEARNED_RAW']
    downstream = {arm: read(spec['downstream_summary']) for arm,spec in plan['models'].items()}
    for i, sid in enumerate(ids):
        if fixed[i]['input_tensor_SHA256'] != learned[i]['input_tensor_SHA256']:
            raise ValueError('fixed/learned observations used different native inputs')
        with np.load(fixed[i]['path'],allow_pickle=False) as f, np.load(learned[i]['path'],allow_pickle=False) as l:
            for key in ['ray_ids','patch_ids','candidates','candidate_valid','source_valid','prior']:
                if not np.array_equal(f[key],l[key]): raise ValueError('fixed/learned evidence support/prior differs: '+key)
        label = plan['labels'][sid]; check.check(label['path'], label['receipt'])
        row = dataset.rows[byid[sid]]
        with np.load(label['path'], allow_pickle=False) as truth:
            if (str(truth['sample_id']) != sid or str(truth['split']) != 'validation'
                    or str(truth['input_sha256']) != fingerprint(row,dataset._resolve)
                    or str(truth['input_sha256']) != matching_reference['records'][i]['GT_fingerprint']):
                raise ValueError('GT labels belong to a different native input')
            labels = {key: truth[key].copy() for key in ['positive_indices','positive_visible','target_foreground']}
        for arm in plan['models']:
            observation = observations[arm][i]
            with np.load(observation['path'],allow_pickle=False) as z: counts = utilization(**dict(z),**labels)
            values = {role: downstream[arm]['per_asset'][role]['SELECTED'][i] for role in ['target8','source9_10','all8_9_10']}
            if any(v['sample_id'] != sid or v['source_cluster_id'] != row['source_cluster_id'] for v in values.values()):
                raise ValueError('downstream result is a different asset/source')
            record = dict(**observation, counts=counts, GT_fingerprint=matching_reference['records'][i]['GT_fingerprint'],
                          all_model_forwards_precede_GT_open=all_forwards_done, existing_downstream=values,
                          SS_noise0=index['SS_archives'][arm]['path'])
            records[arm].append(record); write(out/arm/'records'/f'{i:02d}.json',record)
    summaries = {arm:summarize_counts(rows) for arm,rows in records.items()}
    population = validate_native_population(records, matching_reference, read(plan['ray_address_evidence']))
    guard()
    report = dict(status='PASS_ACTUAL_TRAINED_RAY_ATTENTION_UTILIZATION_DEV32',assets=32,source_clusters=32,
                  models=model_proofs,summaries=summaries,records=records,
                  recoverable_mass_intervals={arm:mass_intervals(rows) for arm,rows in records.items()},
                  actual_relation_population=population,
                  actual_all_NN_forward_device='cuda',
                  physical_GPU=gpu,GPU_name=torch.cuda.get_device_name(0),actual_TF32=torch.backends.cuda.matmul.allow_tf32,
                  peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                  all_forwards_precede_GT_open=True,uniform_K_is_diagnostic_not_matched_training=True,
                  downstream_is_closed_existing_measurement_not_new_generation=True,
                  correspondence_or_transfer_causality_not_claimed=True,optimizer_updates=0,TEST=False,
                  model_adopted=False,full_goal_complete=False,verified_inputs=check.inputs)
    write(out/'audit.json',report);print(report['status'],flush=True)


if __name__ == '__main__': main()
