"""Cache actual SS VAL32 native ray observations, with GT loaded after forward.

No RGB rendering, latent encoding, Omega inference or optimizer is performed.
This is a matching diagnostic panel, not an independent publication test.
"""
from pathlib import Path
import argparse
import json
import os
import time

import numpy as np
import torch

from pixal3d.datasets.deformnext import DeformNeXtSS64
from pixal3d.modules.ray_feature_attention import RayFeatureAttentionConditioner
from . import ray_set_head_training as shared
from .topology_labels import fingerprint
from .validation_ray_matching import positive_sets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'panel-validation', 'auxiliary-index', 'cold-conditioner', 'teacher-root', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if physical not in {'0', '1', '2', '3'} or physical not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one explicitly assigned GPU0-3 required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.1, 0)
    torch.use_deterministic_algorithms(True)
    cfg = shared.read(args.config)
    validation = cfg['validation']
    records = list(map(json.loads, args.panel_validation.read_text().splitlines()))
    selected = [r for r in records if r['step'] == 96]
    if len(selected) != 1 or len(selected[0]['quick_per_asset']) != 32:
        raise ValueError('fixed actual initial96 native validation32 required')
    panel = selected[0]['quick_per_asset']
    if len({r['source_cluster_id'] for r in panel}) != 32:
        raise ValueError('32 declared validation source clusters required')
    auxiliary = list(map(json.loads, args.auxiliary_index.read_text().splitlines()))
    if ({r['sample_id'] for r in panel} & {r['sample_id'] for r in auxiliary}
            or {r['source_cluster_id'] for r in panel} & {r['source_cluster_id'] for r in auxiliary}):
        raise ValueError('actual validation and auxiliary fit/calibration clusters overlap')
    from . import validation_ray_matching, topology_labels
    from pixal3d.modules import ray_feature_attention, task_matching_attention
    from pixal3d.datasets import deformnext
    paths = [Path(__file__), Path(shared.__file__), Path(validation_ray_matching.__file__),
        Path(topology_labels.__file__), Path(ray_feature_attention.__file__),
        Path(task_matching_attention.__file__), Path(deformnext.__file__),
        args.config, args.auxiliary_index, args.cold_conditioner, Path(validation['roots_file'])]
    bound = {str(path.resolve()): shared.sha(path) for path in paths}
    # The live panel file may append later epochs. Bind only its immutable row.
    panel96 = json.dumps(selected[0], sort_keys=True)
    dataset = DeformNeXtSS64(validation['roots_file'], **validation['dataset_args'])
    rows = {r['sample_id']: r for r in dataset.rows}
    kwargs = dict(cfg['trainer']['args']['deform_conditioner'], arm='c1')
    kwargs.pop('deformation_dim')
    model = RayFeatureAttentionConditioner(**kwargs).cuda().eval().requires_grad_(False)
    state = torch.load(args.cold_conditioner, map_location='cpu', weights_only=True)
    if torch.count_nonzero(state['ray_readout.output.weight']):
        raise ValueError('untouched cold output required')
    state.pop('retargeter.deformation_projection.weight')
    model.load_state_dict(state, strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'panel96.json').write_text(json.dumps(selected[0], indent=2)+'\n')
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    index = []
    for item in panel:
        row = rows[item['sample_id']]
        if row['split'] != 'validation':
            raise ValueError('actual validation manifest row required')
        for path in [dataset._resolve(row['transport_cache']), dataset.match_root/(row['sample_id']+'.npz')]:
            bound[str(path)] = shared.sha(path)
        data = dataset._common(row)
        transport = {key[len('transport_'):]: value[None].cuda() for key, value in data.items()
            if key.startswith('transport_') and key not in ['transport_patch_grid', 'transport_patch_ids']}
        if {'deformation', 'spatial_indices', 'positive_indices', 'positive_visible'} & transport.keys():
            raise ValueError('loss-only geometry cannot enter native forward')
        captured = []

        def capture(module, inputs, output):
            captured.append((inputs[0].detach().cpu(), inputs[1].detach().cpu(),
                inputs[4].detach().cpu(), inputs[5].detach().cpu(), output[1].detach().cpu()))

        handle = model.ray_readout.register_forward_hook(capture)
        try:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(**transport)
        finally:
            handle.remove()
        patches = result.query_valid[0].nonzero().flatten().cpu().numpy()
        ids = result.candidate_indices[0, patches].cpu().numpy().astype(np.int32)
        native = torch.cat([r[0] for r in captured]).numpy() if captured else np.empty((0, 1024), np.float32)
        context = torch.cat([r[1] for r in captured]).numpy() if captured else native.copy()
        valid = torch.cat([r[2] for r in captured]).numpy() if captured else np.empty((0, 64), bool)
        prior = torch.cat([r[3] for r in captured]).numpy() if captured else np.empty((0, 64), np.float32)
        expected = torch.cat([r[4] for r in captured]) if captured else torch.empty(0, 4, 64)
        bank = data['transport_source_values'].flatten(0, 1).numpy()
        keys = data['transport_source_keys'].flatten(0, 1).numpy()
        replay = []
        for begin in range(0, len(patches), model.feature_chunk_size):
            take = slice(begin, begin+model.feature_chunk_size)
            values = [torch.from_numpy(v).cuda() for v in [native[take], context[take],
                keys[ids[take]].astype(np.float32), bank[ids[take]].astype(np.float32), valid[take], prior[take]]]
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                replay.append(model.ray_readout(*values)[1].cpu())
        actual = torch.cat(replay) if replay else torch.empty(0, 4, 64)
        if not valid.all() or not torch.equal(actual, expected):
            raise ValueError('actual native cold-head probability replay must be exact')
        # Only after all native forward calls, open and validate loss labels.
        label_path = args.teacher_root/(row['sample_id']+'.npz')
        bound[str(label_path)] = shared.sha(label_path)
        with np.load(label_path, allow_pickle=False) as teacher:
            digest = fingerprint(row, dataset._resolve)
            if digest != str(teacher['input_sha256']):
                raise ValueError('current validation RGB/geometry teacher fingerprint differs')
            relation, exact, near = positive_sets(ids, patches, teacher)
        folder = args.output/'validation'
        folder.mkdir(exist_ok=True)
        inputs = folder/(row['sample_id']+'.inputs.npz')
        labels = folder/(row['sample_id']+'.labels.npz')
        np.savez(inputs, schema='deformnext-ray-set-inputs-v1', query_unit='edited-target-patch-ray',
            sample_id=row['sample_id'], native_query=native, context_query=context,
            scalar_prior_logits=prior, candidate_indices=ids, candidate_valid=valid,
            source_value_bank=bank, source_key_bank=keys, query_patch_indices=patches.astype(np.int32),
            reference_cold_probabilities=expected.numpy())
        np.savez(labels, schema='deformnext-ray-set-loss-labels-v1', sample_id=row['sample_id'],
            teacher_input_sha256=digest, relation_query_indices=relation,
            positive_exact_bits=np.packbits(exact, axis=1, bitorder='little'),
            positive_near1_bits=np.packbits(near, axis=1, bitorder='little'))
        index.append(dict(role='validation', split='validation', sample_id=row['sample_id'],
            source_cluster_id=item['source_cluster_id'], query_unit='edited-target-patch-ray',
            queries=len(patches), observable_relations=len(relation), GT_teacher_fingerprint=digest,
            inputs_relative_path=str(inputs.relative_to(args.output)),
            labels_relative_path=str(labels.relative_to(args.output)),
            inputs_SHA256=shared.sha(inputs), labels_SHA256=shared.sha(labels), native_probability_replay_exact=True))
        progress = dict(completed=len(index), total=32, seconds=time.monotonic()-began)
        (args.output/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
        print(json.dumps(progress), flush=True)
        del data, transport, captured, result, replay, expected, actual
    if any(shared.sha(path) != digest for path, digest in bound.items()):
        raise ValueError('read-only validation cache source or inputs changed')
    selected_now = [r for r in map(json.loads, args.panel_validation.read_text().splitlines()) if r['step'] == 96]
    if len(selected_now) != 1 or json.dumps(selected_now[0], sort_keys=True) != panel96:
        raise ValueError('declared native initial96 panel changed')
    (args.output/'cache-index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index))
    shared.write_new(args.output/'completed.json', dict(status='PASS_ACTUAL_NATIVE_SS_VALIDATION32_MATCHING_CACHE',
        assets=32, source_clusters=32, split='validation', feature_chunk_size=model.feature_chunk_size,
        probability_replay_partition='same-native256-including-final-tail', auxiliary_fit_or_calibration_overlap=False,
        all_native_probability_replays_exact=True, GT_in_forward=False, GT_only_labels=True,
        optimizer_updates=0, rendering_or_latent_or_Omega_rebuilt=False, source_SHA256=bound,
        physical_GPU=physical, allocator_fraction=.1, peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        seconds=time.monotonic()-began, quality_gain_claimed=False, model_adopted=False, full_goal_complete=False))


if __name__ == '__main__':
    main()
