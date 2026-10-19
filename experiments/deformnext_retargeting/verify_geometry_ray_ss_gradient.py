"""Native SS identity and eight-tensor learnability without appearance VO."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from pixal3d import models
from pixal3d.datasets.deformnext_deformation_query import DeformNeXtDeformationQuerySS64
from pixal3d.modules.ray_geometry_retargeting import RayGeometryRetargetingConditioner
from pixal3d.utils.data_utils import recursive_to_device
from .anchor_ray_runtime import AnchorRayCompositionTrainer
from .verify_anchor_ray_ss_gradient import sha, tensor_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cache-index', type=Path, required=True)
    parser.add_argument('--utility-head', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in {'0', '1', '2', '3'} or visible not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one assigned physical GPU0-3 required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    torch.use_deterministic_algorithms(True)
    repo = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), args.config, args.utility_head,
        *[repo / path for path in ['pixal3d/modules/ray_geometry_retargeting.py',
            'pixal3d/modules/ray_deformation_query.py', 'pixal3d/modules/ray_feature_attention.py',
            'pixal3d/modules/task_matching_attention.py', 'pixal3d/datasets/deformnext_deformation_query.py']]]
    bound = {str(path): sha(path) for path in sources}
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    ds = DeformNeXtDeformationQuerySS64(cfg['data_dir'], deformation_query_channels=2048, **cfg['dataset']['args'])
    rows = list(map(json.loads, args.cache_index.read_text().splitlines()))
    selected = [row['sample_id'] for row in rows if row['role'] == 'train' and row['usable_queries'] > 0][:2]
    index = {row['sample_id']: i for i, row in enumerate(ds.rows)}
    pack = recursive_to_device(ds.collate_fn([ds[index[sid]] for sid in selected]), torch.device('cuda:0'))
    with torch.device('meta'):
        md = {name: getattr(models, spec['name'])(**spec['args']) for name, spec in cfg['models'].items()}
    kwargs = dict(cfg['trainer']['args'])
    kwargs.update(batch_size_per_gpu=2, batch_split=1, num_workers=0,
        prefetch_data=False, max_steps=1, p_uncond=0.)
    trainer = AnchorRayCompositionTrainer(md, ds, **kwargs, output_dir=str(args.output),
        load_dir=None, step=None, wandb_run=None)
    original = trainer.deform_conditioner
    reference = {name: value.detach().cpu().clone() for name, value in original.reference_state().items()}
    base_pack = {key: value for key, value in pack.items()
        if key not in ['transport_target_context', 'transport_source_context']}

    def loss(batch):
        torch.manual_seed(101)
        torch.cuda.manual_seed_all(101)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            values, _ = trainer.training_losses(**batch)
        return values['loss']

    with torch.no_grad():
        baseline = float(loss(base_pack))
        evidence = trainer._last_deform_condition.evidence.clone()
    args_module = dict(cfg['trainer']['args']['deform_conditioner'], arm='c2', deformation_dim=2048)
    feature = RayGeometryRetargetingConditioner(**args_module).cuda()
    feature.load_reference(reference)
    feature.ray_readout.score_selector.load_state_dict(
        torch.load(args.utility_head, map_location='cpu', weights_only=True), strict=True)
    if (hasattr(feature.ray_readout, 'value') or hasattr(feature.ray_readout, 'output')
            or torch.count_nonzero(feature.ray_readout.query.weight)):
        raise ValueError('appearance VO must be physically absent and matching prior initially preserved')
    trainer.deform_conditioner = feature
    trainer.models['deform_conditioner'] = feature
    trainer.training_models['deform_conditioner'] = feature
    params = [p for p in feature.parameters() if p.requires_grad]
    if len(params) != 8 or len(list(feature.ray_readout.parameters())) != 7:
        raise ValueError('seven matching tensors plus one fresh physical query projection required')
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    before = tensor_digest(trainer.models['denoiser'])
    results = []
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for update in range(3):
        optimizer.zero_grad(set_to_none=True)
        value = loss(pack)
        if update == 0 and (float(value.detach()) != baseline
                or not torch.equal(trainer._last_deform_condition.evidence, evidence)):
            raise ValueError('geometry-only module changed actual initial native SS loss or evidence')
        value.backward()
        gradients = {name: float(p.grad.detach().float().abs().sum()) if p.grad is not None else None
            for name, p in feature.named_parameters() if p.requires_grad}
        if (any(v is None or not torch.isfinite(torch.tensor(v)) for v in gradients.values())
                or gradients['retargeter.deformation_projection.weight'] <= 0):
            raise ValueError('actual geometry path must learn from the first SS backward')
        if update == 2 and any(v <= 0 for v in gradients.values()):
            raise ValueError('all eight native tensors must receive gradients by update3')
        stats = dict(feature.last_feature_attention_stats)
        if stats['deformation_query_RMS'] <= 0 or stats['feature_residual_RMS'] != 0:
            raise ValueError('actual nonzero deformation and absent appearance correction required')
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
        row = dict(update=update+1, loss=float(value.detach()), gradient_L1=gradients, statistics=stats)
        results.append(row)
        print(json.dumps(row), flush=True)
    if tensor_digest(trainer.models['denoiser']) != before or any(
            not torch.equal(value, feature.reference_state()[name].detach().cpu()) for name, value in reference.items()):
        raise ValueError('frozen NN, R, or original matcher/fusion changed')
    if any(sha(path) != digest for path, digest in bound.items()):
        raise ValueError('native geometry proof source changed')
    result = dict(status='PASS_ACTUAL_SS_GEOMETRY_ONLY_IDENTITY_AND_EIGHT_TENSOR_GRADIENT',
        variant='ray-geometry-only-prior-anchored', source_SHA256=bound, TRAIN_samples=selected,
        initial_loss=baseline, new_initial_loss=results[0]['loss'], initial_evidence_exact=True,
        original_Top1_appearance_preserved=True, appearance_VO_physically_absent=True,
        seven_matching_tensors=True, additional_query_initial_weights_zero=True,
        physical_deformation_channels=2048, retargeter_input_channels=4096,
        actual_deformation_observations_nonzero=True, all_eight_new_tensors_receive_actual_SS_gradients=True,
        native_NN_and_R_exact=True, common_R_parameter_shapes_unchanged=True,
        temporary_updates=3, scientific_training_updates=0, NN_updates=0, R_updates=0,
        GT_correspondence_or_deformation_input=False, prototype_discarded=True,
        physical_GPU=visible, peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        seconds=time.monotonic()-start, updates=results, model_adopted=False, full_goal_complete=False,
        scope='Actual TRAIN micro2 identity and gradients; not DDP capacity or quality improvement')
    (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
