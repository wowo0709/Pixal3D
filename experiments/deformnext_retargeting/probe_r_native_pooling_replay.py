"""Read-only diagnosis of source-ray floating-point reduction repeatability.

Reuse the strict saved-pair loader, intercept its quick evaluation, and inspect
one real eight-asset conditioning batch. This is an engineering diagnostic;
it neither changes a trainer nor certifies generation quality or epoch replay.
"""
import copy
import json
from pathlib import Path


def compare_tensors(first, second):
    import torch
    if first.shape != second.shape or first.dtype != second.dtype:
        raise ValueError('repeat comparisons require identical shapes and dtypes')
    difference = first.double() - second.double()
    if not torch.isfinite(difference).all():
        raise ValueError('nonfinite repeated tensor')
    return dict(exact=torch.equal(first, second),
                different_elements=int(torch.count_nonzero(difference)),
                max_absolute=float(difference.abs().max()),
                RMS=float(difference.square().mean().sqrt()))


def diagnose(trainer, dataset, validation):
    import torch
    from pixal3d.utils.data_utils import recursive_to_device
    from pixal3d.modules.ray_group_query import pool_ray_queries
    from .quick_metrics import panel_indices
    from .r_mediated_native_runtime import state_sha, tensor_sha
    from pixal3d.modules import r_mediated_source_contrast as source_code

    module = trainer.deform_conditioner
    if module is None or module.training or torch.is_grad_enabled():
        raise ValueError('loaded eval source and no_grad required')
    before = state_sha(module.state_dict().items())
    selected = panel_indices(dataset, validation['quick_metrics'])[:8]
    pack = dataset.collate_fn([dataset[i] for i in selected])
    pack = recursive_to_device(pack if isinstance(pack, dict) else pack[0], trainer.device)
    for key in ('x_0', '_dataset_name', '_sha256'):
        pack.pop(key, None)
    captured = {}

    class Captured(Exception):
        pass

    def intercept(_module, args, kwargs):
        if args or captured:
            raise ValueError('one explicit source forward expected')
        captured.update(kwargs)
        raise Captured()

    hook = module.register_forward_pre_hook(intercept, with_kwargs=True)
    try:
        try:
            trainer.get_inference_cond(**pack)
        except Captured:
            pass
    finally:
        hook.remove()
    if not captured:
        raise ValueError('actual native conditioning did not reach source module')
    deterministic = torch.are_deterministic_algorithms_enabled()
    previous_pool = source_code.pool_ray_queries
    report = dict(asset_ids=[dataset.rows[i]['sample_id'] for i in selected],
        actual_input_SHA256={k: tensor_sha(v) for k, v in captured.items() if torch.is_tensor(v)},
        allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        GPU=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
        pooling={}, source={})
    try:
        # Inspect both the native DINO query and cached target context.
        for field in ('native_voxels', 'target_native_voxels'):
            report['pooling'][field] = {}
            for enabled in (False, True):
                torch.use_deterministic_algorithms(enabled)
                values = [pool_ray_queries(captured[field], captured['ray_ids'], num_rays=1024).values
                          for _ in range(6)]
                report['pooling'][field][str(enabled)] = [compare_tensors(values[0], x) for x in values[1:]]
                del values

        def stable_pool(*args, **kwargs):
            old = torch.are_deterministic_algorithms_enabled()
            try:
                torch.use_deterministic_algorithms(True)
                return previous_pool(*args, **kwargs)
            finally:
                torch.use_deterministic_algorithms(old)

        # This local intervention changes ONLY pooling's reduction algorithm.
        for label, pool in [('native', previous_pool), ('deterministic_pool_only', stable_pool)]:
            source_code.pool_ray_queries = pool
            torch.use_deterministic_algorithms(False)
            outputs = []
            for _ in range(3):
                value = module(**captured)
                outputs.append((value.source_numerator.detach().clone(),
                                value.semantic_weight.detach().clone(),
                                module.last_feature_probabilities[1].detach().clone()))
            report['source'][label] = {
                name: [compare_tensors(outputs[0][i], out[i]) for out in outputs[1:]]
                for i, name in enumerate(('source_numerator', 'semantic_weight', 'feature_probabilities'))}
            del outputs
    finally:
        source_code.pool_ray_queries = previous_pool
        torch.use_deterministic_algorithms(deterministic)
    if before != state_sha(module.state_dict().items()):
        raise ValueError('read-only diagnostic changed saved source parameters/buffers')
    report.update(status='COMPLETED_ACTUAL_SOURCE_REDUCTION_REPEATABILITY_DIAGNOSIS',
        loaded_source_state_unchanged=True, optimizer_updates=0, denoiser_sampler_calls=0,
        exact_saved_pair_epoch_metric_replay=False, quality_gain_proven=False,
        full_goal_complete=False)
    return report


def main():
    import torch
    from . import evaluate_r_native_checkpoint as loader
    from . import quick_metrics
    original = quick_metrics.evaluate_quick
    report = None

    class Finished(Exception):
        pass

    def inspect(trainer, dataset, validation):
        nonlocal report
        with torch.no_grad(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
            report = diagnose(trainer, dataset, copy.deepcopy(validation))
        path = Path(trainer.output_dir).parent / 'audit.json'
        path.write_text(json.dumps(report, indent=2) + '\n')
        raise Finished()

    quick_metrics.evaluate_quick = inspect
    try:
        try:
            loader.main()
        except Finished:
            pass
    finally:
        quick_metrics.evaluate_quick = original
    if report is None:
        raise ValueError('saved-pair loader did not execute diagnosis')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
