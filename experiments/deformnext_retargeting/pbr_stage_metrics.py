"""Bounded native PBR validation with real input/noise hashes per asset."""
from __future__ import annotations

import math

import numpy as np

from .evaluate_pbr_streaming import aggregate_metrics, tensor_sha


def install_pbr_stage_metrics():
    from . import quick_metrics as quick
    if getattr(quick.evaluate_quick, '_pbr_stage_dispatch', False):
        return
    original_evaluate, original_pbr, original_sample = quick.evaluate_quick, quick._pbr_metrics, quick._sample

    def evaluate(trainer, dataset, config):
        name = config['dataset_name']
        if name == 'DeformNeXtDeformationQueryPBR1024':
            if getattr(dataset, 'deformation_query_channels', None) != 2048:
                raise ValueError('real physical2048 dataset required for PBR metric dispatch')
            metric_config = dict(config, dataset_name='DeformNeXtPBR1024')
        elif name == 'DeformNeXtPBR1024':
            metric_config = config
        else:
            return original_evaluate(trainer, dataset, config)
        events = []

        def sample(actual_trainer, actual_dataset, index, steps, seed):
            get_sampler = actual_trainer.get_sampler
            record = dict(sample_id=actual_dataset.rows[index]['sample_id'], seed=seed,
                source_cluster_id=actual_dataset.rows[index]['source_cluster_id'],
                conditioning_input_sha256=actual_dataset.rows[index]['conditioning_input_sha256'])

            def factory():
                sampler = get_sampler()
                call = sampler.sample

                def generated(*args, **kwargs):
                    record['actual_noise_SHA256'] = tensor_sha(kwargs['noise'])
                    return call(*args, **kwargs)

                sampler.sample = generated
                return sampler

            actual_trainer.get_sampler = factory
            try:
                prediction,target,pack = original_sample(actual_trainer,actual_dataset,index,steps,seed)
            finally:
                actual_trainer.get_sampler = get_sampler
            record.update(actual_RGB_input_SHA256=tensor_sha(pack['cond']),
                actual_GT_PBR_latent_SHA256=tensor_sha(target),
                actual_GT_Shape_condition_SHA256=tensor_sha(pack['concat_cond']))
            events.append(record)
            return prediction,target,pack

        def streaming(actual_trainer, actual_dataset, indices, options):
            seeds = quick.panel_seeds(actual_dataset,indices,options)
            panels,numeric,cases = [],[],[]
            for number,(index,seed) in enumerate(zip(indices,seeds)):
                metrics,panel = original_pbr(actual_trainer,actual_dataset,[index],dict(options,seed=seed))
                if len(events) != number+1 or not all(math.isfinite(v) for v in metrics.values()):
                    raise ValueError('exactly one finite native rendering result and input audit per asset required')
                cases.append(dict(events[-1],**metrics))
                numeric.append(metrics)
                panels.append(panel)
            actual_trainer._quick_per_asset = cases
            return aggregate_metrics(numeric),np.concatenate(panels,axis=0)

        quick._sample,quick._pbr_metrics = sample,streaming
        try:
            return original_evaluate(trainer,dataset,metric_config)
        finally:
            quick._sample,quick._pbr_metrics = original_sample,original_pbr

    evaluate._pbr_stage_dispatch = True
    quick.evaluate_quick = evaluate
