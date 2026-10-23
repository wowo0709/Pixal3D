"""Frozen NoDQ288 direct-C delivery diagnostic using the original evaluator."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .direct_delivery_controls import CONTROLS, direct_delivery_control
from .evaluate_spatial_readout_control import fingerprint, tensor_digest
from .evaluate_ss_artifact_subset import capture_ss_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    control = plan['control']
    if control not in CONTROLS:
        raise ValueError('explicit direct-C control required')
    check = ClosedInputs(plan['canonical_local_root'])

    def guard():
        check.check(args.plan, args.plan.parent/'file4-closure.json')
        for path, digest in {**plan['source_SHA256'], **plan['input_SHA256']}.items():
            if sha(path) != digest:
                raise ValueError('bound direct diagnostic source/input changed: '+path)
        for path, receipt in plan['closed_inputs'].items():
            check.check(path, receipt)

    guard()
    if control != 'normal':
        proof = Path(plan['normal_gate'])
        check.check(proof, proof.parent/'file4-closure.json')
        gate = json.loads(proof.read_text())
        if (gate['status'] != 'PASS_ACTUAL_NODQ288_TWO_INDEPENDENT_NORMALS'
                or gate['actual_checkpoint_SHA256'] != {p: sha(p) for p in plan['model_files']}):
            raise ValueError('own actual two independent normal loads required before interventions')
    from . import quick_metrics as quick
    from . import evaluate_direct_no_query as entry
    from .r_mediated_native_runtime import state_sha
    from pixal3d.trainers.flow_matching.flow_matching import ImageConditionedProjFlowMatchingCFGTrainer as selected

    original_encode, original_sampler = selected._encode_deform_views, selected.get_sampler
    prior_encode, prior_sampler = selected.__dict__.get('_encode_deform_views'), selected.__dict__.get('get_sampler')
    events, encoders, samplers, captured, stored = [], [], [], [], []
    before = {}

    def state(trainer):
        return {name: state_sha(model.state_dict().items()) for name, model in trainer.models.items()}

    def encode(self, *args, **kwargs):
        if not captured:
            captured.append(self)
            before.update(state(self))
        elif captured[0] is not self:
            raise ValueError('one frozen NoDQ trainer required')
        actual_inputs = dict(arguments=fingerprint(args), keywords=fingerprint(kwargs))
        result = original_encode(self, *args, **kwargs)
        encoders.append(dict(**actual_inputs, actual_global=fingerprint(result[0]['global']),
                             actual_negative=fingerprint(result[1])))
        return result

    def get_sampler(self):
        sampler = original_sampler(self)
        sample = sampler.sample

        def sampled(*args, **kwargs):
            samplers.append(dict(actual_noise_SHA256=tensor_digest(kwargs['noise']),
                actual_global=fingerprint(kwargs['cond']['global']), negative=fingerprint(kwargs['neg_cond']),
                steps=kwargs['steps'], guidance=kwargs['guidance_strength']))
            return sample(*args, **kwargs)

        sampler.sample = sampled
        return sampler

    previous = sys.argv
    selected._encode_deform_views, selected.get_sampler = encode, get_sampler
    try:
        with direct_delivery_control(selected, events, control=control, fingerprint=fingerprint), \
                capture_ss_artifacts(quick, Path(plan['output']), stored):
            sys.argv = [entry.__file__, *plan['evaluation_arguments']]
            entry.main()
    finally:
        sys.argv = previous
        for name, prior in [('_encode_deform_views', prior_encode), ('get_sampler', prior_sampler)]:
            if prior is None:
                delattr(selected, name)
            else:
                setattr(selected, name, prior)
    path = Path(plan['output'])/'evaluation.json'
    actual = json.loads(path.read_text())
    if (len(captured) != 1 or captured[0].optimizer.state or before != state(captured[0])
            or len(stored) != 32 or any(len(v) != 4 for v in [events, encoders, samplers])):
        raise ValueError('complete frozen NoDQ288/four batch8 evaluation required')
    for artifact in stored:
        with np.load(Path(plan['output'])/artifact['artifact']) as data:
            artifact.update(prediction_latent_SHA256=tensor_digest(torch.from_numpy(data['prediction_latent'])),
                            target_latent_SHA256=tensor_digest(torch.from_numpy(data['target_latent'])))
    old_rows = json.loads(Path(plan['quick_reference']).read_text())
    if isinstance(old_rows, dict):
        old_rows = old_rows['rows']
    old = next(row for row in old_rows if row['step'] == 288)['quick_per_asset']
    compared_fields = ['sample_id', 'source_cluster_id', 'seed', 'target_voxels_sha256',
                       'target_voxel_shape', 'target_representation', 'tp', 'fp', 'fn',
                       'voxel_iou', 'voxel_precision', 'voxel_recall', 'voxel_f1']
    old_identity_fields = compared_fields[:6]
    if len(old) != 32 or any(any(a[k] != b[k] for k in old_identity_fields)
                            for a, b in zip(actual['per_asset'], old)):
        raise ValueError('original quick288 ordered asset/GT/seed identity changed')
    quick_exact = all(all(a[k] == b[k] for k in compared_fields) for a, b in zip(actual['per_asset'], old))
    actual['direct_delivery_diagnostic'] = dict(
        status='PASS_ACTUAL_FROZEN_NODQ288_DIRECT_C_CONTROL32', control=control,
        actual_trained_step=288, optimizer_updates=0, actual_unused_optimizer_states=0,
        actual_loaded_model_state_SHA256=before, actual_model_states_unchanged=True,
        actual_checkpoint_SHA256={p: sha(p) for p in plan['model_files']},
        actual_numeric_runtime=dict(torch_version=torch.__version__,
            CUDA_matmul_allow_TF32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_TF32=torch.backends.cudnn.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        actual_original_quick288_counts_and_metrics_exact=quick_exact,
        historical_quick_prediction_hashes_unavailable=True,
        protocol='independent deterministic normal parity first; historical quick metric parity reported separately',
        events=events, encoder_events=encoders, sampler_events=samplers, actual_SS_artifacts=stored,
        original_matching_F_attention_and_R_computation_preserved=True,
        direct_C_coefficient_only_intervened=True, no_new_training=True,
        not_NoScore_NoCorr_or_R_off=True, not_retrained_module_effect=True,
        verified_inputs=check.inputs, model_adopted=False, full_goal_complete=False)
    actual['diagnostic_only'] = True
    guard()
    path.write_text(json.dumps(actual, indent=2)+'\n')
    print(json.dumps(dict(status=actual['direct_delivery_diagnostic']['status'], control=control,
                         assets=32, original_quick_counts_exact=quick_exact)), flush=True)


if __name__ == '__main__':
    main()
