"""Exact saved Spatial480 native SS32 readout controls; no new training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .evaluate_ss_artifact_subset import capture_ss_artifacts, validate_subset_reference
from .spatial_readout_controls import CONTROLS, spatial_readout_control


def tensor_digest(value):
    dense=value.detach().contiguous().cpu()
    digest=hashlib.sha256(str((tuple(dense.shape),str(dense.dtype))).encode())
    digest.update(dense.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def fingerprint(value):
    if isinstance(value, torch.Tensor): return tensor_digest(value)
    if isinstance(value, dict): return {k:fingerprint(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [fingerprint(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    raise ValueError('explicit native tensor/scalar fingerprint required')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());control=plan['control']
    if control not in CONTROLS: raise ValueError('named native spatial3 readout control required')
    check=ClosedInputs(plan['canonical_local_root'])
    def guard():
        check.check(args.plan,args.plan.parent/'file4-closure.json')
        for p,digest in {**plan['source_SHA256'],**plan['input_SHA256']}.items():
            if sha(p)!=digest: raise ValueError('bound spatial diagnostic source/input changed: '+p)
        for p,receipt in plan['closed_inputs'].items(): check.check(p,receipt)
    guard()
    refpath=Path(plan['reference_archive']);check.check(refpath,refpath.parent/'file4-closure.json')
    reference=json.loads(refpath.read_text());artifacts=reference['native_SS_artifacts']['artifacts']
    if len(reference['per_asset'])!=32 or len(artifacts)!=32: raise ValueError('actual archived spatial32 reference required')
    if control!='normal':
        proof=Path(plan['normal_proof']);check.check(proof,proof.parent/'file4-closure.json')
        normal=json.loads(proof.read_text())['spatial_readout_diagnostic']
        if (normal['control']!='normal' or not normal['actual_original32_prediction_latents_and_voxels_exact']
                or normal['reference_archive_SHA256']!=sha(refpath)):
            raise ValueError('own actual exact32 normal proof required before controls')
    from . import quick_metrics as quick
    from . import member_feature_control_runtime as entry
    from .r_mediated_native_runtime import state_sha
    from .r_native_signal_probe import consumer_signal
    from pixal3d.trainers.flow_matching import member_feature_controls as consumer
    selected=consumer.MemberFeatureControlImageConditionedProjFlowMatchingCFGTrainer
    original_encode=selected._encode_deform_views;original_get_sampler=selected.get_sampler
    prior_sampler=selected.__dict__.get('get_sampler');original_add=consumer.add_source_ray_correction
    events,consumers,inputs,samplers,captured,stored=[],[],[],[],[],[]
    before={}
    def state(trainer):
        return {name:state_sha(model.state_dict().items())
                for name,model in trainer.models.items()}
    def encode(self,*args,**kwargs):
        if not captured:
            captured.append(self);before.update(state(self))
        elif captured[0] is not self: raise ValueError('one saved native trainer required')
        with spatial_readout_control(self.deform_conditioner,events,control=control):
            result=original_encode(self,*args,**kwargs)
        inputs.append(dict(encoder_arguments=fingerprint(args),encoder_keywords=fingerprint(kwargs),
            actual_native_global=fingerprint(result[0]['global']),actual_negative=fingerprint(result[1])))
        return result
    def get_sampler(self):
        sampler=original_get_sampler(self);sample=sampler.sample
        def sampled(*args,**kwargs):
            samplers.append(dict(actual_initial_noise_SHA256=tensor_digest(kwargs['noise']),
                native_global=fingerprint(kwargs['cond']['global']),negative=fingerprint(kwargs['neg_cond']),
                steps=kwargs['steps'],guidance_strength=kwargs['guidance_strength']))
            return sample(*args,**kwargs)
        sampler.sample=sampled
        return sampler
    def add(native,delta):
        result=original_add(native,delta)
        consumers.append(dict(consumer_signal(native,delta,result),
            native_projected_SHA256=tensor_digest(native['proj']),
            actual_delivered_projected_SHA256=tensor_digest(result['proj']),
            actual_precast_delta_SHA256=tensor_digest(delta)))
        return result
    previous_argv=sys.argv
    selected._encode_deform_views=encode;selected.get_sampler=get_sampler;consumer.add_source_ray_correction=add
    try:
        with capture_ss_artifacts(quick,Path(plan['output']),stored):
            sys.argv=[entry.__file__,*plan['evaluation_arguments']];entry.main()
    finally:
        sys.argv=previous_argv;selected._encode_deform_views=original_encode
        if prior_sampler is None: delattr(selected,'get_sampler')
        else: selected.get_sampler=prior_sampler
        consumer.add_source_ray_correction=original_add
    path=Path(plan['output'])/'evaluation.json';actual=json.loads(path.read_text())
    if (len(captured)!=1 or captured[0].optimizer.state or before!=state(captured[0])
            or len(stored)!=32 or not all(len(v)==4 for v in [events,consumers,inputs,samplers])):
        raise ValueError('complete frozen spatial32/native batch8 diagnostic required')
    for a,b in zip(actual['per_asset'],reference['per_asset']):
        if any(a[k]!=b[k] for k in ['sample_id','source_cluster_id','seed','target_voxels_sha256','target_voxel_shape','target_representation']):
            raise ValueError('source/GT/noise identity changed')
    original_exact=False
    if control=='normal':
        validate_subset_reference(actual,reference,stored)
        for a,b in zip(stored,artifacts):
            if any(a[k]!=b[k] for k in ['sample_id','seed','prediction_voxels_SHA256','target_voxels_SHA256','occupancy_counts']):
                raise ValueError('actual archived normal occupancy differs')
            old=refpath.parent/b['artifact'];check.check(old,refpath.parent/'file4-closure.json')
            with np.load(Path(plan['output'])/a['artifact']) as left,np.load(old) as right:
                if not all(np.array_equal(left[k],right[k]) for k in
                    ['prediction_latent','target_latent','prediction_occupancy_packed','target_occupancy_packed']):
                    raise ValueError('actual original32 normal prediction/GT latent or occupancy differs')
        original_exact=True
    diag=dict(status='PASS_ACTUAL_SAVED_SPATIAL480_READOUT_CONTROL32',control=control,
        reference_archive=str(refpath),reference_archive_SHA256=sha(refpath),
        actual_original32_prediction_latents_and_voxels_exact=original_exact,
        actual_trained_updates=480,optimizer_updates=0,actual_unused_optimizer_states=0,
        actual_loaded_model_state_SHA256=before,actual_model_states_unchanged=True,
        actual_source_checkpoint_SHA256={p:sha(p) for p in plan['model_files']},
        events=events,consumer_events=consumers,encoder_events=inputs,sampler_events=samplers,
        actual_SS_artifacts=stored,actual_native3_proposal_control=True,
        prior_only_is_not_uniform_NoScore=True,Top1_is_not_retrained_NoScore=True,
        zero_transformed_value_does_not_remove_R=True,wrong_V_is_not_NoCorr=True,
        no_independent_module_training_quality_or_model_adoption_claim=True,
        verified_inputs=check.inputs,full_goal_complete=False)
    actual['spatial_readout_diagnostic']=diag;actual['diagnostic_only']=True
    guard();path.write_text(json.dumps(actual,indent=2)+'\n')
    print(json.dumps(dict(status=diag['status'],control=control,assets=32,optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
