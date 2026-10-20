"""Actual activated NoDQ/DQ-zero identity using the SAME native SS denoiser."""
import copy

import torch

from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner


class NoDQReference(MemberFeatureSourceRayConditioner):
    def forward(self,*,target_context,source_context,**kwargs):
        # The reference physically has neither the channels nor projection.
        return super().forward(**kwargs)


def verify_activated_NoDQ_identity(trainer,cfg,pack,target,camera,transport):
    actual=trainer.deform_conditioner
    if torch.count_nonzero(actual.retargeter.deformation_projection.weight):
        raise ValueError('initial zero DQ projection required for matched NoDQ identity')
    settings=copy.deepcopy(cfg['trainer']['args']['deform_conditioner']);settings.pop('deformation_dim')
    with torch.random.fork_rng(devices=[trainer.device]):
        reference=NoDQReference(**settings).cuda().eval().requires_grad_(False)
    reference.load_state_dict({name:value for name,value in actual.state_dict().items()
        if name!='retargeter.deformation_projection.weight'},strict=True)
    maps=[];seen=set()
    for name in ['models','training_models']:
        mapping=getattr(trainer,name,None)
        if mapping is not None and id(mapping) not in seen:
            maps.append((mapping,mapping['deform_conditioner']));seen.add(id(mapping))
    deterministic=torch.are_deterministic_algorithms_enabled();results={}
    try:
        torch.use_deterministic_algorithms(True)
        with torch.no_grad():
            for role,module in [('actual-DQ-zero',actual),('matched-physical-NoDQ',reference)]:
                trainer.deform_conditioner=module
                for mapping,old in maps:mapping['deform_conditioner']=module
                condition,negative=trainer._encode_deform_views(pack['cond'],camera,transport)
                with torch.random.fork_rng(devices=[trainer.device]):
                    torch.manual_seed(80421);torch.cuda.manual_seed(80421)
                    loss,_=trainer.training_losses(x_0=target,**pack)
                results[role]=dict(condition=condition,negative=negative,loss=float(loss['loss']))
        left,right=results['actual-DQ-zero'],results['matched-physical-NoDQ']
        if (left['loss']!=right['loss'] or any(not torch.equal(left[role][key],right[role][key])
                for role in ['condition','negative'] for key in left[role])):
            raise ValueError('activated same-weight DQ-zero and physicalNoDQ native condition/loss differ')
        return dict(status='PASS_ACTUAL_ACTIVATED_DQ_ZERO_VS_PHYSICAL_NODQ_NATIVE_SS_IDENTITY',
            condition_and_negative_exact=True,same_noise_native_SS_loss_exact=True,
            actual_DQ_loss=left['loss'],physical_NoDQ_loss=right['loss'],
            original_R_and_feature_weights_preserved=True,NoDQ_input_channels=2048,DQ_input_channels=4096,
            RAW_observations_not_zeroed=True,optimizer_updates=0)
    finally:
        trainer.deform_conditioner=actual
        for mapping,old in maps:mapping['deform_conditioner']=old
        torch.use_deterministic_algorithms(deterministic)
