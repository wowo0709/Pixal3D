"""Same Direct delivery as the full arm, with no deformation-query module."""
import json
from pathlib import Path

from .anchor_joint_runtime import AnchorJointTrainer
from .ray_feature_joint_runtime import RayFeatureJointTrainer


class DirectNoQueryJointTrainer(RayFeatureJointTrainer):
    def __init__(self,*args,anchor_stage,**kwargs):
        module = kwargs['deform_conditioner']
        if (anchor_stage.get('variant')!='direct-physical-NoDQ'
                or module['arm']!='c1' or module.get('deformation_dim') is not None):
            raise ValueError('explicit Direct variant with query channels physically absent required')
        super().__init__(*args,anchor_stage=anchor_stage,**kwargs)

    def _verify_reference(self):
        super()._verify_reference()
        spec = self.anchor_stage
        proof = json.loads(Path(spec['feature_identity_proof']).read_text())
        if (proof.get('variant')!='direct-physical-NoDQ'
                or proof.get('retargeter_input_channels')!=2048
                or proof.get('physical_deformation_channels')!=0
                or proof.get('deformation_projection_present') is not False
                or proof.get('zero_deformation_observations_used') is not False
                or proof.get('direct_no_query_native_adapter') is not True):
            raise ValueError('actual Direct NoDQ proof required;input-side proof is insufficient')
        if not spec.get('diagnostic_profile',False):
            capacity = json.loads(Path(spec['capacity_proof']).read_text())
            if (capacity.get('variant')!='direct-physical-NoDQ'
                    or capacity.get('physical_deformation_channels')!=0
                    or capacity.get('deformation_projection_present') is not False):
                raise ValueError('actual Direct NoDQ dedicated capacity required')

    def _write_stage_initialization(self,restored,sampler):
        from pixal3d.modules.direct_ray_retargeting_no_query import DirectRayRetargetingNoQueryConditioner
        module = self.deform_conditioner
        if (not isinstance(module,DirectRayRetargetingNoQueryConditioner)
                or hasattr(module.retargeter,'deformation_projection')
                or module.retargeter.input_dim!=2048
                or any('deformation_projection' in name for name in module.state_dict())):
            raise ValueError('the actual constructed module must omit query channels and projection')
        AnchorJointTrainer._write_stage_initialization(self,restored,sampler)
        path = Path(self.output_dir)/f'NoDQ-stage-initialization.rank{self.rank}.json'
        with path.open('x') as stream:
            json.dump(dict(variant='direct-physical-NoDQ',actual_class=type(module).__name__,
                physical_deformation_channels=0,retargeter_input_channels=module.retargeter.input_dim,
                deformation_projection_present=False,zero_deformation_observations_used=False,
                direct_feature_delivery=True,common_Adam_states=restored,new_feature_tensors=9,
                reference_step=96,final_step=480,remaining_updates=384),stream,indent=2)
            stream.write('\n')
