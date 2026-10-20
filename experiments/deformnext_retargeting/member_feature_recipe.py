"""One additive NoDQ feature arm under the frozen source-ray SS protocol."""
import copy
import re

from .member_posterior_runtime import NAME as MEMBER, SCHEMA as MEMBER_SCHEMA, validate_member_recipe

NAME = 'MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer'
SCHEMA = 'member-feature-source-ray-v1'
ARM = 'feature-direct-nodq'
READOUT = 'Historical RayFeatureReadout4x64 QKV+zero-output; bounded direct delivery outside semantic gate'


def validate_feature_recipe(cfg):
    experiment, spec = cfg['experiment'], cfg['trainer']['args']
    utility = spec.get('ray_feature_utility', {})
    if (cfg['trainer']['name'] != NAME
            or experiment.get('conditioning_schema') != SCHEMA
            or experiment.get('readout_arm') != ARM
            or experiment.get('feature_readout') != READOUT
            or set(utility) != {'checkpoint', 'SHA256', 'completed'}
            or not all(isinstance(utility[key], str) and utility[key] for key in utility)
            or re.fullmatch('[0-9a-f]{64}', utility['SHA256']) is None
            or spec['deform_conditioner'].get('task_attention') is not None
            or spec['deform_conditioner'].get('ray_mode') != 'semantic-top1'
            or spec['deform_conditioner'].get('arm') != 'c1'
            or 'deformation_dim' in spec['deform_conditioner']):
        raise ValueError('one explicit actual QKV/direct NoDQ arm and complete utility binding required')
    # Validate all original native optimization/data/LoRA/evaluation fields on
    # a copy. Only the constructor/readout identity differs from Top1+R.
    adapted = copy.deepcopy(cfg)
    adapted['trainer']['name'] = MEMBER
    adapted['experiment'].update(conditioning_schema=MEMBER_SCHEMA, readout_arm='top1')
    validate_member_recipe(adapted)
    return ARM
