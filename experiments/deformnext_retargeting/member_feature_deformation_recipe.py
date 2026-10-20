"""One additive physical-DQ arm, matched to the sealed No-DQ source-ray arm."""
import copy

from .member_feature_recipe import NAME as NODQ,SCHEMA as NODQ_SCHEMA,ARM as NODQ_ARM,validate_feature_recipe

NAME='MemberFeatureDeformationSourceRayImageConditionedProjFlowMatchingCFGTrainer'
DATASET='DeformNeXtMemberRawSS64'
SCHEMA='member-feature-deformation-source-ray-v1'
ARM='feature-direct-dq'


def validate_feature_deformation_recipe(cfg):
    experiment,spec=cfg['experiment'],cfg['trainer']['args']
    if (cfg['trainer']['name']!=NAME or cfg['dataset']['name']!=DATASET
            or cfg['validation']['dataset_name']!=DATASET
            or experiment.get('conditioning_schema')!=SCHEMA
            or experiment.get('readout_arm')!=ARM
            or experiment.get('physical_deformation_query_channels')!=2048
            or spec['deform_conditioner'].get('deformation_dim')!=2048):
        raise ValueError('explicit matched source-ray four-head physical-DQ arm required')
    for args in [cfg['dataset']['args'],cfg['validation']['dataset_args']]:
        if (args.get('deformation_query_channels')!=2048 or args.get('match_root') is not None
                or args.get('deformation_root') is not None or args.get('task_matching_features',False) is not False
                or args.get('transport_arm')!='c1' or args.get('transport_implementation')!='r9-native-v5'):
            raise ValueError('RAW-only2048 RGB/Omega context without old match/deformation sidecars required')
    adapted=copy.deepcopy(cfg)
    adapted['trainer']['name']=NODQ;adapted['dataset']['name']='DeformNeXtSS64'
    adapted['validation']['dataset_name']='DeformNeXtSS64'
    adapted['trainer']['args']['deform_conditioner'].pop('deformation_dim')
    adapted['experiment'].update(conditioning_schema=NODQ_SCHEMA,readout_arm=NODQ_ARM,physical_deformation_query_channels=0)
    for args in [adapted['dataset']['args'],adapted['validation']['dataset_args']]:
        args.pop('deformation_query_channels')
    validate_feature_recipe(adapted)
    return ARM
