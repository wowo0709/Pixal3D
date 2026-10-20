"""Two matched controls, retaining the sealed native SS optimization recipe."""
import copy

from .member_feature_recipe import NAME as FULL_NAME, SCHEMA as FULL_SCHEMA, ARM as FULL_ARM, READOUT as FULL_READOUT, validate_feature_recipe
from .member_posterior_runtime import PROPOSAL

NAME = 'MemberFeatureControlImageConditionedProjFlowMatchingCFGTrainer'
SCHEMA = 'member-feature-controls-v1'
CONTROLS = {
    'top1-feature': dict(arm='feature-top1-only', source_tensors=12, head_tensors=2, candidates=64,
                         correspondence='semantic', proposal=PROPOSAL,
                         readout='Physical semantic Top1 feature plus identical V/output and R;no Q/K/score parameters'),
    'spatial-feature-attention': dict(arm='feature-spatial-only', source_tensors=19, head_tensors=9, candidates=3,
                         correspondence='native-spatial',
                         proposal='Same distinct observed target-patch members;original native camera-projected spatial3 source keys/values,D1 member contexts and score averaging;no semantic retrieval',
                         readout='Original projected spatial3 features;identical four-head QKV/score/R with distinct member queries'),
}


def control_spec(cfg):
    key = cfg['trainer']['args'].get('feature_control')
    if key not in CONTROLS:
        raise ValueError('one explicit physical Top1 or original spatial feature control required')
    return dict(CONTROLS[key], control=key)


def validate_control_recipe(cfg):
    spec = control_spec(cfg); experiment = cfg['experiment']; args = cfg['trainer']['args']
    if (cfg['trainer']['name'] != NAME or experiment.get('conditioning_schema') != SCHEMA
            or experiment.get('readout_arm') != spec['arm'] or experiment.get('feature_readout') != spec['readout']
            or experiment.get('matching_proposals') != spec['proposal']
            or args['deform_conditioner'].get('correspondence_mode', 'semantic') != spec['correspondence']):
        raise ValueError('physical control identity, source features or same-head contract changed')
    adapted = copy.deepcopy(cfg)
    adapted['trainer']['name'] = FULL_NAME
    adapted['trainer']['args'].pop('feature_control')
    adapted['trainer']['args']['deform_conditioner']['correspondence_mode'] = 'semantic'
    adapted['experiment'].update(conditioning_schema=FULL_SCHEMA, readout_arm=FULL_ARM, feature_readout=FULL_READOUT, matching_proposals=PROPOSAL)
    validate_feature_recipe(adapted)
    return spec['arm']
