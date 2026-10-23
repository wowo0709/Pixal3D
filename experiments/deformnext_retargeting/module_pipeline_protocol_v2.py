"""Separate D1 context decoding from its scalar score instead of combining them."""
from copy import deepcopy

from .module_pipeline_protocol import protocol as original_protocol


def protocol():
    value=original_protocol()
    value.update(schema='unified-module-pipeline-v2',extends_schema='unified-module-pipeline-v1')
    value['arms']['C_D1_COS']=dict(value['arms']['C_DECODER'],matching='D1-cosine')
    for contrast in value['module_contrasts']:
        if contrast['effect']=='matching_decoder':contrast['candidate']='C_D1_COS'
    value['module_contrasts'].insert(2,dict(effect='matching_scalar_score',candidate='C_DECODER',
        reference='C_D1_COS',changed_factor='matching'))
    value['evaluation']['module_family_size']=10
    value['implementation_requirements'].update(D1_cosine_has_decoder_but_no_scalar_score_head=True,
        decoder_and_scalar_score_not_claimed_as_one_isolated_effect=True)
    validate_protocol(value)
    return value


def validate_protocol(value):
    if len(value['arms'])!=12 or len(value['module_contrasts'])!=10:
        raise ValueError('twelve focused arms and ten single-factor comparisons required')
    for contrast in value['module_contrasts']:
        a,b=[value['arms'][contrast[k]] for k in ['candidate','reference']]
        changed={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        if changed!={contrast['changed_factor']}:raise ValueError('confounded module contrast')
    if value['evaluation']['module_family_size']!=10:
        raise ValueError('keep the complete planned module family for final inference')
    return deepcopy(value)
