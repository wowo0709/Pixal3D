"""Change only the explicitly authorized execution devices of a native recipe.

No model, optimizer, dataset, noise seed, batch or runtime math may differ.
Physical devices are supplied by the closed session policy, not this module.
"""
from copy import deepcopy


EXECUTION_FIELDS = ('assigned_GPUs','CPU_affinity_per_GPU','recipe_path')


def validate_execution_metadata(original, execution, assignment):
    if set(assignment)!={'authorized_GPUs',*EXECUTION_FIELDS}:
        raise ValueError('explicit complete execution assignment required')
    devices=assignment['assigned_GPUs'];authorized=assignment['authorized_GPUs']
    if (not isinstance(devices,list) or not devices or len(set(devices))!=len(devices)
            or not isinstance(authorized,list) or not set(devices)<=set(authorized)
            or any(not isinstance(x,str) or not x.isdigit() for x in devices+authorized)):
        raise ValueError('distinct explicitly authorized physical devices required')
    affinity=assignment['CPU_affinity_per_GPU']
    if (not isinstance(affinity,dict) or set(affinity)!=set(devices)
            or any(not isinstance(v,list) or not v or len(set(v))!=len(v)
                   or any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in v)
                   for v in affinity.values())
            or not isinstance(assignment['recipe_path'],str) or not assignment['recipe_path']):
        raise ValueError('declared CPU affinity for every device and actual recipe path required')
    if (not set(EXECUTION_FIELDS)<=original['experiment'].keys()
            or any(execution['experiment'].get(k)!=assignment[k] for k in EXECUTION_FIELDS)):
        raise ValueError('actual recipe must equal the closed execution assignment')
    normalized=deepcopy(execution)
    for field in EXECUTION_FIELDS:normalized['experiment'][field]=deepcopy(original['experiment'][field])
    if normalized!=original:
        raise ValueError('non-execution model/training/data/provenance field changed')
    return dict(status='PASS_ONLY_NATIVE_EXECUTION_METADATA_CHANGED',
        changed_fields=[k for k in EXECUTION_FIELDS if original['experiment'][k]!=execution['experiment'][k]],
        original_assigned_GPUs=original['experiment']['assigned_GPUs'],
        execution_assigned_GPUs=devices,
        all_other_configuration_exact=True,
        native_GPU_initial_identity_and_capacity_still_required=True,
        quality_gain_proven=False,full_goal_complete=False)
