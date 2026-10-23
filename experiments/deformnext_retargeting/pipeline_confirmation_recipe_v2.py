"""Independent seed recipes for the real physical candidate, including follow-ups.

Only recorded artifact paths differ in addition to the ordinary seed/run fields.
No architecture, optimizer, data or initial-value field is normalized away.
"""
from copy import deepcopy
from pathlib import Path

from . import module_confirmation_recipe as original
from .consistent_evidence_native_recipe import VARIANTS as FOLLOWUPS, validate_config
from .pipeline_population_contract_v2 import ARMS


ARTIFACT_FIELDS=('paired_reference_native_root','cold_proof_json','micro96_capacity_json')
ALIASES=original.ALIASES
PHASES=original.PHASES


def validate_arm(cfg,arm,phase):
    if arm not in FOLLOWUPS:
        return original.validate_arm(cfg,arm,phase)
    if phase not in PHASES or validate_config(cfg,phase,original._reference(cfg))!=arm:
        raise ValueError('actual same-family shared-evidence phase required')
    return arm


def _paths(confirmation):
    values=[confirmation['experiment'].get(k) for k in ARTIFACT_FIELDS]
    if any(not isinstance(p,str) or not Path(p).is_absolute() for p in values):
        raise ValueError('explicit absolute own seed202 proof/capacity and paired V1 root required')
    return values


def validate_confirmation(seed101, confirmation, *, arm, phase):
    if arm not in FOLLOWUPS:
        return original.validate_confirmation(seed101,confirmation,arm=arm,phase=phase)
    if type(seed101['training_seed']) is not int or seed101['training_seed']!=101:
        raise ValueError('actual seed101 original scientific config required')
    validate_arm(seed101,arm,'science');validate_arm(confirmation,arm,phase)
    _paths(confirmation)
    normalized=deepcopy(confirmation)
    for field in ARTIFACT_FIELDS:
        if field in seed101['experiment']:
            normalized['experiment'][field]=seed101['experiment'][field]
        else:
            normalized['experiment'].pop(field,None)
    original._validate_changes(seed101,normalized)
    old_ref,new_ref=original._reference(seed101),original._reference(confirmation)
    if old_ref['training_seed']!=101:
        raise ValueError('actual seed101 numerical reference required')
    original._validate_changes(old_ref,new_ref)
    if original.native.validate_config(new_ref,'science')!=original.native.validate_config(old_ref,'science'):
        raise ValueError('reference physical family changed')
    if any(confirmation['experiment'].get(k)==seed101['experiment'].get(k) for k in ARTIFACT_FIELDS):
        raise ValueError('seed202 must use its own paired V1 root, proof and capacity paths')
    return dict(status='PASS_SHARED_EVIDENCE_SEED202_RECIPE_ONLY',arm=arm,actual_variant=arm,
        training_seed=202,all_other_scientific_values_and_types_exact=True,
        operational_artifact_paths_recorded=list(ARTIFACT_FIELDS),
        original_starting_multiview_checkpoint_unchanged=True,fresh_optimizer_required=True,
        actual_GPU_parity_capacity_still_required=True,candidate_selected=False,
        new_training_queued=False,held_out_quality_read=False,full_goal_complete=False)


def make_confirmation(seed101, *, arm, phase, output_dir, run_id, recipe_path,
                      reference_path=None, reference_SHA256=None,
                      paired_reference_native_root=None,cold_proof_json=None,micro96_capacity_json=None):
    if arm not in FOLLOWUPS:
        if any(p is not None for p in (paired_reference_native_root,cold_proof_json,micro96_capacity_json)):
            raise ValueError('extra shared-evidence artifact fields do not belong to a legacy family')
        return original.make_confirmation(seed101,arm=arm,phase=phase,output_dir=output_dir,
            run_id=run_id,recipe_path=recipe_path,reference_path=reference_path,
            reference_SHA256=reference_SHA256)
    validate_arm(seed101,arm,'science')
    if reference_path is None or reference_SHA256 is None:
        raise ValueError('own seed202 numerical reference recipe required')
    value=original._run(seed101,phase=phase,output_dir=output_dir,run_id=run_id,recipe_path=recipe_path)
    value['experiment'].update(reference_science_config=str(reference_path),
        reference_science_config_SHA256=reference_SHA256,
        paired_reference_native_root=str(paired_reference_native_root) if paired_reference_native_root is not None else None,
        cold_proof_json=str(cold_proof_json) if cold_proof_json is not None else None,
        micro96_capacity_json=str(micro96_capacity_json) if micro96_capacity_json is not None else None)
    validate_confirmation(seed101,value,arm=arm,phase=phase)
    return value


def models_for_selected(selected):
    if selected not in ARMS:
        raise ValueError('an actually measured physical candidate required')
    return tuple(dict.fromkeys(('V1','C_DECODER',selected)))
