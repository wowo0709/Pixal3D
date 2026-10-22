"""Prepare an independent native seed without changing the SS learning recipe."""
import copy

from .r_mediated_native_recipe import common_contract,validate_config as validate_native
from .r_top1_native_recipe import validate_config as validate_top1


def seed202_reference(original, *, output_dir, run_id):
    if original['training_seed']!=101 or validate_native(original,'science')!='R_feature':
        raise ValueError('the actual seed101 R19 reference recipe is required')
    value=copy.deepcopy(original)
    value.update(training_seed=202,output_dir=str(output_dir),wandb_id=run_id,wandb_name=run_id)
    validate_native(value,'science')
    return value


def seed202_recipe(original, *, variant, phase, output_dir, run_id, reference=None,
                   reference_path=None, reference_SHA256=None):
    if original['training_seed']!=101 or phase not in {'proof','capacity','science'}:
        raise ValueError('explicit seed101 original and native phase required')
    if variant=='vanilla':
        if validate_native(original,'science')!='vanilla':raise ValueError('own native vanilla original required')
    elif variant=='R_top1_only':
        # Read the original pinned reference before changing any field.
        if validate_top1(original,'science')!=variant:raise ValueError('own physical Top1 original required')
        if reference is None or reference['training_seed']!=202 or not reference_path or not reference_SHA256:
            raise ValueError('explicit new seed202 R19 recipe-only comparison reference required')
    else:raise ValueError('confirmation compares vanilla and physical Top1+R only')
    value=copy.deepcopy(original)
    value.update(training_seed=202,output_dir=str(output_dir),wandb_id=run_id,wandb_name=run_id,use_wandb=phase=='science')
    value['experiment']['engineering_only']=phase!='science'
    if variant=='R_top1_only':
        value['experiment'].update(reference_science_config=str(reference_path),reference_science_config_SHA256=reference_SHA256)
        validate_top1(value,phase,reference)
    else:validate_native(value,phase)
    return value


def validate_pair(vanilla,top1,reference,*,phase):
    if (vanilla['training_seed']!=202 or top1['training_seed']!=202 or reference['training_seed']!=202
            or validate_native(vanilla,phase)!='vanilla'
            or validate_top1(top1,phase,reference)!='R_top1_only'
            or common_contract(vanilla)!=common_contract(top1)):
        raise ValueError('matched native seed202 pair required')
    return dict(training_seed=202,variants=['vanilla','R_top1_only'],training_assets=9149,
        global_batch=96,microbatch=96,accumulation=1,world_size=1,updates=480,epochs=5,
        R19_reference_is_recipe_only_not_a_trained_model=True,
        actual_GPU_proof_capacity_and_seed202_quality_not_proven=True)
