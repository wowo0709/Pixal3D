"""Independent native-PBR training seed with the exact measured science recipe.

The seed101 engineering proof remains a fixed architectural/capacity reference,
as already supported by validate_science. It is not called a new seed202 proof.
Every seed202 science run still starts fresh and must pass its own original
input/noise/latent initial4 identity before the first optimizer update.
"""
from copy import deepcopy
import json
from pathlib import Path

from .r_mediated_pbr_science_recipe import validate_science


RUN_FIELDS = ('training_seed', 'output_dir', 'wandb_id', 'wandb_name', 'name',
              'master_port', 'config')


def validate_confirmation(original, confirmation, *, proof, original_multiview):
    mode = validate_science(original, proof, original_multiview)
    confirmed_mode = validate_science(confirmation, proof, original_multiview)
    if (type(original['training_seed']) is not int or original['training_seed'] != 101
            or type(confirmation['training_seed']) is not int or confirmation['training_seed'] != 202
            or confirmed_mode != mode):
        raise ValueError('actual same-mode seed101 to independent integer seed202 required')
    normalized = deepcopy(confirmation)
    for field in RUN_FIELDS:
        if field in original:
            normalized[field] = deepcopy(original[field])
        else:
            normalized.pop(field, None)
    if json.dumps(normalized, sort_keys=True, allow_nan=False) != json.dumps(original, sort_keys=True, allow_nan=False):
        raise ValueError('PBR confirmation changed scientific values or JSON types beyond the training seed')
    if (confirmation['output_dir'] == original['output_dir']
            or not Path(confirmation['output_dir']).is_absolute()
            or confirmation['wandb_id'] == original['wandb_id']):
        raise ValueError('separate fresh output and logging identity required; preserve the seed101 run')
    port = confirmation.get('master_port')
    if (not isinstance(port, str) or not port.isdigit() or not 1024 <= int(port) <= 65535
            or not isinstance(confirmation.get('config'), str)
            or not Path(confirmation['config']).is_absolute()):
        raise ValueError('actual absolute recipe and valid DDP port required')
    return dict(status='PASS_SELECTED_PBR_SEED202_SCIENCE_RECIPE_ONLY', mode=mode,
        training_seed=202, epochs=5, updates=1900, training_assets=9139,
        global_batch=24, world_size=confirmation['num_gpus'],
        microbatch=confirmation['trainer']['args']['batch_size_per_gpu'], accumulation=1,
        original_PBR16850_full_base_and_LoRA_initializer_unchanged=True,
        original_scientific_values_and_types_exact_except_independent_seed=True,
        seed101_engineering_architecture_capacity_reference_reused=True,
        actual_seed202_engineering_proof_or_capacity_not_claimed=True,
        actual_seed202_initial4_input_noise_latent_identity_before_updates_required=True,
        actual_seed202_training_and_quality_not_yet_proven=True,
        fresh_optimizer_required=True, new_training_queued=False,
        model_selected=False, full_goal_complete=False)


def make_confirmation(original, *, proof, original_multiview, output_dir, run_id,
                      recipe_path, master_port):
    if (not isinstance(run_id, str) or not run_id or not Path(output_dir).is_absolute()
            or not Path(recipe_path).is_absolute() or type(master_port) is not int
            or not 1024 <= master_port <= 65535):
        raise ValueError('explicit new run identity, absolute paths and integer DDP port required')
    value = deepcopy(original)
    value.update(training_seed=202, output_dir=str(output_dir), wandb_id=run_id,
                 wandb_name=run_id, config=str(recipe_path), master_port=str(master_port))
    if 'name' in value:
        value['name'] = run_id
    validate_confirmation(original, value, proof=proof, original_multiview=original_multiview)
    return value
