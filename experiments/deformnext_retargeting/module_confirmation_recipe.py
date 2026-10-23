"""Independent-seed preparation for the finite module study, without selection.

Only the eventual measured candidate needs a new training run. Vanilla and
Top1+R confirmations already prepared elsewhere can be reused. This factory
does not queue twelve confirmations or authorize a held-out test forward.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import projected_native_recipe as projected
from . import raw_correspondence_native_recipe as correspondence
from . import r_mediated_native_recipe as native
from . import r_top1_native_recipe as top1
from . import unified_design_native_recipe as design
from .module_pipeline_protocol_v2 import protocol


ALIASES = {'V1': 'vanilla', 'C_DECODER': 'R_top1_only', 'FULL': 'R_raw_topk'}
PHASES = ('proof', 'capacity', 'science')
RUN_FIELDS = ('training_seed', 'output_dir', 'wandb_id', 'wandb_name', 'name', 'use_wandb')
REFERENCE_FIELDS = ('reference_science_config', 'reference_science_config_SHA256')


def _exact(a, b):
    # Python equality alone accepts True == 1 and 96.0 == 96.
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def _reference(cfg):
    e = cfg['experiment']
    path = Path(e['reference_science_config'])
    if not path.is_absolute():
        raise ValueError('absolute pinned reference recipe required')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != e['reference_science_config_SHA256']:
        raise ValueError('reference recipe bytes changed')
    return json.loads(data)


def validate_arm(cfg, arm, phase):
    if arm not in protocol()['arms'] or phase not in PHASES:
        raise ValueError('one of the twelve prespecified arms and an explicit phase required')
    if arm in ('V1', 'FULL'):
        actual = native.validate_config(cfg, phase)
    elif arm == 'C_DECODER':
        actual = top1.validate_config(cfg, phase, _reference(cfg))
    elif arm in projected.VARIANTS:
        actual = projected.validate_config(cfg, phase, _reference(cfg))
    elif arm in correspondence.VARIANTS:
        actual = correspondence.validate_config(cfg, phase, _reference(cfg))
    else:
        actual = design.validate_config(cfg, phase, _reference(cfg))
    if actual != ALIASES.get(arm, arm):
        raise ValueError('physical runtime variant differs from the selected arm')
    return actual


def _run(original, *, phase, output_dir, run_id, recipe_path):
    if phase not in PHASES or not isinstance(run_id, str) or not run_id:
        raise ValueError('explicit phase and unique run identity required')
    if (not Path(output_dir).is_absolute() or not Path(recipe_path).is_absolute()
            or str(output_dir) == original.get('output_dir')
            or str(recipe_path) == original['experiment'].get('recipe_path')):
        raise ValueError('new absolute output and recipe paths required; preserve the seed101 artifacts')
    value = deepcopy(original)
    value.update(training_seed=202, output_dir=str(output_dir), wandb_id=run_id,
                 wandb_name=run_id, use_wandb=phase == 'science')
    if 'name' in value:
        value['name'] = run_id
    value['experiment'].update(engineering_only=phase != 'science', recipe_path=str(recipe_path))
    return value


def _validate_changes(original, confirmation):
    normalized = deepcopy(confirmation)
    for field in RUN_FIELDS:
        if field in original:
            normalized[field] = deepcopy(original[field])
        else:
            normalized.pop(field, None)
    for field in ('engineering_only', 'recipe_path', *REFERENCE_FIELDS):
        if field in original['experiment']:
            normalized['experiment'][field] = deepcopy(original['experiment'][field])
        else:
            normalized['experiment'].pop(field, None)
    if not _exact(normalized, original):
        raise ValueError('confirmation changed a model, data, optimizer, initialization or numerical field')
    if type(confirmation['training_seed']) is not int or confirmation['training_seed'] != 202:
        raise ValueError('an independent integer training seed202 required')


def make_reference(original, *, output_dir, run_id, recipe_path):
    """Recipe-only seed202 reference; never a trained or evaluated model."""
    if (type(original['training_seed']) is not int or original['training_seed'] != 101
            or native.validate_config(original, 'science') not in ('R_feature', 'R_raw_topk')):
        raise ValueError('actual seed101 native reference recipe required')
    value = _run(original, phase='science', output_dir=output_dir, run_id=run_id, recipe_path=recipe_path)
    _validate_changes(original, value)
    native.validate_config(value, 'science')
    return value


def validate_confirmation(original, confirmation, *, arm, phase):
    if type(original['training_seed']) is not int or original['training_seed'] != 101:
        raise ValueError('actual seed101 scientific recipe required')
    validate_arm(original, arm, 'science')
    validate_arm(confirmation, arm, phase)
    _validate_changes(original, confirmation)
    if arm not in ('V1', 'FULL'):
        old_ref, new_ref = _reference(original), _reference(confirmation)
        if old_ref['training_seed'] != 101:
            raise ValueError('original control must bind its actual seed101 reference')
        _validate_changes(old_ref, new_ref)
        if native.validate_config(new_ref, 'science') != native.validate_config(old_ref, 'science'):
            raise ValueError('confirmation reference family changed')
    return dict(status='PASS_SELECTED_MODULE_SEED202_RECIPE_ONLY', arm=arm,
                actual_variant=ALIASES.get(arm, arm), training_seed=202,
                all_other_scientific_values_and_types_exact=True,
                original_starting_multiview_checkpoint_unchanged=True,
                fresh_optimizer_required=True, actual_GPU_parity_capacity_still_required=True,
                candidate_selected=False, new_training_queued=False,
                held_out_quality_read=False, full_goal_complete=False)


def make_confirmation(original, *, arm, phase, output_dir, run_id, recipe_path,
                      reference_path=None, reference_SHA256=None):
    validate_arm(original, arm, 'science')
    value = _run(original, phase=phase, output_dir=output_dir, run_id=run_id, recipe_path=recipe_path)
    needs_reference = arm not in ('V1', 'FULL')
    if needs_reference:
        if reference_path is None or reference_SHA256 is None:
            raise ValueError('own physically compatible seed202 reference recipe required')
        value['experiment'].update(reference_science_config=str(reference_path),
                                   reference_science_config_SHA256=reference_SHA256)
    elif reference_path is not None or reference_SHA256 is not None:
        raise ValueError('native vanilla/full has no control-reference binding')
    validate_confirmation(original, value, arm=arm, phase=phase)
    return value
