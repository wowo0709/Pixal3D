"""Physical selected/baseline compositions and genuine508 SS admission.

Selection and the original independent-seed gates belong to the closed DEV
freeze. This consumer never creates a substitute freeze or reinterprets an
architecture under another arm's name.
"""
from copy import deepcopy

from .native_final_freeze import validate_spec
from .native_population_contract import validate_capture
from .original_zero_population import MODES, SCHEMA as ZERO_SCHEMA
from .selected_pipeline_freeze_v2 import exact, validate_evaluation_protocol
from .selected_pipeline_joint_summary_v2 import compositions
from .selected_pipeline_model_contract import require, validate_ss_binding, validate_ss_evaluation, validate_pbr_binding


FREEZE_SCHEMA = 'selected-physical-pipeline-development-freeze-v2'
CONTROL_STATUS = 'FROZEN_ORIGINAL_ZERO_RGB_CONTROLS_BEFORE_TEST'


def validate_original_stages(original):
    require(set(original) == {'ss64','shape512','shape1024','pbr1024'}, 'all four original multiview stages required')
    for name, step in [('ss64',4210),('shape512',4210),('shape1024',16850),('pbr1024',16850)]:
        stage = original[name]
        require(type(stage['step']) is int and stage['step'] == step
            and stage['checkpoint']['path'].endswith(f'/{name}/ckpts/denoiser_step{step:07d}.pt'),
            'actual original stage family/checkpoint step required')
        for key in ['config','checkpoint']: validate_spec(stage[key])


def validate_original_models(models, original):
    """Retain an actually evaluated local copy only with the original byte hash."""
    validate_original_stages(original)
    require(set(models) == set(MODES), 'both original four-RGB controls required')
    config = None
    for alias, model in models.items():
        require(model.get('schema') == ZERO_SCHEMA and model.get('alias') == alias
            and model.get('conditioning_mode') == MODES[alias]
            and type(model.get('original_multiview_step')) is int and model['original_multiview_step'] == 4210
            and type(model.get('deformation_finetuning_updates')) is int and model['deformation_finetuning_updates'] == 0
            and model.get('conditioner_checkpoint') is None
            and model['checkpoint']['path'].endswith('/denoiser_step0004210.pt')
            and model['checkpoint']['SHA256'] == original['ss64']['checkpoint']['SHA256'],
            'original SS4210 base+LoRA with the same actual original bytes and zero deformation updates required')
        for key in ['native_config','checkpoint']: validate_spec(model[key])
        if config is None: config = model['native_config']
        require(exact(config, model['native_config']), 'the two zero controls must use the same actual SS loader config')
    require(exact(models['zero-source4']['checkpoint'], models['zero-edited4']['checkpoint']),
            'both zero controls must load the same actual original checkpoint copy')


def validate_parent(parent):
    require(parent.get('schema') == FREEZE_SCHEMA and parent.get('status') == 'FROZEN_FROM_DEVELOPMENT_ONLY'
        and parent.get('heldout_quality_read_before_freeze') is False,
        'actual selected-pipeline DEV-only freeze required before held-out execution')
    pair = parent['selected_pair']
    require(isinstance(pair, list) and len(pair) == 2, 'one explicit frozen measured SS/PBR pair required')
    pairs, selected = compositions(tuple(pair))
    require(set(parent['models']) == set(pairs) and parent['selected_alias'] == selected,
            'exact deduplicated frozen candidate and two strong controls required')
    selection = parent['selection']
    require(selection.get('status') == 'SELECTED_FROM_ACTUAL_DEVELOPMENT_ONLY'
        and selection.get('selected_SS_arm') == pair[0] and selection.get('selected_PBR_mode') == pair[1]
        and selection.get('heldout_quality_read') is False and selection.get('mandatory_all_modules') is False,
        'actual DEV model choice must match the final frozen composition')
    validate_evaluation_protocol(parent['evaluation_protocol'])
    roles = parent['development_evidence_roles']
    require(set(roles) == {'SS_core12','SS_followup','joint','SS_two_seeds','PBR_two_seeds','selection','cache_QA'}
        and exact(parent['development_evidence'], list(roles.values())), 'retain all original DEV evidence roles')
    for spec in roles.values(): validate_spec(spec)
    population = parent['population']; ids = population['sample_ids']; clusters = population['source_clusters_by_sample']
    require(len(ids) == len(set(ids)) == 508 and set(clusters) == set(ids)
        and len(set(clusters.values())) == 134 and population['rendered_cameras'] == [8,9,10]
        and set(population['generation_seeds']) == set(ids)
        and all(type(s) is int and 0 <= s < 2**31 for s in population['generation_seeds'].values()),
        'complete prescribed508/134-source population and original SS noise required')
    validate_spec(population['manifest'])
    original = parent['original_stages']
    validate_original_stages(original)
    own_ss = {}; own_pbr = {}
    for alias, (arm, mode) in pairs.items():
        model = parent['models'][alias]
        validate_ss_binding(model['ss64'], arm); validate_pbr_binding(model['pbr1024'], mode)
        require(all(exact(model[k], original[k]) for k in ['shape512','shape1024']), 'original frozen Shape weights/configs required')
        for key, value, seen in [(arm,model['ss64'],own_ss),(mode,model['pbr1024'],own_pbr)]:
            require(key not in seen or exact(value,seen[key]), 'same physical arm cannot silently use another saved model')
            seen[key] = value
    return pairs


def make_original_controls(parent, models):
    validate_parent(parent)
    validate_original_models(models, parent['original_stages'])
    return deepcopy(dict(status=CONTROL_STATUS, models=models, heldout_model_quality_read=False,
        downstream_original_stages={k:parent['original_stages'][k] for k in ['shape512','shape1024','pbr1024']},
        original_SS=parent['original_stages']['ss64'], parent_schema=FREEZE_SCHEMA,
        original_SS_copy_lineage={name:dict(actual_used=model['checkpoint'],
            original=parent['original_stages']['ss64']['checkpoint']) for name,model in models.items()},
        no_candidate_or_control_quality_used=True))


def compose_models(parent, controls):
    pairs = validate_parent(parent)
    expected = make_original_controls(parent, controls['models'])
    require(controls.get('status') == CONTROL_STATUS and controls.get('heldout_model_quality_read') is False
        and all(exact(controls.get(key), expected[key]) for key in ['downstream_original_stages',
            'original_SS','parent_schema','original_SS_copy_lineage','no_candidate_or_control_quality_used']),
        'two original controls and the same original downstream stages required')
    original = expected['downstream_original_stages']; result = {}
    for alias in MODES:
        result[alias] = dict(kind='original-zero', ss64=controls['models'][alias], pbr1024=original['pbr1024'],
            geometry_stages=deepcopy(original), geometry_group=alias)
    for alias, (arm, mode) in pairs.items():
        model = parent['models'][alias]
        result[alias] = dict(kind='selected-trained', ss64=model['ss64'], pbr1024=model['pbr1024'],
            SS_arm=arm, PBR_mode=mode, geometry_group=arm, geometry_stages=deepcopy(original))
    return deepcopy(result)


def admit_ss(alias, model, result, *, parent, controls, manifest_SHA256):
    models = compose_models(parent, controls)
    require(alias in models and exact(model, models[alias]), 'actual own frozen model composition required')
    pop = parent['population']; ids = pop['sample_ids']; binding = model['ss64']; zero = alias in MODES
    require(manifest_SHA256 == pop['manifest']['SHA256']
        and result.get('manifest_sha256') == manifest_SHA256
        and result.get('status') == 'PASS' and result.get('stage') == 'ss64'
        and result.get('population_scope') == 'final-test508' and result.get('diagnostic_only') is False
        and exact(result.get('frozen_native_model_binding'), binding)
        and result.get('checkpoint') == binding['checkpoint']['path']
        and result.get('conditioner_checkpoint') == (None if binding['conditioner_checkpoint'] is None
            else binding['conditioner_checkpoint']['path'])
        and type(result.get('step')) is int and result['step'] == (0 if zero else 480)
        and result.get('input_states') == (['M0']*4 if alias == 'zero-source4' else ['M1','M0','M0','M0'])
        and result.get('input_camera_ids') == [8,9,10,11] and result.get('target_modality') == 'rgb'
        and result.get('optimizer_updates_during_evaluation') == 0
        and result.get('generation_batch_size') == 8 and result.get('sampling_steps') == 12
        and result.get('guidance') == 3. and result.get('quick_panel_ids') == ids,
        'own genuine TEST508 saved model, RGB protocol, sampler and prescribed population required')
    if zero:
        require(result.get('original_multiview_SS_step') == 4210 and result.get('deformation_finetuning_updates') == 0
            and result.get('actual_original_full_base_and_LoRA_validation', {}).get(
                'original_full_base_and_LoRA_exact_before_evaluation') is True
            and result['actual_original_full_base_and_LoRA_validation'].get('actual_original_LoRA_tensors') == 360
            and result.get('deterministic_evaluation', {}).get('actual_deterministic_algorithms') is True,
            'actual complete original zero-shot SS weights must be checked before evaluation')
    else:
        validate_ss_evaluation(result, binding, model['SS_arm'])
    archive = result['native_SS_artifacts']
    require(archive.get('status') == ('PASS_ACTUAL_ORIGINAL_ZERO_SS_BOUND_POPULATION_ARCHIVE' if zero else
        'PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION_ARCHIVE') and type(archive.get('archived_asset_count')) is int
        and archive['archived_asset_count'] == 508 and archive.get('predicted_SS_is_available_for_full_cascade') is True
        and archive.get('GT_SS_is_for_scoring_only') is True,
        'all508 actual predicted SS archives required; no fabricated DEV-prefix status')
    validate_capture(result, archive['artifacts'], 508)
    require(all(r['source_cluster_id'] == pop['source_clusters_by_sample'][r['sample_id']]
        and r['seed'] == pop['generation_seeds'][r['sample_id']] for r in result['per_asset']),
        'same complete508 source identities and paired frozen noise required')
    return {r['sample_id']:r for r in archive['artifacts']}


def admit_all_ss(records, *, parent, controls):
    models = compose_models(parent, controls)
    require(set(records) == set(models), 'every distinct frozen composition and both zero controls required')
    reference = None; admitted = {}
    for alias, model in models.items():
        record = records[alias]
        admitted[alias] = admit_ss(alias, model, record, parent=parent, controls=controls,
            manifest_SHA256=parent['population']['manifest']['SHA256'])
        identities = [tuple(r[k] for k in ['sample_id','source_cluster_id','seed','target_voxels_sha256'])
            for r in record['per_asset']]
        if reference is None: reference = identities
        require(identities == reference, 'every model must score the same actual GT arrays/noise/source population')
    return admitted
