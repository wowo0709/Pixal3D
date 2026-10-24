"""Own DEV128 cascade identities for the five final Fixed-context controls.

Only metadata is constructed here. The native Shape, texture replay, renderer
and perceptual entries are unchanged. The two historical baselines are reused
as baselines; they can never stand in for a newly trained control.
"""
from copy import deepcopy
from pathlib import Path

from . import fixed_correspondence_pipeline_contract as matching
from . import shared_ablation_pipeline_contract as delivery
from .cascade_view_roles import summarize_view_roles
from .dev128_pipeline_contract import validate_generation, validate_population
from .selected_pipeline_model_contract import require, validate_pbr_binding


SCHEMA = 'final-Fixed-context-own-DEV128-cascade-v1'
ARMS = ('M_FIXED_SPATIAL', 'M_FIXED_RAW', 'M_FIXED_D1_COS',
        'A_FIXED_UNIFORM', 'A_FIXED_NO_R')
BASELINES = ('C_DECODER', 'E_FIXED_RAW')
CONTRASTS = (
    ('M_FIXED_RAW', 'M_FIXED_SPATIAL'),
    ('M_FIXED_D1_COS', 'M_FIXED_RAW'),
    ('E_FIXED_RAW', 'M_FIXED_D1_COS'),
    ('E_FIXED_RAW', 'C_DECODER'),
    ('E_FIXED_RAW', 'A_FIXED_UNIFORM'),
    ('E_FIXED_RAW', 'A_FIXED_NO_R'),
)


def own_contract(arm):
    require(arm in ARMS, 'one of the five required final-context controls required')
    return matching if arm.startswith('M_') else delivery


def validate_models(bindings):
    require(len(bindings) == 1, 'one own control per evaluation campaign required')
    arm = next(iter(bindings)); contract = own_contract(arm)
    model = bindings[arm]
    contract.validate_ss_binding(model['ss64'], arm, seed=101)
    validate_pbr_binding(model['pbr1024'], 'top1-only', seed=101)
    return arm


def validate_archive128(result, arm, reference, *, binding):
    return own_contract(arm).validate_archive128(result, arm, reference, binding=binding)


def validate_index(index):
    require(index.get('schema') == SCHEMA and index.get('TEST') is False,
            'explicit own development campaign required; no TEST execution')
    arm = validate_models(index['bindings'])
    ids, clusters = index['sample_ids'], index['source_clusters_by_sample']
    require(len(ids) == len(set(ids)) == 128 and set(clusters) == set(ids)
            and len(set(clusters.values())) == 93,
            'complete existing128/93-source panel required')
    require(set(index['models']) == {arm} and set(index['SS_archives']) == {arm},
            'a control cannot borrow another control archive or case population')
    rows = index['models'][arm]
    require([r['sample_id'] for r in rows] == ids
            and all(r['source_cluster_id'] == clusters[r['sample_id']] for r in rows),
            'all128 own cases in exact paired order required')
    return arm


def case_plans(*, arm, model, asset, evaluation, artifact, terminal,
               generation_seed, geometry_template, replay_template, replay_common_case,
               render_template, sources, output, reference_root,
               reference_render_root, devices):
    """Substitute only the own sample/archive and explicit artifact locations."""
    validate_models({arm: model})
    row = asset['row']; sid = asset['sample_id']; root = Path(output)
    require(row['sample_id'] == sid and row['split'] == 'validation'
            and row['source_cluster_id'] == asset['source_cluster_id']
            and row['conditioning_camera_ids'] == [8, 9, 10, 11]
            and row['target_anchor_kind'] == 'rgb'
            and row['target_anchor'] == row['target_rgb_input']
            and len(asset['RGB_paths']) == len(set(asset['RGB_paths'])) == 4,
            'exact edited RGB target and original three sources in validation required')
    require(artifact['sample_id'] == sid and artifact['seed'] == generation_seed
            and type(generation_seed) is int,
            'own captured SS and the same explicit generation noise required')
    require(bool(devices) and len(set(devices)) == len(devices)
            and all(isinstance(d, str) and d.isdigit() for d in devices),
            'explicit physical GPU assignment required')
    require(all(Path(p).is_absolute() for p in (output, reference_root, reference_render_root)),
            'explicit absolute output and existing reference roots required')
    for template in (geometry_template, replay_template, render_template):
        require(all(sources.get(p) == d for p,d in template.get('source_SHA256', {}).items()),
                'all original numerical sources must retain their actual hashes')
    pbr = model['pbr1024']
    require(replay_template['mode'] == 'top1-only'
            and replay_template['checkpoint_kind'] == 'trained-native1900'
            and replay_template['engineering_only'] is False,
            'actual trained Top-1 R replay required')
    for key, field in (('native_config','PBR_config'), ('checkpoint','PBR_checkpoint'),
                       ('conditioner_checkpoint','conditioner_checkpoint')):
        spec = pbr[key]
        require(replay_template[field] == spec['path']
                and spec['path'] in replay_template['closed_inputs'],
                'same actual common PBR paths and canonical receipts required')
        if spec['path'] in replay_template['input_SHA256']:
            require(replay_template['input_SHA256'][spec['path']] == spec['SHA256'],
                    'existing PBR hashes must match the actual model binding')

    data = {v['path']: v['SHA256'] for v in asset['inputs'].values()}
    gp = deepcopy(geometry_template)
    old_files = set(gp['RGB_paths']) | {gp['source_geometry'], gp['target_vertices']}
    old_SS = Path(gp['SS_evaluation']).parent
    def static_geometry(mapping):
        return {p:v for p,v in mapping.items() if p not in old_files
                and not Path(p).is_relative_to(old_SS)}
    gp.update(output=str(root/'geometry'), row=deepcopy(row),
        SS_evaluation=evaluation['path'], RGB_paths=deepcopy(asset['RGB_paths']),
        source_geometry=asset['inputs']['source_geometry']['path'],
        target_vertices=asset['inputs']['target_vertices']['path'], seed=generation_seed,
        source_SHA256=deepcopy(sources), assigned_GPUs=list(devices),
        inputs_SHA256={**static_geometry(gp['inputs_SHA256']), **data,
            evaluation['path']: evaluation['SHA256'], artifact['path']: artifact['SHA256']},
        closed_inputs={**static_geometry(gp.get('closed_inputs', {})),
            evaluation['path']: evaluation['receipt'], artifact['path']: artifact['receipt']})

    rp = deepcopy(replay_template); old_geometry = Path(rp['generated_shape_root'])
    closed = {p:r for p,r in rp['closed_inputs'].items()
        if not Path(p).is_relative_to(old_geometry)
        and p != str(Path(replay_common_case)/'audit.json')}
    hashes = {p:d for p,d in rp['input_SHA256'].items() if p in closed}
    rp.update(name=arm+'-'+sid+'-PBR', row={k:deepcopy(row[k]) for k in rp['row']},
        generated_shape_root=str(root/'geometry'), native_guides=str(root/'geometry/native-shape-guides.pt'),
        output=str(root/'PBR'), RGB_paths=deepcopy(asset['RGB_paths']),
        source_SHA256=deepcopy(sources), assigned_GPUs=list(devices),
        input_SHA256={**hashes, **data, **{pbr[k]['path']:pbr[k]['SHA256']
            for k in ('native_config','checkpoint','conditioner_checkpoint')}}, closed_inputs=closed)
    for filename in ('generation.json','prediction.npz','geometry-evaluation.npz','native-shape-guides.pt'):
        rp['closed_inputs'][str(root/'geometry'/filename)] = str(root/'geometry/file4-closure.json')
    render = deepcopy(render_template)
    render.update(generation_root=str(root/'PBR'), output=str(root/'render'), row=deepcopy(row),
        views=deepcopy(asset['render_views']), source_SHA256=deepcopy(sources),
        input_SHA256={render['environment_map']: render_template['input_SHA256'][render['environment_map']],
            **{v['GT_RGBA']:data[v['GT_RGBA']] for v in asset['render_views']}})
    shared = dict(output=str(root/'shared'), sample_id=sid, views=deepcopy(asset['render_views']),
        prediction_render_root=str(root/'render'), fixed_GT_shared_root=str(reference_root),
        fixed_GT_render_root=str(reference_render_root), source_SHA256=deepcopy(sources))
    return dict(geometry=gp, replay=rp, render=render, shared=shared,
                SS_terminal=deepcopy(terminal), actual_GPU_execution=False)


def summarize(records, *, sample_ids, clusters, draws=4000):
    require(set(records) == set(ARMS + BASELINES), 'five actual controls and two existing baselines required')
    require(len(sample_ids) == len(set(sample_ids)) == 128 and set(clusters) == set(sample_ids)
            and len(set(clusters.values())) == 93, 'same complete128/93-source panel required')
    for arm, rows in records.items():
        require([r['sample_id'] for r in rows] == sample_ids
                and all(r['source_cluster_id'] == clusters[r['sample_id']] for r in rows),
                'complete ordered paired population required for '+arm)
    result = summarize_view_roles(records, CONTRASTS, draws=draws, seed=20261025)
    result.update(status='PASS_ACTUAL_FINAL_FIXED_CONTEXT_DEV128_EIGHT_METRICS',
        independent_training_seeds=1, generation_noises=1, TEST=False,
        exploratory_extension_after_DEV32_screening=True,
        statistics_scope='exploratory paired source-cluster CI95; not held-out confirmation',
        spatial3_vs_semantic64_is_module_substitution=True,
        scalar_can_change_support=True, Uniform_changes_common_evidence_weighting=True,
        source_preservation=['LPIPS','DreamSim','CLIP'],
        target_alignment=['PSNR','SSIM','M_IoU','CD','FScore'],
        full_goal_complete=False)
    return result
