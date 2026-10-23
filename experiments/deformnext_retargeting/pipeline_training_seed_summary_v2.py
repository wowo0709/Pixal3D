"""Compare one measured module candidate and strong controls across TRAIN seeds.

The reducer selects no model and queues no work. File/checkpoint integrity and
actual frozen-base tensor comparisons belong to the closed-input caller.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np

from .pipeline_confirmation_recipe_v2 import validate_arm, validate_confirmation, models_for_selected
from .r_mediated_native_recipe import common_contract
from .pipeline_population_contract_v2 import ALIASES, SOURCE_COUNTS
from .consistent_evidence_native_recipe import VARIANTS as FOLLOWUPS


METRICS = ('voxel_iou', 'voxel_precision', 'voxel_recall', 'voxel_f1')
SEEDS = (101, 202)




def _exact(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_evaluations(evaluations, cfg, arm):
    validate_arm(cfg, arm, 'science')
    if len(evaluations) != 3:
        raise ValueError('three actual matched generation-noise evaluations required')
    root = Path(cfg['output_dir'])
    if not root.is_absolute():
        raise ValueError('explicit actual native model root required')
    identities, values, binding = [], [], None
    for result in evaluations:
        rows = result['per_asset']
        if (result.get('status') != 'PASS' or result.get('stage') != 'ss64'
                or type(result.get('step')) is not int or result['step'] != 480
                or result.get('native_variant') != ALIASES.get(arm, arm)
                or not _exact([result.get('actual_source_tensors'), result.get('actual_LoRA_tensors'),
                               result.get('optimizer_updates_during_evaluation')], [SOURCE_COUNTS[arm], 360, 0])
                or result.get('evaluation_mode') != 'final128'
                or result.get('weights') != 'raw'
                or not _exact([result.get('generation_batch_size'), result.get('sampling_steps')], [8, 12])
                or result.get('guidance') != 3.
                or result.get('optimizer_updates_during_evaluation') != 0
                or result.get('deterministic_algorithms') is not True
                or result.get('checkpoint') != str(root/'ckpts/denoiser_step0000480.pt')
                or result.get('conditioner_checkpoint') != (None if arm == 'V1' else
                    str(root/'ckpts/deform_conditioner_step0000480.pt'))
                or result.get('arm_config') != str(root/'config.json')
                or result.get('input_states') != ['M1', 'M0', 'M0', 'M0']
                or not _exact(result.get('input_camera_ids'), [8, 9, 10, 11])
                or len(rows) != 128 or len({r['sample_id'] for r in rows}) != 128):
            raise ValueError('own complete native480 physical family and deterministic DEV128 required')
        paths = [result['checkpoint'], result['arm_config']] + ([] if arm == 'V1' else [result['conditioner_checkpoint']])
        own_binding = {p: result.get('verified_inputs', {}).get(p) for p in paths}
        if any(not _sha(v) for v in own_binding.values()) or (binding is not None and binding != own_binding):
            raise ValueError('all generation noises must use the same actually byte-verified checkpoint/source/config')
        binding = own_binding
        if arm in FOLLOWUPS and (result.get('conditioning_schema') != 'consistent-raw-ray-evidence-native-v1'
                or result.get('followup_study_schema') != 'consistent-raw-ray-evidence-followup-v1'
                or result.get('shared_carrier_use_retarget_evidence') is not True
                or result.get('learned_projection_tensors') != 0
                or result.get('learned_attention_head_tensors') != (7 if arm=='E_LEARNED_RAW' else 0)
                or type(result.get('training_seed')) is not int
                or result['training_seed'] != cfg['training_seed']):
            raise ValueError('actual shared-evidence runtime, physical head and own training seed required')
        for row in rows:
            if (not isinstance(row['sample_id'], str) or not row['sample_id']
                    or not isinstance(row['source_cluster_id'], str) or not row['source_cluster_id']
                    or type(row['seed']) is not int or not 0 <= row['seed'] < 2**31
                    or not _sha(row['target_voxels_sha256'])):
                raise ValueError('actual asset, source-cluster, integer noise and GT identities required')
        identities.append([(r['sample_id'], r['source_cluster_id'], r['seed'], r['target_voxels_sha256']) for r in rows])
        values.append([[r[m] for m in METRICS] for r in rows])
    if any(len({identities[n][i][2] for n in range(3)}) != 3 for i in range(128)):
        raise ValueError('three distinct generation noises per actual asset required')
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (3, 128, 4) or not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():
        raise ValueError('finite complete voxel metrics in[0,1] required')
    return identities, matrix.mean(0), binding


def validate_training(traces, startups, *, selected):
    arms = models_for_selected(selected)
    if set(traces) != set(SEEDS) or set(startups) != set(SEEDS):
        raise ValueError('actual training records from101 and202 required')
    initial = set()
    for seed in SEEDS:
        if set(traces[seed]) != set(arms) or set(startups[seed]) != set(arms):
            raise ValueError('selected candidate and both deduplicated strong controls required')
        reference, start = traces[seed]['V1'], startups[seed]['V1']
        for arm in arms:
            rows, own = traces[seed][arm], startups[seed][arm]
            if (len(rows) != 480 or not _exact(rows, reference)
                    or [r['step'] for r in rows] != list(range(1, 481))
                    or any(type(r['step']) is not int or len(r['asset_ids']) != 1
                           or len(r['asset_ids'][0]) != 96
                           or any(not isinstance(s, str) or not s for s in r['asset_ids'][0])
                           or any(not _sha(r[k]) for k in ('x_t_SHA256', 't_SHA256', 'global_SHA256')) for r in rows)
                    or own.get('status') != 'PASS_ACTUAL_COLD_NATIVE_TASK_SS_STARTUP'
                    or own.get('variant') != ALIASES.get(arm, arm)
                    or own.get('source_tensors') != SOURCE_COUNTS[arm]
                    or not _exact([own.get('source_tensors'), own.get('LoRA_tensors'), own.get('global_batch'),
                        own.get('microbatch'), own.get('accumulation'), own.get('actual_optimizer_states')],
                        [SOURCE_COUNTS[arm], 360, 96, 96, 1, 0])
                    or not _exact(own['post_constructor_RNG'], start['post_constructor_RNG'])
                    or own['TRAIN_order_SHA256'] != start['TRAIN_order_SHA256']
                    or set(own['post_constructor_RNG']) != {'CPU', 'CUDA', 'numpy', 'python'}
                    or any(not _sha(v) for v in own['post_constructor_RNG'].values())
                    or not _sha(own['TRAIN_order_SHA256']) or not _sha(own['initial_LoRA_SHA256'])):
                raise ValueError('complete paired480 actual TRAIN96/noise/timestep/global and cold startup records required')
            initial.add(own['initial_LoRA_SHA256'])
    if (len(initial) != 1 or _exact(startups[101]['V1']['post_constructor_RNG'], startups[202]['V1']['post_constructor_RNG'])
            or _exact(traces[101]['V1'], traces[202]['V1'])):
        raise ValueError('same original LoRA initializer and genuinely changed training RNG/stream required')


def summarize(results, configs, traces, startups, *, selected, draws=4000, seed=20261023):
    arms = models_for_selected(selected)
    validate_training(traces, startups, selected=selected)
    if set(results) != set(SEEDS) or set(configs) != set(SEEDS) or type(draws) is not int or draws < 1:
        raise ValueError('complete two-training-seed evidence and positive bootstrap draws required')
    matrices, identities, bindings = {}, None, {}
    execution = {}
    for train_seed in SEEDS:
        if set(results[train_seed]) != set(arms) or set(configs[train_seed]) != set(arms):
            raise ValueError('actual measured candidate and deduplicated baselines required')
        matrices[train_seed] = {}
        bindings[train_seed] = {}
        for arm in arms:
            cfg = configs[train_seed][arm]
            if type(cfg['training_seed']) is not int or cfg['training_seed'] != train_seed:
                raise ValueError('own actual training seed required')
            own, values, binding = validate_evaluations(results[train_seed][arm], cfg, arm)
            if identities is None:
                identities = own
            elif own != identities:
                raise ValueError('asset/cluster/generation-noise/GT pairing differs across models or training seeds')
            matrices[train_seed][arm] = values
            bindings[train_seed][arm] = binding
            if not _exact(common_contract(cfg), common_contract(configs[train_seed]['V1'])):
                raise ValueError('native model/data/numerical training contract differs within seed')
    for arm in arms:
        original, actual = configs[101][arm], configs[202][arm]
        normalized = deepcopy(actual)
        # Device assignment is recorded operational metadata; this reducer
        # authorizes no GPU execution. Original runtime guards and closed run
        # plans govern actual device assignment and initial/capacity admission.
        execution[arm] = {}
        for field in ('assigned_GPUs', 'CPU_affinity_per_GPU'):
            execution[arm][field] = {str(s): configs[s][arm]['experiment'].get(field) for s in SEEDS}
            if field in original['experiment']:
                normalized['experiment'][field] = deepcopy(original['experiment'][field])
            else:
                normalized['experiment'].pop(field, None)
        validate_confirmation(original, normalized, arm=arm, phase='science')
    clusters = np.asarray([r[1] for r in identities[0]])
    groups = [np.flatnonzero(clusters == c) for c in sorted(set(clusters))]
    sizes = np.asarray([len(g) for g in groups])
    sample = np.random.default_rng(seed).integers(0, len(groups), (draws, len(groups)))
    pairs = [('C_DECODER', 'V1')]
    if selected not in ('V1', 'C_DECODER'):
        pairs += [(selected, 'V1'), (selected, 'C_DECODER')]

    def compare(delta):
        sums = np.stack([delta[g].sum(0) for g in groups])
        boot = sums[sample].sum(1)/sizes[sample].sum(1)[:, None]
        source = (sums/sizes[:, None])[sample].mean(1)
        return dict(mean_asset_delta=dict(zip(METRICS, delta.mean(0).tolist())),
            source_cluster_CI95={m: np.quantile(boot[:, i], [.025, .975]).tolist() for i, m in enumerate(METRICS)},
            mean_source_delta=dict(zip(METRICS, (sums/sizes[:, None]).mean(0).tolist())),
            equal_source_cluster_CI95={m: np.quantile(source[:, i], [.025, .975]).tolist() for i, m in enumerate(METRICS)},
            IoU_family_size=len(pairs), IoU_family_CI95=np.quantile(boot[:, 0], [.025/len(pairs), 1-.025/len(pairs)]).tolist())

    per_seed, averaged = {}, {}
    for s in SEEDS:
        per_seed[s] = dict(means={a: dict(zip(METRICS, matrices[s][a].mean(0).tolist())) for a in arms},
            comparisons={a+'_minus_'+b: compare(matrices[s][a]-matrices[s][b]) for a, b in pairs})
    for a, b in pairs:
        differences = [matrices[s][a]-matrices[s][b] for s in SEEDS]
        averaged[a+'_minus_'+b] = dict(compare(np.stack(differences).mean(0)),
            mean_IoU_direction_positive_in_both_training_seeds=all(d[:, 0].mean() > 0 for d in differences))
    return dict(status='PASS_SELECTED_PIPELINE_SS_TWO_TRAINING_SEEDS_PAIRED_DEV128X3',
        selected_candidate=selected, physical_family_explicit=True, original_finite12_results_unchanged=True, measured_models=list(arms), training_seeds=list(SEEDS),
        assets=128, source_clusters=len(groups), generation_noises=3,
        primary_metric='mean asset voxel IoU after per-asset three-generation-noise average',
        per_training_seed=per_seed, fixed_two_seed_average=averaged,
        reported_byte_verified_model_bindings=bindings,
        execution_assignments_recorded=execution,
        actual480_training_traces_paired_within_seed_and_distinct_between_seeds=True,
        source_cluster_CIs_condition_on_two_fixed_training_seeds_not_seed_population=True,
        repeated_DEV_search_not_independent_TEST=True,
        native_pooled_IoU_is_not_the_reported_asset_mean=True,
        actual_frozen_base_and_checkpoint_byte_evidence_required_separately=True,
        full_cascade_and_PBR_seed_confirmation_not_proven_here=True,
        automatic_model_adoption=False, full_goal_complete=False)
