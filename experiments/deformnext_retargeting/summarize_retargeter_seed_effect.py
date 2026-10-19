"""Summarize the already trained matched Top1 adapter addition at both seeds."""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path

from .compare_ss_noise_panels import compare
from .frozen_utility_confirmation import verify_closed_audit
from .semantic_topk_campaign_wandb import validate_saved_qa
from .semantic_topk_selection import supported
from .semantic_topk_search import write_new
from .summarize_semantic_topk_search import aggregate, validate_evaluation
from .summarize_ss_module_study import audit_final_replay, audit_training


ARMS = ['top1-reference', 'static-top1-c1']


def read(path):
    return json.loads(path.read_text())


def validate_pair(no_adapter, adapter):
    for key in ['models', 'training_data', 'data_dir', 'num_gpus', 'num_nodes', 'training_seed']:
        if no_adapter[key] != adapter[key]:
            raise ValueError('adapter comparison changed common recipe: ' + key)
    for cfg in [no_adapter, adapter]:
        args = cfg['trainer']['args']
        if (cfg['ckpt'] != 'none' or cfg['load_dir'] != '' or cfg['num_gpus'] != 2
                or args['max_steps'] != 480 or args['batch_size_per_gpu'] != 48 or args['batch_split'] != 1):
            raise ValueError('fresh raw480/global96/DDP2/micro48/acc1 comparison required')
    left, right = copy.deepcopy(no_adapter['trainer']), copy.deepcopy(adapter['trainer'])
    specs = [left['args'].pop('deform_conditioner'), right['args'].pop('deform_conditioner')]
    for spec, arm in zip(specs, ['b2', 'c1']):
        if (spec.pop('arm') != arm or spec.get('implementation') != 'r9-native-v5'
                or spec.get('freeze_q2c') is not True or spec.get('task_attention') is not None
                or spec.get('deformation_dim') is not None
                or spec.get('correspondence_mode', 'semantic') != 'semantic'):
            raise ValueError('frozen semantic Top1 comparison without deformation channels required')
        aggregation = spec.pop('candidate_aggregation', 'weighted')
        if aggregation != ('weighted' if arm == 'b2' else 'top1'):
            raise ValueError('both consumers must select raw-score Top1')
        temperature = spec.pop('score_temperature')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('positive dormant temperature required')
    if left != right or specs[0] != specs[1]:
        raise ValueError('adapter comparison changed optimizer,trainer or matching inputs')
    for key, args_key in [('dataset', 'args'), ('validation', 'dataset_args')]:
        normalized = [copy.deepcopy(cfg[key]) for cfg in [no_adapter, adapter]]
        for value, arm in zip(normalized, ['b2', 'c1']):
            if value[args_key].pop('transport_arm') != arm:
                raise ValueError('dataset transport arm differs')
        if normalized[0] != normalized[1]:
            raise ValueError('adapter comparison changed RGB/data/validation protocol')


def summarize(campaign, output):
    results, inputs = {}, []
    for seed, location in [(101, campaign), (202, campaign / 'seed-confirmation-v1')]:
        configs = [location / 'configs' / (arm + '.json') for arm in ARMS]
        cfgs = [read(path) for path in configs]
        validate_pair(*cfgs)
        if any(cfg['training_seed'] != seed for cfg in cfgs):
            raise ValueError('two declared independent training seeds required')
        records, training, models = {}, {}, {}
        for arm, cfg, recipe in zip(ARMS, cfgs, configs):
            run = location / 'runs_ss64' / arm
            initial = json.loads((run / 'validation.jsonl').read_text().splitlines()[0])['quick_metrics']['voxel_iou']
            training[arm] = audit_training(location, arm, initial)
            paths = [location / 'evaluations' / arm / f'noise-{n}/evaluation.json' for n in range(3)]
            records[arm] = [read(path) for path in paths]
            for value in records[arm]:
                validate_evaluation(value)
            training[arm]['final128_replays_epoch32'] = audit_final_replay(training[arm], records[arm][0])
            qa_path = campaign / 'functional-QA-v1' / (f's{seed}-' + arm) / 'summary.json'
            qa = read(qa_path)
            validate_saved_qa(qa, cfg['trainer']['args']['deform_conditioner'])
            if qa.get('saved_frozen_head_exact_to_selected_checkpoint') is not True or len(qa.get('frozen_head_state_keys', [])) != 24:
                raise ValueError('actual original24 frozen matching tensors required')
            job = ('' if seed == 101 else 'confirm-seed202-') + 'train-' + arm
            final_folder = campaign / 'final-weight-audits-v1' / job
            final = verify_closed_audit(final_folder, campaign)
            expected_states = 366 if arm == 'static-top1-c1' else 361
            if (final['frozen_base_tensors_exact'] != 701 or final['LoRA_tensors_updated'] != 360
                    or final['actual_optimizer_step'] != 480 or final['optimizer_states'] != expected_states
                    or final['matching_tensors_changed'] != {'decoder': [], 'score': []}
                    or final['physical_deformation_query_channels'] != 0
                    or final['retargeter_input_channels'] != (2048 if arm == 'static-top1-c1' else None)):
                raise ValueError('actual final adapter/control state differs')
            inputs.extend([final_folder / 'audit.json', final_folder / 'file4-closure.json'])
            for number in range(1, 6):
                folder = campaign / 'epoch-audits-v1' / job / f'epoch{number:02d}'
                epoch = verify_closed_audit(folder, campaign)
                if (epoch['training_seed'] != seed or epoch['step'] != 96 * number
                        or (epoch['global_batch'], epoch['world_size'], epoch['microbatch_per_rank'], epoch['accumulation']) != (96, 2, 48, 1)
                        or epoch['actual_optimizer_states'] != expected_states
                        or not epoch['actual_remote_four_numeric_fields_verified'] or not epoch['actual_remote_PNG_byte_exact']):
                    raise ValueError('actual epoch budget/optimizer/logging evidence differs')
                inputs.extend([folder / 'audit.json', folder / 'file4-closure.json'])
            models[arm] = aggregate(records[arm])
            inputs.extend([recipe, qa_path, run / 'validation.jsonl', *paths])
        if training[ARMS[0]]['initial_iou'] != training[ARMS[1]]['initial_iou']:
            raise ValueError('matched models did not share the initial validation score')
        contrast = compare([(str(n), records[ARMS[0]][n], records[ARMS[1]][n]) for n in range(3)])
        results[str(seed)] = {'models': models, 'contrast': contrast,
                              'positive_CI_and_all_noise_direction': supported(contrast), 'training': training}
    positive = all(value['positive_CI_and_all_noise_direction'] for value in results.values())
    write_new(output, {'status': 'PASS_ACTUAL_MATCHED_RETARGETER_TWO_SEED_COMPARISON',
        'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'per_training_seed': results,
        'retargeter_effect_replicated': positive, 'optimizer_updates_during_analysis': 0,
        'no_new_training_or_generation': True, 'reserved121_evaluated': False,
        'dormant_temperature_exception': 'B2 and C1 Top1 consumers both select raw-logit argmax before temperature. '
            'Their0.005/0.05 values are dormant;all other matching fields are equal.',
        'inputs': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
                   for path in dict.fromkeys([Path(__file__), *inputs])],
        'scope': 'Same Top1 consumer,added C1 retargeter;two independent optimization seeds on existing '
                 'development128/93sources/3generation noises. Each source-cluster CI conditional on its trained '
                 'checkpoint;no training-seed population CI or independent-test claim. This proves neither '
                 'TopK-attention benefit nor decoder/score/correspondence effects in a new full model.',
        'model_adopted': False, 'full_goal_complete': False})
    print(json.dumps({'status': 'PASS_ACTUAL_MATCHED_RETARGETER_TWO_SEED_COMPARISON', 'replicated': positive,
        'effects_pp': {seed: {'gain': 100 * value['contrast']['gain'],
            'CI95': [100 * x for x in value['contrast']['paired_cluster_bootstrap_ci95']]}
            for seed, value in results.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.campaign_root, args.output)


if __name__ == '__main__':
    main()
