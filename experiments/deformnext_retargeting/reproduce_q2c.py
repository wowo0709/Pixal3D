"""Replay the historical D0/D1 controlled RGB task with current vendored math.

The packed inputs are exported once from authenticated historical authorities.
They are a mechanism reproduction, not DeformNeXt validation or SS training.
Evaluation cue arrays are opened only after the fixed-final checkpoint is saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .historical_q2c_training import (
    R9TrainingRunner, TrainingTuple, _normalize_records, wrong_indices, component_rows,
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(reference):
    path = Path(reference['path'])
    if sha(path) != reference['sha256']:
        raise ValueError('historical input checksum differs: ' + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def reference(path):
    return {'path': str(path), 'sha256': sha(path)}


def record_from_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {**{key: data[key].copy() for key in
                   ['candidates', 'raw_scores', 'label_rgb', 'query_vertex_indices', 'target_rgb']},
                'query': {12: data['query_L12'].copy()},
                'source': {12: data['source_L12'].copy()}}


def prepare(old, output):
    """Freeze exact original fold/carrier/proposal inputs under the new file4 root."""
    if (output / 'summary.json').exists():
        raise FileExistsError('immutable prepared Q2C inputs already exist')
    schedule = read(old / 'exp4/_R9_SCHEDULE.json')
    lock_path = checked(schedule['input_lock'])
    lock = read(lock_path)
    evaluator_path = checked(schedule['evaluator_authority'])
    evaluator = read(evaluator_path)
    table = read(checked(schedule['runs'][0]['table_authority']))
    source_records = {}
    tuples = []
    for entry in schedule['runs']:
        info = entry['tuple']
        if info['decoder_arm'] not in ['D0-PAIRWISE', 'D1-Q2C']:
            continue
        if (info['wave'], info['backbone'], info['projector_arm'], info['multi_arm'], info['region']) != (
                'exp4', 'OMEGA_L12_FRAME_RAW', 'P0-RAW', 'M0-L12', 'REG-GLOBAL64'):
            raise ValueError('only the exact historical promoted upstream is supported')
        path = checked(entry['manifest'])
        manifest = read(path)
        if manifest['updates'] != 1200 or manifest['queries_per_update'] != 256:
            raise ValueError('historical training budget differs')
        expected_train = sorted(lock['folds'][info['fold']]['train_pair_ids'])
        if sorted(manifest['train_pair_ids']) != expected_train:
            raise ValueError('component-disjoint historical training fold differs')
        calibration = read(checked(manifest['calibration']))
        if calibration['temperature'] != .005 or calibration['selection_source'] != 'fold_train_label_only':
            raise ValueError('historical TRAIN-only temperature differs')
        upstream_path = checked(manifest['upstream_checkpoint'])
        # P0 has no learned representation parameters. Verify the actual artifact,
        # and retain an empty upstream state so the copied Exp4 initializer freezes it.
        upstream = torch.load(upstream_path, map_location='cpu', weights_only=False)
        if any(k.startswith('projector.') for k in upstream['model']):
            raise ValueError('P0 upstream unexpectedly contains projector parameters')
        for record in manifest['records']:
            pair = record['pair_id']
            if pair in source_records:
                if source_records[pair]['original'] != record:
                    raise ValueError('same-pair historical TRAIN inputs differ across D0/D1/folds')
                continue
            paths = {key: checked(record[key]) for key in ['packed', 'descriptor', 'table', 'label']}
            fields = record['source_fields']
            if fields['layers'] != {'12': {'source': 'source_L12_FRAME', 'target': 'target_L12_FRAME'}}:
                raise ValueError('historical descriptor layers differ')
            with np.load(paths['packed'], allow_pickle=False) as packed, np.load(
                    paths['descriptor'], allow_pickle=False) as desc, np.load(
                    paths['table'], allow_pickle=False) as arrays, np.load(paths['label'], allow_pickle=False) as label:
                ids = arrays[fields['query_vertex_indices']].astype(np.int64)
                expected = {'query_vertex_indices': ids,
                            'candidates': arrays[fields['candidates']].astype(np.int64),
                            'raw_scores': arrays[fields['raw_scores']].astype(np.float32),
                            'query_L12': desc['target_L12_FRAME'][ids].astype(np.float32),
                            'source_L12': desc['source_L12_FRAME'].astype(np.float32),
                            'label_rgb': label['label_rgb'], 'target_rgb': label['label_rgb'][:, ids]}
                if set(packed.files) != set(expected):
                    raise ValueError('packed historical training fields differ')
                for key, value in expected.items():
                    if packed[key].dtype != value.dtype or not np.array_equal(packed[key], value):
                        raise ValueError('packed training differs from authenticated source: ' + key)
            target = output / 'train' / (pair + '.npz')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            shutil.copyfile(paths['packed'], target)
            source_records[pair] = {'original': record, 'staged': reference(target)}
        train_log = old / 'exp4/runs' / entry['run_id'] / 'train_log.jsonl'
        log = [json.loads(line) for line in train_log.read_text().splitlines()]
        if len(log) != 1200:
            raise ValueError('historical fixed-final log incomplete')
        tuples.append({'tuple': info, 'source_manifest': reference(path),
                       'upstream_checkpoint': manifest['upstream_checkpoint'],
                       'temperature': calibration['temperature'],
                       'train_records': {pair: source_records[pair]['staged'] for pair in expected_train},
                       'historical_steps': [{k: row[k] for k in
                           ['update', 'pair_id', 'sampled_query_sha256', 'loss']} for row in log],
                       'historical_log': reference(train_log),
                       'held_out_pair_ids': sorted(lock['folds'][info['fold']]['validation_pair_ids'])})
    if len(tuples) != 24 or len(source_records) != 63:
        raise ValueError('historical 4-fold x 3-seed x 2-arm population differs')
    eval_records = {}
    for pair, authority in evaluator['pairs']['OMEGA_L12_FRAME_RAW'].items():
        proposal = table['records']['OMEGA_L12_FRAME_RAW'][pair]['REG-GLOBAL64']
        with np.load(checked(authority['descriptor']), allow_pickle=False) as desc, np.load(
                checked(proposal['table']), allow_pickle=False) as arrays, np.load(
                checked(authority['evidence']), allow_pickle=False) as evidence, np.load(
                checked(authority['parent_table']), allow_pickle=False) as parent:
            ids = arrays['eval_query_ids'].astype(np.int64)
            if not np.array_equal(ids, parent['eval_query_indices']):
                raise ValueError('historical evaluator query slots differ')
            eligible = evidence['evaluator_eligible'].astype(bool)
            if np.flatnonzero(eligible).tolist() != evaluator['eligible_cue_ids'][pair]:
                raise ValueError('historical evaluator cue eligibility differs')
            maps = np.stack([wrong_indices(parent['visible'].astype(bool),
                cue_seed=lock['evaluator_cues_by_pair'][pair][int(cue)]['cue_seed'],
                lock_seed=evaluator['r8_seed']) for cue in np.flatnonzero(eligible)])
            target = output / 'evaluation' / (pair + '.npz')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            np.savez_compressed(target, query_L12=desc['target_L12_FRAME'][ids].astype(np.float32),
                source_L12=desc['source_L12_FRAME'].astype(np.float32), query_ids=ids,
                candidates=arrays['eval_REG-GLOBAL64_candidate_indices'].astype(np.int64),
                raw_scores=arrays['eval_REG-GLOBAL64_raw_scores'].astype(np.float32),
                evaluator_rgb=evidence['evaluator_rgb'].astype(np.float64), eligible=eligible, wrong_maps=maps)
        eval_records[pair] = reference(target)
        print('prepared historical Q2C pair', len(eval_records), '/63', flush=True)
    for entry in tuples:
        entry['evaluation_records'] = {p: eval_records[p] for p in entry['held_out_pair_ids']}
        info = entry['tuple']
        path = output / 'tuples' / f"{info['decoder_arm']}-fold{info['fold']}-seed{info['seed']}.json"
        write_new(path, entry)
    write_new(output / 'summary.json', {'status': 'PASS', 'source_input_lock': reference(lock_path),
              'source_evaluator': reference(evaluator_path), 'pairs': 63, 'training_tuples': len(tuples),
              'data_scope': 'historical controlled cue task; distinct from current DeformNeXt SS endpoint',
              'training_records': source_records, 'evaluation_records': eval_records})


def train(path, output, wandb_id=None):
    spec = read(path)
    info = TrainingTuple(**spec['tuple'])
    if (output / 'summary.json').exists() or (output / 'checkpoint-final.pt').exists():
        raise FileExistsError('immutable Q2C reproduction output exists')
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    records = {pair: record_from_npz(checked(ref)) for pair, ref in spec['train_records'].items()}
    runner = R9TrainingRunner(_normalize_records(records), temperature=spec['temperature'],
               device='cuda:0', tuple_info=info, upstream_state={},
               authority={'reproduction_input_sha256': sha(path)})
    output.mkdir(parents=True, exist_ok=True)
    run = None
    if wandb_id:
        import wandb
        run = wandb.init(project='pixal3d-deformnext-retargeting', entity='youngwoojeon-unist',
                         id=wandb_id, resume='never', dir=str(output), config={
                             'scope': 'historical-controlled-RGB-reproduction', **spec['tuple'],
                             'updates': 1200, 'queries_per_update': 256, 'candidate_budget': 64,
                             'loss': 'RGB_L1+0.05_preservation', 'inference_RGB_input': False})
    started = time.monotonic()
    differences = []
    with (output / 'train-log.jsonl').open('x') as log:
        for update in range(1200):
            row = runner.step()
            previous = spec['historical_steps'][update]
            for key in ['update', 'pair_id', 'sampled_query_sha256']:
                if row[key] != previous[key]:
                    raise ValueError('reproduction historical sampler differs: ' + key)
            differences.append(abs(row['loss'] - previous['loss']))
            log.write(json.dumps(row) + '\n')
            if update == 0 or (update + 1) % 25 == 0:
                log.flush()
                event = {**row, 'elapsed_seconds': time.monotonic() - started,
                         'historical_loss_abs_difference': differences[-1]}
                print(json.dumps(event), flush=True)
                if run:
                    run.log(event, step=update + 1)
    checkpoint = output / 'checkpoint-final.pt'
    torch.save({'model': runner.model.state_dict(), 'optimizer': runner.optimizer.state_dict(),
                'tuple': spec['tuple'], 'completed_updates': 1200,
                'input_sha256': sha(path), 'zero_start_proof': runner.zero_start_proof}, checkpoint)
    frozen = reference(checkpoint)
    write_new(output / 'checkpoint-freeze.json', {'checkpoint': frozen, 'completed_updates': 1200,
              'learned_evaluation_opened': False})
    # First evaluator array access follows publication of the final checkpoint.
    runner.model.eval()
    rows = []
    with torch.no_grad():
        for pair, ref in spec['evaluation_records'].items():
            with np.load(checked(ref), allow_pickle=False) as data:
                source = {12: torch.tensor(data['source_L12'], device='cuda:0')}
                scores = []
                for start in range(0, len(data['query_ids']), 32):
                    query = {12: torch.tensor(data['query_L12'][start:start+32], device='cuda:0')}
                    ids = torch.tensor(data['candidates'][start:start+32], device='cuda:0')
                    raw = torch.tensor(data['raw_scores'][start:start+32], device='cuda:0')
                    scores.append(runner.model(query, source, ids, raw_scores=raw)['scores'].cpu().numpy())
                rows.extend(component_rows(metadata={'pair_id': pair, 'arm': info.decoder_arm,
                    'fold': info.fold, 'seed': info.seed, 'split': 'DEV'}, scores=np.concatenate(scores),
                    candidates=data['candidates'], query_ids=data['query_ids'], cue_rgb=data['evaluator_rgb'],
                    eligible=data['eligible'], wrong_maps=data['wrong_maps'], temperature=spec['temperature']))
    gates = {name: float(value.detach()) for name, value in runner.model.named_parameters()
             if name.endswith(('attention_gate', 'ffn_gate'))}
    pair_errors = [np.mean([r['task_rgb_error'] for r in rows if r['pair_id'] == pair])
                   for pair in spec['held_out_pair_ids']]
    summary = {'status': 'PASS', 'scope': 'historical-controlled-RGB-task-only', 'tuple': spec['tuple'],
               'completed_updates': 1200, 'checkpoint': frozen, 'input': reference(path),
               'evaluation_opened_after_fixed_final_freeze': True,
               'all_1200_sampled_pairs_and_query_hashes_match_historical': True,
               'historical_loss_difference_first': differences[0],
               'historical_loss_difference_max': max(differences),
               'historical_loss_difference_final': differences[-1],
               'zero_start_proof': runner.zero_start_proof, 'context_gates': gates,
               'held_out_pairs': len(pair_errors), 'task_rgb_l1_pair_mean': float(np.mean(pair_errors)),
               'elapsed_seconds': time.monotonic() - started, 'component_rows': rows,
               'performance_promotion': False, 'current_DeformNeXt_SS_gain_claim': False}
    write_new(output / 'summary.json', summary)
    if run:
        run.summary.update({k: v for k, v in summary.items() if isinstance(v, (int, float, bool, str))})
        run.finish()
    print(json.dumps({k: v for k, v in summary.items() if k != 'component_rows'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'train'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wandb-id')
    args = parser.parse_args()
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    if args.mode == 'prepare':
        torch.set_num_threads(2)
        prepare(args.input, args.output)
    else:
        train(args.input, args.output, args.wandb_id)


if __name__ == '__main__':
    main()
