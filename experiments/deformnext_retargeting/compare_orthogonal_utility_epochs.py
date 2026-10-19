"""Compare closed common32 follow-ups, including the same-temperature static control."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

from .compare_frozen_utility_epochs import closed_snapshot, record_at
from .compare_ss_noise_panels import compare
from .semantic_topk_search import write_new
from .summarize_frozen_utility_followup import zero_records


NAMESPACES = {
    'voxel-query-c1': 'voxel-query-followup-v1',
    'utility-softprior-c1': 'utility-softprior-followup-v1',
}


def read(path):
    return json.loads(path.read_text())


def paired_records(baseline, candidate, ids):
    """Require the saved order and GT hashes before using the cluster bootstrap."""
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError('exact distinct common32 panel required')
    for records in [baseline, candidate]:
        if [row['sample_id'] for row in records] != ids:
            raise ValueError('paired saved panel order differs')
    result = compare([('0', {'per_asset': baseline}, {'per_asset': candidate})])
    if result['paired_target_hashes_verified'] != 32:
        raise ValueError('all32 paired actual target hashes required')
    return result


def summarize(root, epoch, candidate_name):
    if epoch not in range(1, 6) or candidate_name not in NAMESPACES:
        raise ValueError('declared candidate and epoch1..5 required')
    if root.name != NAMESPACES[candidate_name]:
        raise ValueError('candidate namespace differs')
    campaign = root.parent
    folder = root / 'epoch-audits-v1' / ('train-' + candidate_name) / f'epoch{epoch:02d}'
    closure = closed_snapshot(folder, campaign)
    audit = read(folder / 'audit.json')
    step = 96 * epoch
    expected = {
        'status': 'PASS_ACTUAL_EPOCH_OPTIMIZER_LOGGING_AND_BATCH',
        'step': step, 'epoch': epoch, 'training_seed': 101,
        'global_batch': 96, 'world_size': 2, 'microbatch_per_rank': 48,
        'accumulation': 1, 'actual_optimizer_states': 371, 'all_optimizer_steps': step,
        'actual_remote_four_numeric_fields_verified': True,
        'actual_remote_PNG_byte_exact': True,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError('actual closed epoch budget/logging audit differs')
    candidate = read(folder / 'validation-snapshot.json')
    initial = read(folder / 'initial-validation-snapshot.json')
    ids = candidate['quick_panel_ids']
    if candidate['step'] != step or initial['step'] != 0 or initial['quick_panel_ids'] != ids:
        raise ValueError('closed snapshot step or initial panel differs')
    baselines = {
        'V1': (campaign.parent / 'ss-seed-confirmation-v1', 'v1'),
        'same-retargeter-Top1': (campaign, 'static-top1-c1'),
        'static-T005': (campaign.parent / 'ss-seed-confirmation-v1', 'c1'),
        'static-T05': (campaign, 'static-c1'),
        'utility-scale4': (campaign / 'frozen-utility-followup-v1', 'frozen-utility-c1'),
        'utility-scale16': (campaign / 'utility-strength-followup-v1', 'utility-scale16-c1'),
    }
    inputs = [closure, folder / 'audit.json', folder / 'validation-snapshot.json',
              folder / 'initial-validation-snapshot.json']
    contrasts = {}
    for label, (base, arm) in baselines.items():
        path = base / 'runs_ss64' / arm / 'validation.jsonl'
        baseline, start = record_at(path, step), record_at(path, 0)
        if baseline['quick_panel_ids'] != ids or start['quick_per_asset'] != initial['quick_per_asset']:
            raise ValueError('reference panel or initial predictions differ: ' + label)
        contrasts[label] = paired_records(baseline['quick_per_asset'], candidate['quick_per_asset'], ids)
        inputs.append(path)
    zeros, paths = zero_records(campaign)
    zero_contrasts = {}
    for label, repeats in zeros.items():
        lookup = {row['sample_id']: row for row in repeats[0]['per_asset']}
        zero_contrasts[label] = paired_records([lookup[sid] for sid in ids], candidate['quick_per_asset'], ids)
    inputs.extend(paths)
    if epoch == 1:
        previous = initial
    else:
        previous_folder = folder.parent / f'epoch{epoch - 1:02d}'
        inputs.append(closed_snapshot(previous_folder, campaign))
        inputs.append(previous_folder / 'validation-snapshot.json')
        previous = read(previous_folder / 'validation-snapshot.json')
    if previous['step'] != step - 96 or previous['quick_panel_ids'] != ids:
        raise ValueError('previous closed epoch identity differs')
    temporal = paired_records(previous['quick_per_asset'], candidate['quick_per_asset'], ids)
    if any(abs(value['candidate_iou'] - candidate['quick_metrics']['voxel_iou']) > 1e-12
           for value in [*contrasts.values(), *zero_contrasts.values(), temporal]):
        raise ValueError('saved voxel aggregation differs')
    inputs.append(Path(__file__))
    return {
        'status': 'PASS_CLOSED_ORTHOGONAL_COMMON32_WITH_T05_AND_BOTH_ZEROS',
        'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'candidate': candidate_name, 'epoch': epoch, 'step': step,
        'candidate_metrics': candidate['quick_metrics'], 'contrasts': contrasts,
        'zero_baseline_contrasts': zero_contrasts, 'candidate_vs_previous_epoch': temporal,
        'inputs': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in dict.fromkeys(inputs)],
        'scope': 'Exploratory common32/noise0/one training seed. Source-cluster bootstrap; '
                 'fixed raw480/dev128x3 decision remains unchanged. Not model adoption.',
        'static_T05_scope': 'Same score temperature isolates learned utility from temperature-only '
                            'change in softprior arm; voxel-query arm also changes the query.',
        'model_adopted': False, 'full_goal_complete': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', choices=list(NAMESPACES), required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.epoch, args.candidate)
    write_new(args.output, result)
    print(json.dumps({'status': result['status'], 'candidate': args.candidate,
                      'epoch': args.epoch, 'contrasts': result['contrasts']}), flush=True)


if __name__ == '__main__':
    main()
