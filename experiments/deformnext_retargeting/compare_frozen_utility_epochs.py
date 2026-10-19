"""Intermediate paired SS comparisons, using closed epoch snapshots only."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

from .compare_ss_noise_panels import compare
from .semantic_topk_search import write_new
from .summarize_frozen_utility_followup import zero_records


def read(path):
    return json.loads(path.read_text())


def record_at(path, step):
    records = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return next(record for record in records if record['step'] == step)


def summarize(root, epoch):
    if epoch not in range(2, 6):
        raise ValueError('later epoch must be in 2..5')
    campaign = root.parent
    folder = root / 'epoch-audits-v1/train-frozen-utility-c1' / f'epoch{epoch:02d}'
    closure = folder / 'file4-closure.json'
    if read(closure)['status'] != 'PASS':
        raise ValueError('actual epoch file4 closure required')
    audit = read(folder / 'audit.json')
    step = 96 * epoch
    if (audit['step'] != step or audit['actual_optimizer_states'] != 371
            or not audit['actual_remote_four_numeric_fields_verified']
            or not audit['actual_remote_PNG_byte_exact']):
        raise ValueError('actual reference-profile epoch audit differs')
    candidate = read(folder / 'validation-snapshot.json')
    initial = read(folder / 'initial-validation-snapshot.json')
    if candidate['step'] != step or initial['step'] != 0:
        raise ValueError('epoch snapshot step differs')
    ids = candidate['quick_panel_ids']
    if len(ids) != len(set(ids)) or len(ids) != 32 or ids != initial['quick_panel_ids']:
        raise ValueError('common32 panel identity differs')
    baselines = {
        'V1': (campaign.parent / 'ss-seed-confirmation-v1', 'v1'),
        'old-C1-T005': (campaign.parent / 'ss-seed-confirmation-v1', 'c1'),
        'matched-Top1-C1': (campaign, 'static-top1-c1'),
        'joint-trained-Full': (campaign, 'task-full'),
    }
    inputs = [closure, folder / 'audit.json', folder / 'validation-snapshot.json',
              folder / 'initial-validation-snapshot.json']
    contrasts = {}

    def paired(baseline):
        result = compare([('0', {'per_asset': baseline}, {'per_asset': candidate['quick_per_asset']})])
        if (result['paired_target_hashes_verified'] != 32
                or abs(result['candidate_iou'] - candidate['quick_metrics']['voxel_iou']) > 1e-12):
            raise ValueError('paired actual GT hashes or voxel aggregation differs')
        return result

    for label, (base, arm) in baselines.items():
        path = base / 'runs_ss64' / arm / 'validation.jsonl'
        baseline, start = record_at(path, step), record_at(path, 0)
        if baseline['quick_panel_ids'] != ids or start['quick_per_asset'] != initial['quick_per_asset']:
            raise ValueError('baseline panel or actual initial predictions differ: ' + label)
        contrasts[label] = paired(baseline['quick_per_asset'])
        inputs.append(path)
    zeros, paths = zero_records(campaign)
    zero_contrasts = {}
    for label, repeats in zeros.items():
        lookup = {row['sample_id']: row for row in repeats[0]['per_asset']}
        zero_contrasts[label] = paired([lookup[sid] for sid in ids])
    inputs.extend(paths)
    previous_folder = folder.parent / f'epoch{epoch - 1:02d}'
    if read(previous_folder / 'file4-closure.json')['status'] != 'PASS':
        raise ValueError('previous actual epoch closure required')
    previous = read(previous_folder / 'validation-snapshot.json')
    if previous['step'] != step - 96 or previous['quick_panel_ids'] != ids:
        raise ValueError('previous epoch identity differs')
    temporal = paired(previous['quick_per_asset'])
    inputs.extend([previous_folder / 'file4-closure.json', previous_folder / 'validation-snapshot.json'])
    inputs.append(Path(__file__))
    return {
        'status': 'PASS_ACTUAL_LATER_EPOCH_COMMON32_WITH_BOTH_ZERO_BASELINES',
        'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'epoch': epoch, 'step': step, 'candidate_metrics': candidate['quick_metrics'],
        'contrasts': contrasts, 'zero_baseline_contrasts': zero_contrasts,
        'candidate_vs_previous_epoch': temporal,
        'inputs': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in inputs],
        'scope': 'Common32, one sampling noise, training seed101, paired source-cluster bootstrap. '
                 'Intermediate exploratory trend; fixed raw480/final128x3 selection remains unchanged.',
        'model_adopted': False, 'full_goal_complete': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.epoch)
    write_new(args.output, result)
    print(json.dumps({'status': result['status'], 'epoch': args.epoch,
                      'contrasts': result['contrasts']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
