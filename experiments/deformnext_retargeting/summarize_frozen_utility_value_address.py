"""Bind a frozen-utility value-only intervention to its actual final model."""
import argparse
import hashlib
import json
from pathlib import Path

from .semantic_topk_search import write_new
from .semantic_topk_selected_correspondence import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', choices=['frozen-utility-c1', 'utility-scale16-c1'],
                        default='frozen-utility-c1')
    args = parser.parse_args()
    folder = args.root / 'task-value-address-v1'
    contract_path = folder / 'contract.json'
    contract = json.loads(contract_path.read_text())
    if contract['candidate'] != args.candidate or contract['candidate_preselected'] is not False:
        raise ValueError('declared unselected frozen-utility candidate required')
    paths = [Path(contract[key]) for key in ['normal', 'altered', 'reference']]
    records = [json.loads(path.read_text()) for path in paths]
    if any(record.get('arm') != args.candidate for record in records):
        raise ValueError('actual saved model differs from declared diagnostic candidate')
    panel_path, seed_path = [Path(contract[key]) for key in ['panel', 'seed_map']]
    for key, path in [('panel', panel_path), ('seed_map', seed_path)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != contract[key + '_SHA256']:
            raise ValueError('declared diagnostic panel/noise bytes changed')
    ids = json.loads(panel_path.read_text())['sample_ids']
    seeds = json.loads(seed_path.read_text())
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError('declared common32 required')
    for record in records[:2]:
        if [row['sample_id'] for row in record['per_asset']] != ids:
            raise ValueError('actual intervention panel/order differs')
        if any(row['seed'] != seeds[row['sample_id']] for row in record['per_asset']):
            raise ValueError('actual intervention sampling noise differs')
    result = summarize(*records)
    result.update(
        status='PASS_FROZEN_UTILITY_SAME_POOL_VALUE_ADDRESS_DIAGNOSTIC',
        candidate=args.candidate,
        contract_SHA256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        actual_evaluation_SHA256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        scope='Same actual raw480 candidate/common32/noise0; keys/logits/weights/support and per-query value multiset preserved. Value-to-weight association changes after attention. Not the native-spatial No-corr training arm,not matching accuracy or a model-adoption gate.',
        model_adopted=False, full_goal_complete=False,
    )
    write_new(folder / 'summary.json', result)
    print(json.dumps({'status': result['status'],
                      'normal_minus_wrong_address': result['normal_minus_wrong_address']['gain'],
                      'quality_effect_supported_at_this_panel': result['quality_effect_supported_at_this_panel']}), flush=True)


if __name__ == '__main__':
    main()
