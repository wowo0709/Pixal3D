"""Fixed-final candidate and proposal-value diagnostics, without model selection claims."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .compare_ss_noise_panels import compare


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, assets, *, legacy_receipt=None, archive_root=None):
    result = json.loads(path.read_text())
    if (result['status'] != 'PASS' or result['step'] != 480
            or result['generation_batch_size'] != 8 or len(result['per_asset']) != assets
            or result['optimizer_updates_during_evaluation'] != 0
            or result['conditioning_mode'] != 'edited-rgb'):
        raise ValueError('fixed480 saved weights,paired RGB generation and zero updates required')
    if legacy_receipt is None:
        audit = path.parent/'closed-evaluation-audit.json'
        record = json.loads(audit.read_text())
        if (record['status'] != 'PASS_CLOSED_NATIVE_SS_EVALUATION'
                or record['evaluation_SHA256'] != sha(path)):
            raise ValueError('closed native saved-weight evaluation required')
    else:
        record = json.loads(legacy_receipt.read_text())
        relative = str(path.relative_to(archive_root))
        if (record['status'] != 'PASS' or not record['local_equals_actual_file4_SHA256']
                or not any(row['relative_path']==relative and row['sha256']==sha(path)
                           for row in record['files'])):
            raise ValueError('original baseline evaluation must match its actual file4 receipt')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['strong-reference','proposal-control'],required=True)
    parser.add_argument('--campaign-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    q = args.campaign_root/'ray-group-study-v1'
    f = q/'anchor-ray-composition-v1/feature-readout-v1'
    direct = f/'direct-retargeting-v1'
    inputs = {}

    def read(path,assets,receipt=None):
        result = load(path,assets,legacy_receipt=receipt,archive_root=args.campaign_root.parent)
        inputs[str(path)] = sha(path)
        return result

    if args.kind == 'strong-reference':
        roots = dict(control=f/'matched-final-v1/control',feature=f/'matched-final-v1/feature',
            dq=f/'deformation-query-v1/matched-final-v1',direct=direct/'matched-final-v1',
            v1=q/'deterministic-evaluation-v1/study-v1/models/v1',
            original_positive_R=q/'deterministic-evaluation-v1/study-v1/models/old-top1-R')
        records = {name:[read(root/f'noise-{i}/evaluation.json',128,
                             root/'file4-closure.json' if name in {'v1','original_positive_R'} else None)
                        for i in range(3)]
                   for name,root in roots.items()}
        contrasts = {}
        support = {}
        for candidate in ['feature','dq','direct']:
            for baseline in ['control','original_positive_R','v1']:
                result = compare([(str(i),records[baseline][i],records[candidate][i]) for i in range(3)])
                if (result['assets'],result['source_clusters'],result['paired_target_hashes_verified']) != (128,93,384):
                    raise ValueError('full source-cluster paired128x3 final comparison required')
                contrasts[candidate+' minus '+baseline] = result
            support[candidate] = all(contrasts[candidate+' minus '+baseline]['paired_cluster_bootstrap_ci95'][0]>0
                for baseline in ['control','original_positive_R','v1'])
        details = dict(contrasts=contrasts,candidate_supports_independent_training_seed_followup=support,
            reference='Matched new R continuation AND the original positive R480 AND vanilla V1.',
            multiple_candidate_comparisons_exploratory=True,independent_training_seed_confirmation=False,
            heldout_test_claim=False,predeclared_final_step=480,best_epoch_selection=False,
            no_module_effect_claim_from_delivery_and_query_changed_together=True)
    else:
        normal = read(direct/'matched-final-v1/replay32/evaluation.json',32)
        replaced = read(direct/'outside-proposal-control-v1/evaluation.json',32)
        for key in ['checkpoint','conditioner_checkpoint','arm_config','weights']:
            if normal[key] != replaced[key]:
                raise ValueError('proposal-value diagnostic must retain exactly the same weights/config')
        intervention = replaced['outside_proposal_value_control']
        if (intervention['status'] != 'EXECUTED' or intervention['reassigned_valid_slots'] <= 0
                or not intervention['original_Top1_carrier_and_keys_and_scores_and_probabilities_retained']
                or not intervention['appearance_and_RAW_values_replaced_together'] or intervention['GT_added']):
            raise ValueError('actual RGB-only outside-proposal payload intervention required')
        result = compare([('noise0',replaced,normal)])
        if result['paired_target_hashes_verified'] != 32:
            raise ValueError('all32 target geometry hashes must agree')
        details = dict(full_minus_outside_proposal_values=result,
            selected_proposal_values_help_on_fixed32=result['paired_cluster_bootstrap_ci95'][0]>0,
            outside_valid_slots_reassigned=intervention['reassigned_valid_slots'],
            original_Top1_carrier_retained=True,keys_scores_attention_probabilities_retained=True,
            donor_reuse_allowed_when_fewer_than64_outside_patches=True,
            scope='Saved-weight added-payload sensitivity;not native-spatial NoCorr or a retrained matching ablation.',
            fixed_development_panel_only=True,independent_training_seed_confirmation=False)
    args.output.parent.mkdir(parents=True,exist_ok=False)
    with args.output.open('x') as stream:
        json.dump(dict(status='PASS_FIXED_FINAL_SS_'+args.kind.upper().replace('-','_'),
            **details,inputs=inputs,source_SHA256={str(Path(__file__).resolve()):sha(__file__)},
            model_adopted=False,full_goal_complete=False),stream,indent=2)
        stream.write('\n')


if __name__=='__main__':
    main()
