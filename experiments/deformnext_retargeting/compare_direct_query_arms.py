"""Separate Direct delivery and physical query effects at the fixed SS final."""
from pathlib import Path
import argparse
import hashlib
import json

from .audit_ss_matching_candidates import load
from .compare_ss_noise_panels import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    n = args.campaign_root
    q = n/'ray-group-study-v1'
    f = q/'anchor-ray-composition-v1/feature-readout-v1'
    roots = dict(control=f/'matched-final-v1/control',input_noDQ=f/'matched-final-v1/feature',
        input_DQ=f/'deformation-query-v1/matched-final-v1',
        direct_DQ=f/'direct-retargeting-v1/matched-final-v1',
        direct_noDQ=f/'direct-no-query-v1/matched-final-v1',
        v1=q/'deterministic-evaluation-v1/study-v1/models/v1',
        original_positive_R=q/'deterministic-evaluation-v1/study-v1/models/old-top1-R')
    records,inputs = {},{}
    for name,root in roots.items():
        records[name] = []
        for noise in range(3):
            path = root/f'noise-{noise}/evaluation.json'
            receipt = root/'file4-closure.json' if name in {'v1','original_positive_R'} else None
            records[name].append(load(path,128,legacy_receipt=receipt,archive_root=n.parent))
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    contrasts = {}
    for candidate,baseline,meaning in [
        ('direct_DQ','direct_noDQ','Physical query effect with Direct delivery held fixed.'),
        ('direct_noDQ','input_noDQ','Direct delivery effect with query channels absent in both.'),
        ('direct_DQ','input_DQ','Direct delivery effect with the same physical query module.'),
        ('input_DQ','input_noDQ','Physical query effect with input-side delivery held fixed.'),
        ('direct_noDQ','control','Added TopK feature readout with original R/carrier retained.'),
        ('direct_noDQ','original_positive_R','Strong historical benchmark;different post96 training stream.'),
        ('direct_noDQ','v1','Vanilla benchmark at equal total training budget.'),
        ('direct_DQ','original_positive_R','Strong historical benchmark;different post96 training stream.'),
        ('direct_DQ','v1','Vanilla benchmark at equal total training budget.')]:
        result = compare([(str(i),records[baseline][i],records[candidate][i]) for i in range(3)])
        if (result['assets'],result['source_clusters'],result['paired_target_hashes_verified'])!=(128,93,384):
            raise ValueError('full paired128x3 source-cluster comparison required')
        contrasts[candidate+' minus '+baseline] = dict(result=result,interpretation=meaning)
    path = roots['direct_noDQ']/'replay32/evaluation.json'
    replay = load(path,32)
    full = {r['sample_id']:r for r in records['direct_noDQ'][0]['per_asset']}
    exact = all(row==full[row['sample_id']] for row in replay['per_asset'])
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    support = all(contrasts['direct_noDQ minus '+reference]['result']['paired_cluster_bootstrap_ci95'][0]>0
        for reference in ['control','original_positive_R','v1']) and exact
    args.output.parent.mkdir(parents=True,exist_ok=False)
    with args.output.open('x') as stream:
        json.dump(dict(status='PASS_FIXED480_DIRECT_QUERY_DELIVERY_EFFECTS128x3',contrasts=contrasts,
            replay32_exact=exact,NoDQ_supports_independent_training_seed_followup=support,
            query_channels_physically_absent_in_NoDQ=True,retargeting_adapter_present_in_all_four_cells=True,
            common_reference_step=96,additional_updates=384,total_updates=480,
            independent_training_seed_confirmation=False,multiple_candidate_exploratory=True,
            interaction_significance_claim=False,heldout_test_claim=False,
            inputs=inputs,model_adopted=False,full_goal_complete=False),stream,indent=2)
        stream.write('\n')


if __name__=='__main__':
    main()
