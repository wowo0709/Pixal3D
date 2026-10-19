"""Run a pinned deterministic SS evaluator with an explicit GT-free64-candidate ray proposal control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--proposal-mode',choices=['patch-union'],required=True)
    parser.add_argument('--identity-proof',type=Path,required=True)
    parser.add_argument('--identity-receipt',type=Path,required=True)
    parser.add_argument('--frozen-source-contract',type=Path,required=True)
    parser.add_argument('--frozen-source-receipt',type=Path,required=True)
    parser.add_argument('--campaign-root',type=Path,required=True)
    parser.add_argument('--expected-assets',type=int,default=32)
    local,remaining=parser.parse_known_args()
    for flag in ['--transport-control','--r9-path-control','--task-value-control']:
        if flag in remaining and remaining[remaining.index(flag)+1]!='normal':
            raise ValueError('proposal control cannot combine other interventions')
    own=Path(__file__).resolve();stage=own.parents[2]
    import experiments.deformnext_retargeting as package
    # Native lazy imports can change sys.path. Preserve the explicit extension
    # instead of letting a namespace-package path recalculate and discard it.
    package.__path__=[*list(package.__path__),str(stage/'experiments/deformnext_retargeting')]
    from experiments.deformnext_retargeting.orthogonal_utility_followups import require_source_contract
    from experiments.deformnext_retargeting.ray_proposal_controls import ray_proposal_control
    require_source_contract(local.frozen_source_contract,local.frozen_source_receipt,local.campaign_root)
    proof=json.loads(local.identity_proof.read_text());receipt=json.loads(local.identity_receipt.read_text())
    if (proof.get('status')!='PASS_ACTUAL_RAY_PROPOSAL_IDENTITY_UNION64_AND_CONTEXT_RESTORATION'
            or receipt.get('status')!='PASS' or receipt.get('local_equals_actual_file4_SHA256') is not True):
        raise ValueError('actual saved-model identity proof and actual file4 closure required')
    closed={row['relative_path']:row['sha256'] for row in receipt['files']}
    proof_key=str(local.identity_proof.relative_to(local.campaign_root.parent))
    if closed.get(proof_key)!=sha(local.identity_proof):
        raise ValueError('identity proof differs from closed actual file4 bytes')
    sources={**proof['source_SHA256'],str(own):sha(own)}
    if any(sha(path)!=digest for path,digest in sources.items()):
        raise ValueError('proposal control source differs from actual identity proof')
    output=Path(remaining[remaining.index('--output')+1]);result_path=output/'evaluation.json'
    if result_path.exists():raise FileExistsError('preserve previous evaluation')
    events=[];previous=sys.argv
    try:
        sys.argv=[str(stage/'experiments/deformnext_retargeting/evaluate_deterministic_checkpoint.py'),*remaining]
        with ray_proposal_control(local.proposal_mode,events):
            runpy.run_path(sys.argv[0],run_name='__main__')
    finally:sys.argv=previous
    result=json.loads(result_path.read_text())
    if (len(result['per_asset'])!=local.expected_assets or not events
            or sum(event['batch_assets'] for event in events)!=local.expected_assets
            or any(not event['same_query_support'] or not event['same_pooled_ray_queries'] or event['candidate_count']!=64 or not event['same_D1_scalar_utility_R_math']
                   or event['GT_read_by_control'] or event['optimizer_updates']!=0 for event in events)):
        raise ValueError('actual proposal intervention did not retain matching support/addresses/GT-free inference')
    if any(sha(path)!=digest for path,digest in sources.items()):
        raise ValueError('proposal control changed during evaluation')
    require_source_contract(local.frozen_source_contract,local.frozen_source_receipt,local.campaign_root)
    result.update(diagnostic_only=True,proposal_control=local.proposal_mode,
        proposal_control_events=events,candidate_proposal_source_SHA256=sources,
        saved_identity_proof=str(local.identity_proof),saved_identity_proof_SHA256=sha(local.identity_proof),
        inference_only_counterfactual=True,
        proposal_scope='SavedSS480/camera/RGB/GT/noise/native rayquery/support/64budget retained;onlyGTfreeconstituentpatchcandidateproposals replacepooledqueryproposals. D1/scalar/utility/R math unchanged. Utility checkpoint is explicit;E7headwastrainedonoriginalbank. Inferenceonlycounterfactual,notfreshSStrainingoradoption.')
    result['deterministic_evaluation']['process_local_runtime_forward_intervention']=True
    temporary=result_path.with_suffix('.readout-metadata.tmp')
    with temporary.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    os.replace(temporary,result_path)
    print(json.dumps(dict(status='PASS_DETERMINISTIC_SAVED_RAY_PROPOSAL_CONTROL',mode=local.proposal_mode,
        assets=local.expected_assets,metrics=result['quick_metrics'],actual_control_calls=len(events))),flush=True)


if __name__=='__main__':
    main()
