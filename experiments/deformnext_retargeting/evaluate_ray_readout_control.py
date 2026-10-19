"""Run a pinned deterministic SS evaluator with an explicit GT-free readout control."""
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
    parser.add_argument('--readout-mode',choices=['prior-only','patch-score-pool'],required=True)
    parser.add_argument('--identity-proof',type=Path,required=True)
    parser.add_argument('--identity-receipt',type=Path,required=True)
    parser.add_argument('--frozen-source-contract',type=Path,required=True)
    parser.add_argument('--frozen-source-receipt',type=Path,required=True)
    parser.add_argument('--campaign-root',type=Path,required=True)
    parser.add_argument('--expected-assets',type=int,default=32)
    local,remaining=parser.parse_known_args()
    own=Path(__file__).resolve();stage=own.parents[2]
    import experiments.deformnext_retargeting as package
    # Native lazy imports can change sys.path. Preserve the explicit extension
    # instead of letting a namespace-package path recalculate and discard it.
    package.__path__=[*list(package.__path__),str(stage/'experiments/deformnext_retargeting')]
    from experiments.deformnext_retargeting.orthogonal_utility_followups import require_source_contract
    from experiments.deformnext_retargeting.ray_readout_controls import ray_readout_control
    require_source_contract(local.frozen_source_contract,local.frozen_source_receipt,local.campaign_root)
    proof=json.loads(local.identity_proof.read_text());receipt=json.loads(local.identity_receipt.read_text())
    if (proof.get('status')!='PASS_EXACT_IDENTITY_RAY_READOUT_RECONSTRUCTION_AND_CONTEXT_RESTORATION'
            or receipt.get('status')!='PASS' or receipt.get('local_equals_actual_file4_SHA256') is not True):
        raise ValueError('actual saved-model identity proof and actual file4 closure required')
    closed={row['relative_path']:row['sha256'] for row in receipt['files']}
    proof_key=str(local.identity_proof.relative_to(local.campaign_root.parent))
    if closed.get(proof_key)!=sha(local.identity_proof):
        raise ValueError('identity proof differs from closed actual file4 bytes')
    sources={**proof['source_SHA256'],str(own):sha(own)}
    if any(sha(path)!=digest for path,digest in sources.items()):
        raise ValueError('readout control source differs from actual identity proof')
    output=Path(remaining[remaining.index('--output')+1]);result_path=output/'evaluation.json'
    if result_path.exists():raise FileExistsError('preserve previous evaluation')
    events=[];previous=sys.argv
    try:
        sys.argv=[str(stage/'experiments/deformnext_retargeting/evaluate_deterministic_checkpoint.py'),*remaining]
        with ray_readout_control(local.readout_mode,events):
            runpy.run_path(sys.argv[0],run_name='__main__')
    finally:sys.argv=previous
    result=json.loads(result_path.read_text())
    if (len(result['per_asset'])!=local.expected_assets or not events
            or sum(event['batch_assets'] for event in events)!=local.expected_assets
            or any(not event['same_candidate_addresses'] or not event['same_query_support']
                   or event['GT_read_by_control'] or event['optimizer_updates']!=0 for event in events)):
        raise ValueError('actual intervention did not retain support/addresses/GT-free inference')
    if any(sha(path)!=digest for path,digest in sources.items()):
        raise ValueError('readout control changed during evaluation')
    require_source_contract(local.frozen_source_contract,local.frozen_source_receipt,local.campaign_root)
    result.update(diagnostic_only=True,ray_readout_control=local.readout_mode,
        ray_readout_control_events=events,readout_source_SHA256=sources,
        saved_identity_proof=str(local.identity_proof),saved_identity_proof_SHA256=sha(local.identity_proof),
        inference_only_counterfactual=True,
        readout_scope='SavedFull480 weights/semantic64/camera/RGB/GT/noise unchanged;only GT-free candidate probability readout replaced and same trained R recomputed. No retraining or adopted-model claim.')
    result['deterministic_evaluation']['process_local_runtime_forward_intervention']=True
    temporary=result_path.with_suffix('.readout-metadata.tmp')
    with temporary.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    os.replace(temporary,result_path)
    print(json.dumps(dict(status='PASS_DETERMINISTIC_SAVED_RAY_READOUT_CONTROL',mode=local.readout_mode,
        assets=local.expected_assets,metrics=result['quick_metrics'],actual_control_calls=len(events))),flush=True)


if __name__=='__main__':
    main()
