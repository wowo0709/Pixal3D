"""Pinned SS evaluation with an explicitly registered stage-local anchor composer."""
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
    parser.add_argument('--identity-proof',type=Path,required=True)
    parser.add_argument('--identity-receipt',type=Path,required=True)
    parser.add_argument('--campaign-root',type=Path,required=True)
    parser.add_argument('--frozen-source-contract',type=Path,required=True)
    parser.add_argument('--frozen-source-receipt',type=Path,required=True)
    parser.add_argument('--expected-assets',type=int,default=32)
    local,remaining=parser.parse_known_args()
    from pixal3d import trainers
    from .anchor_ray_runtime import AnchorRayCompositionTrainer
    from .orthogonal_utility_followups import require_source_contract
    require_source_contract(local.frozen_source_contract,local.frozen_source_receipt,local.campaign_root)
    proof=json.loads(local.identity_proof.read_text());declaration_path=local.identity_proof.parent/'declaration.json'
    declaration=json.loads(declaration_path.read_text());receipt=json.loads(local.identity_receipt.read_text())
    assert proof['status']=='PASS_ACTUAL_OLD_R_NEAREST_ZERO_GATE_CONDITION_IDENTITY_AND_PROJECTION_GRADIENT'
    assert receipt['status']=='PASS' and receipt['local_equals_actual_file4_SHA256']
    closed={row['relative_path']:row['sha256'] for row in receipt['files']}
    for path in [local.identity_proof,declaration_path,Path(proof['initializer_checkpoint'])]:
        assert closed[str(path.relative_to(local.campaign_root.parent))]==sha(path)
    config_path=Path(remaining[remaining.index('--arm-config')+1]);config=json.loads(config_path.read_text())
    assert sha(config_path)==declaration['config_SHA256']
    assert config['trainer']['name']=='AnchorRayCompositionTrainer'
    assert config['experiment']['conditioner_family']=='anchor-ray-topk-residual-v1'
    assert config['dataset']['args']['task_matching_features'] and config['validation']['dataset_args']['task_matching_features']
    sources=dict(proof['source_SHA256']);sources[str(Path(__file__).resolve())]=sha(__file__)
    assert all(sha(path)==digest for path,digest in sources.items())
    output=Path(remaining[remaining.index('--output')+1]);result_path=output/'evaluation.json'
    assert not result_path.exists()
    captured=[]
    class EvaluatedAnchorTrainer(AnchorRayCompositionTrainer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);captured.append(self)
    # This stage-local entry avoids modifying the six-arm study's frozen registry.
    previous=trainers.__dict__.get('AnchorRayCompositionTrainer')
    trainers.AnchorRayCompositionTrainer=EvaluatedAnchorTrainer
    original_arguments=sys.argv
    try:
        entry=Path(__file__).with_name('evaluate_deterministic_checkpoint.py')
        sys.argv=[str(entry),*remaining];runpy.run_path(str(entry),run_name='__main__')
    finally:
        sys.argv=original_arguments
        if previous is None:delattr(trainers,'AnchorRayCompositionTrainer')
        else:trainers.AnchorRayCompositionTrainer=previous
    assert len(captured)==1
    trainer=captured[0];module=trainer.deform_conditioner
    result=json.loads(result_path.read_text());assert len(result['per_asset'])==local.expected_assets
    assert result['optimizer_updates_during_evaluation']==0 and module.projection_mode=='nearest'
    assert len(trainer.model_params)==6 and all(not p.requires_grad for p in trainer.models['denoiser'].parameters())
    assert all(sha(path)==digest for path,digest in sources.items())
    require_source_contract(local.frozen_source_contract,local.frozen_source_receipt,local.campaign_root)
    result.update(anchor_ray_composition=dict(protocol='anchor-ray-composition-v1',
        actual_gate=float(module.ray_mix.detach()),original_nearest_carrier=True,
        trainable_composer_tensors=6,SS_and_R_frozen=True,physical_deformation_query_channels=0,
        actual_trainer_class=type(trainer).__name__,source_SHA256=sources,
        source_registration='process-local;frozen registry file unchanged',
        initialization_identity_proof=str(local.identity_proof),initializer_checkpoint_SHA256=proof['initializer_checkpoint_SHA256'],
        SS_additional_optimizer_updates=0,GT_in_forward=False))
    result['deterministic_evaluation']['process_local_runtime_constructor_intervention']=True
    temporary=result_path.with_suffix('.anchor-provenance.tmp')
    with temporary.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    os.replace(temporary,result_path)
    print(json.dumps(dict(status='PASS_ACTUAL_ANCHOR_RAY_SS_EVALUATION',assets=local.expected_assets,
        gate=float(module.ray_mix.detach()),metrics=result['quick_metrics'])),flush=True)


if __name__=='__main__':main()
