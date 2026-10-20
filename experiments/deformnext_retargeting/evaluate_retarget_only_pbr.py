"""Read-only PBR evaluation of the physically smaller native Top1/R control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from .retarget_only_pbr_contract import validate_profile
from .task_ray_science_contract import closed_json,digest


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--R-native-profile',type=Path,required=True)
    p.add_argument('--R-native-profile-receipt',type=Path,required=True)
    p.add_argument('--campaign-root',type=Path,required=True)
    p.add_argument('--R-phase',choices=['initial','trained'],required=True)
    local,remaining=p.parse_known_args()
    proof=closed_json(local.R_native_profile,local.R_native_profile_receipt,local.campaign_root)
    validate_profile(proof,capacity=False)
    config_path=Path(remaining[remaining.index('--arm-config')+1])
    config=json.loads(config_path.read_text());cfg=config['trainer']['args']['deform_conditioner']
    if (config['trainer']['name']!='ImageConditionedProjSparseFlowMatchingCFGTrainer'
            or config['dataset']['name']!='DeformNeXtPBR1024'
            or config['validation']['dataset_name']!='DeformNeXtPBR1024'
            or (cfg['arm'],cfg['value_dim'],cfg['query_dim'])!=('c1',2048,1024)
            or 'deformation_dim' in cfg or cfg['candidate_aggregation']!='top1'):
        raise ValueError('physical Top1/R PBR3072 schema without extra channels required')
    checkpoint=Path(remaining[remaining.index('--conditioner-checkpoint')+1])
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if len(state)!=30 or any(k.startswith('ray_readout.') or 'deformation_projection' in k for k in state):
        raise ValueError('complete R30 state without attention or RAW projection required')
    if local.R_phase=='initial' and any(torch.count_nonzero(state[k]) for k in [
            'fusion_gate','retargeter.net.2.weight','retargeter.net.2.bias']):
        raise ValueError('untouched native V1 initial function required')
    from .frozen_projection_streaming import install_frozen_projection_streaming
    from . import evaluate_pbr_streaming as entry
    output=Path(remaining[remaining.index('--output')+1])
    bindings={str(Path(q).resolve()):digest(q) for q in [__file__,entry.__file__,config_path]}
    install_frozen_projection_streaming(max_assets=1,audit_directory=output/'projection-streaming-audit')
    previous=sys.argv
    try:
        sys.argv=[entry.__file__,*remaining];entry.main()
    finally:sys.argv=previous
    path=output/'evaluation.json';result=json.loads(path.read_text())
    if result['optimizer_updates_during_evaluation']!=0 or any(digest(q)!=s for q,s in bindings.items()):
        raise ValueError('read-only unchanged-source R PBR evaluation required')
    result['retarget_only_PBR']=dict(phase=local.R_phase,conditioner_state_tensors=30,
        physical_R_input_channels=3072,attention_and_RAW_projection_physically_absent=True,
        candidate_aggregation='matched-semantic-Top1-feature',query='edited-target-patch-ray',
        matching_tensors_frozen=24,optimizer_updates=0,source_SHA256=bindings,
        standard_GT_Shape_condition=True,source_preservation_metric=False,full_cascade=False)
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
