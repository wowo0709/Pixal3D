"""Prepare the learned-matcher retargeting follow-up after fixed-final head selection."""
import argparse
import copy
import json
from pathlib import Path

from .matching_v2_finalize import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();out=a.root/'retarget-native-v3'
    head=json.loads((a.root/'selected-head.json').read_text())
    if sha(Path(head['head']))!=head['head_sha256']:raise ValueError('selected head changed')
    contract={**head,'evidence_soft_dtype':'float32','projection':'native-bilinear-support-v3',
        'purpose':'P2 retargeting/interface follow-up; raw matching-v2 gate remains HOLD',
        'confidence_use':'disabled for P1-to-P2 matcher comparison; no untrained confidence fallback'}
    matches=out/'learned-matches';matches.mkdir(exist_ok=True)
    (matches/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    cfg=copy.deepcopy(json.loads((out/'configs/p1-native-grid.json').read_text()))
    name='p2-matcher-grid';run=out/'runs_ss64'/name
    cfg.update(default_output_dir=str(run),output_dir=str(run),load_dir=str(run),master_port='29932',
        wandb_id='ss-native-v3-p2',wandb_name='SS P2 learned matcher native-grid retargeting')
    for args in [cfg['dataset']['args'],cfg['validation']['dataset_args']]:
        args.update(match_root=str(matches),expected_q2c_sha256=head['head_sha256'])
    cfg['experiment'].update(arm=name,head_sha256=head['head_sha256'],q2c_checkpoint=head['head'],
        source_comparison='P1 same consumer; calibrated learned matcher bundle replaces R9; NULL rejection OFF',
        reference_attention_preserved=False,release_status='pending-cache-audit-native-proof-memory-probe')
    cfg['trainer']['args']['deform_conditioner'].update(reject=False,reject_threshold=0.)
    # Probe the largest per-rank batch independently, retaining global96.
    # Both optimizer/sampler states still start from the original base.
    per_rank=96//cfg['num_gpus']
    cfg['trainer']['args'].update(batch_size_per_gpu=per_rank,batch_split=1,max_steps=480)
    cfg['experiment'].update(microbatch_per_gpu=per_rank,world_size=cfg['num_gpus'])
    (out/'configs'/f'{name}.json').write_text(json.dumps(cfg,indent=2)+'\n')
    result={'status':'PASS','head_sha256':head['head_sha256'],'head_arm':head['arm'],
        'fixed_final_epoch':7,'calibration_selected_temperature':head['temperature'],
        'rejection':False,'cache_root':str(matches),'config':str(out/'configs'/f'{name}.json')}
    (out/'p2-config-ready.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
