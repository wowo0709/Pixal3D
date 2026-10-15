"""Check listwise-score shift invariance against absolute view confidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
    torch.set_num_threads(4)
    rows=[json.loads(line) for line in (args.data_root/'manifests_world_v4/validation.jsonl').read_text().splitlines()]
    selected=[next(row for row in rows if row['category']==category) for category in ['Axe','Bed','Bottle','Scissors']]
    result=[]
    with torch.inference_mode():
        for stage in ['ss64','pbr1024']:
            cfg=json.loads((args.config_root/f'{stage}-b3.json').read_text())
            model=DeformationAwareConditioner(**cfg['trainer']['args']['deform_conditioner']).cuda().eval()
            for row in selected:
                field='transport_cache' if stage=='ss64' else 'transport_cache_pbr'
                with np.load('/root/dev/.cache/deformnext-file4'+row[field]) as z:
                    values=torch.from_numpy(z['source_values'].copy()).cuda()[None]
                with np.load('/root/dev/.cache/deformnext-file4'+row['transport_cache']) as z:
                    valid=torch.from_numpy(z['source_valid'].copy()).cuda()[None].bool()
                with np.load(args.data_root/'matches'/(row['sample_id']+'.npz')) as z:
                    indices=torch.from_numpy(z['semantic_indices'].astype(np.int64)).cuda()[None]
                    logits=torch.from_numpy(z['matching_logits'].copy()).cuda()[None]
                    cvalid=torch.from_numpy(z['candidate_valid'].copy()).cuda()[None].bool()
                def call(shift):
                    return model(source_values=values,source_valid=valid,semantic_indices=indices,
                                 matching_logits=logits+shift,candidate_valid=cvalid)
                baseline=call(0)
                for shift in [-10.,10.]:
                    other=call(shift)
                    result.append({'stage':stage,'category':row['category'],'logit_shift':shift,
                                   'within_view_weight_max_abs_change':float((baseline.candidate_weights-other.candidate_weights).abs().max()),
                                   'mean_view_weight_l1_change':float((baseline.view_weights-other.view_weights).abs().sum(-1).mean()),
                                   'evidence_relative_rmse':float((baseline.evidence-other.evidence).square().mean().sqrt()/baseline.evidence.square().mean().sqrt())})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'diagnostic_only':True,'results':result},indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
