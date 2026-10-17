"""Actual GPU integration proof from an epoch-one diagnostic head only."""
import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .matching_v2_finalize import cache_row,sha
from .matching_v2_head import MatchingV2Head


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--config-root',type=Path,required=True)
    p.add_argument('--file4-mirror',type=Path,required=True);p.add_argument('--panels',type=Path,required=True)
    p.add_argument('--projection-mode',choices=['nearest','native-bilinear'],default='nearest')
    a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_device(0)
    root=a.root/('smoke' if a.projection_mode=='nearest' else 'smoke-native-bilinear');(root/'matches').mkdir(parents=True,exist_ok=True);(root/'proposals').mkdir(exist_ok=True)
    row=json.loads((a.panels/'head_validation.jsonl').read_text().splitlines()[0]);sid=row['sample_id']
    head=a.root/'heads/h2/epoch-01.pt';state=torch.load(head,map_location='cpu',weights_only=True)
    model=MatchingV2Head().cuda().eval();model.load_state_dict(state['model'])
    contract={'head':str(head),'head_sha256':sha(head),'topk':state['provenance']['topk'],'temperature':.05,
              'reliability':{'slope':1.,'bias':0.,'threshold':0.},'diagnostic_only':True,'scope':'unrestricted epoch1 integration smoke; not production selection'}
    contractfile=root/'matches/contract.json';contractfile.write_text(json.dumps(contract,indent=2)+'\n')
    shutil.copyfile(a.root/'proposals'/(sid+'.npz'),root/'proposals'/(sid+'.npz'))
    arrays=cache_row(row,model,contract,root,a.file4_mirror)
    np.savez(root/'matches'/(sid+'.npz'),schema='deformnext-matching-v2-cache-v1',contract_sha256=sha(contractfile),
             sample_id=sid,split=row['split'],conditioning_input_sha256=row['conditioning_input_sha256'],**arrays)
    del model;torch.cuda.empty_cache()
    manifest=root/'validation.jsonl';manifest.write_text(json.dumps(row)+'\n')
    base=json.loads((a.config_root/'ss64-v1.json').read_text())
    cases=[('b2-no-reject','b2',False),('b3-reject','b3',True),('c1-no-reject','c1',False),('c2-reject','c2',True)]
    for name,arm,reject in cases:
        cfg=copy.deepcopy(base)
        cfg['validation']['roots_file']=str(manifest)
        cfg['validation']['dataset_args'].update(transport_arm=arm,transport_implementation='matching-v2',
            expected_q2c_sha256=contract['head_sha256'],match_root=str(root/'matches'),
            projection_root=str(a.root.parent/'projection_maps_world_v4'))
        cfg['trainer']['args']['deform_conditioner']={'implementation':'matching-v2','arm':arm,'reject':reject,
            'projection_mode':a.projection_mode,
            'reject_threshold':0.,**({'deformation_dim':2048} if arm=='c2' else {})}
        cfg['trainer']['args']['model_learning_rates']={'deform_conditioner':1e-4}
        cfg['experiment'].update(arm=name,head_sha256=contract['head_sha256'])
        config=root/(name+'.json');config.write_text(json.dumps(cfg,indent=2)+'\n')
        subprocess.run([sys.executable,'-u','-m','experiments.deformnext_retargeting.matching_v2_ss',
            '--mode','proof','--root',str(root),'--config',str(config)],check=True)
    (root/'passed.json').write_text(json.dumps({'status':'PASS','diagnostic_only':True,'head_sha256':contract['head_sha256'],
        'projection_mode':a.projection_mode,'cases':[name for name,_,_ in cases]},indent=2)+'\n')


if __name__=='__main__':main()
