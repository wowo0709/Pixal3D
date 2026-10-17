"""CPU-only full-population audit of the two inference query-context stores."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--expected-assets',type=int,required=True)
    a=p.parse_args();started=time.monotonic();stage=a.root/'retarget-native-v3';output=stage/'query-study'
    rows=[json.loads(l) for l in a.manifest.read_text().splitlines() if l]
    if len(rows)!=a.expected_assets or len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('query context audit population differs')
    from pixal3d.modules.r9_matching import MATCHER_CONTRACT,ATTENTION_CONTRACT
    reference_contract=json.loads((stage/'reference-matches/contract.json').read_text())
    learned_bytes=(stage/'learned-matches/contract.json').read_bytes()
    learned_contract=json.loads(learned_bytes);learned_hash=hashlib.sha256(learned_bytes).hexdigest()
    for i,row in enumerate(rows):
        sid=row['sample_id'];fingerprint=row['conditioning_input_sha256']
        with np.load(a.root.parent/'rgb_v1/matches_r9_v5'/(sid+'.npz')) as z:
            if (str(z['schema'])!='deformnext-match-r9-v5' or str(z['q2c_sha256'])!=reference_contract['head_sha256']
                or str(z['conditioning_input_sha256'])!=fingerprint or str(z['aggregation'])!='global-categorical-v1'
                or str(z['proposal_policy'])!='GLOBAL64_VALID_OR_NATIVE' or int(z['topk'])!=64
                or json.loads(str(z['matcher_contract']))!=MATCHER_CONTRACT
                or json.loads(str(z['attention_contract']))!=ATTENTION_CONTRACT):
                raise ValueError('reference query context contract differs: '+sid)
            value=z['deformation']
            if value.shape!=(1024,2048) or not np.isfinite(value).all():raise ValueError('invalid reference RAW2048: '+sid)
        with np.load(stage/'learned-matches'/(sid+'.npz')) as z:
            if (str(z['schema'])!='deformnext-matching-v2-cache-v1' or str(z['contract_sha256'])!=learned_hash
                or str(z['conditioning_input_sha256'])!=fingerprint or str(z['sample_id'])!=sid or str(z['split'])!=row['split']):
                raise ValueError('learned query context identity differs: '+sid)
            value=z['deformation']
            if value.shape!=(1024,2048) or not np.isfinite(value).all():raise ValueError('invalid learned RAW2048: '+sid)
        if i%500==0:print(json.dumps({'checked':i+1,'total':len(rows)}),flush=True)
    result={'status':'PASS','assets':len(rows),'splits':dict(collections.Counter(r['split'] for r in rows)),
        'stores_checked':2,'context_shape':[1024,2048],'GT_inputs':False,
        'reference_head_sha256':reference_contract['head_sha256'],'learned_head_sha256':learned_contract['head_sha256'],
        'scope':'all field shapes/finite/head/attention/RGB identity; not a CUDA or generation performance test',
        'seconds':time.monotonic()-started}
    output.mkdir(exist_ok=True);(output/'context-population-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
