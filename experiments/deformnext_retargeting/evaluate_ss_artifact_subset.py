"""Archive a32-asset prefix of a pinned native SS evaluation for the cascade.

The original evaluator/sampler runs unchanged. Actual predicted and target
decodes are captured once; every prefix record and archived occupancy count
must reproduce the saved reference before releasing downstream generation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_subset_reference(actual,reference,captured,*,count=32):
    if (actual.get('stage')!='ss64' or reference.get('stage')!='ss64'
            or actual.get('optimizer_updates_during_evaluation')!=0
            or reference.get('optimizer_updates_during_evaluation')!=0
            or len(actual['per_asset'])!=count or len(captured)!=count
            or len(reference['per_asset'])<count
            or actual['per_asset']!=reference['per_asset'][:count]):
        raise ValueError('all actual archived prefix records must exactly reproduce the pinned reference')
    for field in ['checkpoint','conditioner_checkpoint','arm_config','conditioning_mode',
                  'target_modality','input_states','input_camera_ids','manifest_sha256',
                  'generation_batch_size','sampling_steps','guidance']:
        if actual[field]!=reference[field]:
            raise ValueError('archive replay changed reference '+field)
    for artifact,record in zip(captured,actual['per_asset']):
        if (artifact['sample_id']!=record['sample_id'] or artifact['seed']!=record['seed']
                or artifact['target_voxels_SHA256']!=record['target_voxels_sha256']
                or any(artifact['occupancy_counts'][key]!=record[key] for key in ['tp','fp','fn'])):
            raise ValueError('archived predicted/GT occupancy or sample/noise identity differs from actual record')


@contextmanager
def capture_ss_artifacts(quick,output,captured):
    original=quick._ss_metrics
    def archived(trainer,dataset,indices,config):
        original_decode=dataset.decode_latent;tensors=[]
        def decode(latent,*args,**kwargs):
            decoded=original_decode(latent,*args,**kwargs)
            tensors.append((latent.detach().float().cpu().numpy(),decoded.detach().float().cpu().numpy()))
            return decoded
        dataset.decode_latent=decode
        try:
            result=original(trainer,dataset,indices,config)
        finally:
            dataset.decode_latent=original_decode
        if len(tensors)!=1:
            raise ValueError('one native concatenated predicted/GT SS decode required')
        latent,decoded=tensors[0];count=len(indices)
        if len(latent)!=2*count or decoded.shape!=(2*count,1,64,64,64):
            raise ValueError('explicit native predicted/GT SS64 pairs required')
        folder=output/'ss-artifacts';folder.mkdir(exist_ok=False)
        for i,(index,seed) in enumerate(zip(indices,quick.panel_seeds(dataset,indices,config))):
            row=dataset.rows[index];pred=decoded[i,0]>0;gt=decoded[count+i,0]>0
            packed_pred=np.packbits(pred.reshape(-1));packed_gt=np.packbits(gt.reshape(-1))
            path=folder/(hashlib.sha256(row['sample_id'].encode()).hexdigest()+'.npz')
            np.savez_compressed(path,prediction_latent=latent[i],target_latent=latent[count+i],
                prediction_occupancy_packed=packed_pred,target_occupancy_packed=packed_gt,
                occupancy_shape=np.array([64,64,64],dtype=np.int32),sample_id=np.asarray(row['sample_id']),seed=np.int64(seed))
            captured.append(dict(sample_id=row['sample_id'],seed=seed,artifact=str(path.relative_to(output)),artifact_SHA256=sha(path),
                prediction_voxels_SHA256=hashlib.sha256(packed_pred.tobytes()).hexdigest(),
                target_voxels_SHA256=hashlib.sha256(packed_gt.tobytes()).hexdigest(),
                occupancy_counts=dict(tp=int((pred & gt).sum()),fp=int((pred & ~gt).sum()),fn=int((~pred & gt).sum()))))
        return result
    quick._ss_metrics=archived
    try:
        yield
    finally:
        quick._ss_metrics=original


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--evaluation-module',required=True,choices=['evaluate_deterministic_checkpoint',
        'evaluate_native_prior_source_ray','member_posterior_runtime'])
    parser.add_argument('--reference-evaluation',type=Path,required=True)
    local,remaining=parser.parse_known_args()
    from . import quick_metrics as quick
    output=Path(remaining[remaining.index('--output')+1])
    if output.exists():raise FileExistsError('fresh archive/evaluation namespace required')
    reference=json.loads(local.reference_evaluation.read_text())
    entry=importlib.import_module('.'+local.evaluation_module,__package__)
    bound={str(Path(p).resolve()):sha(p) for p in [__file__,quick.__file__,entry.__file__,local.reference_evaluation]}
    captured=[];previous=sys.argv
    try:
        with capture_ss_artifacts(quick,output,captured):
            sys.argv=[entry.__file__,*remaining];entry.main()
    finally:
        sys.argv=previous
    path=output/'evaluation.json';actual=json.loads(path.read_text())
    validate_subset_reference(actual,reference,captured)
    if any(sha(p)!=digest for p,digest in bound.items()):raise ValueError('archive source or actual reference changed')
    actual['native_SS_artifacts']=dict(status='PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_EXACT_REFERENCE_SUBSET_REPLAY',
        original_prefix32_all_records_and_occupancy_counts_exact=True,artifacts=captured,
        reference_evaluation=str(local.reference_evaluation),reference_SHA256=sha(local.reference_evaluation),source_SHA256=bound,
        reference_asset_count=len(reference['per_asset']),archived_asset_count=32,
        reference_full_panel_quick_metrics_not_compared_to_subset=True,
        predicted_SS_is_available_for_full_cascade=True,GT_SS_is_for_scoring_only=True,full_cascade_generated=False,optimizer_updates=0)
    path.write_text(json.dumps(actual,indent=2)+'\n')
    print(json.dumps(dict(status=actual['native_SS_artifacts']['status'],assets=len(captured))),flush=True)


if __name__=='__main__':main()
