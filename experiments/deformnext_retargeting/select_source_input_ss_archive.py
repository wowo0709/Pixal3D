"""Select a prescribed DEV32 subset of actual native128 SS archive bytes.

This is a CPU subset operation, not a second inference/replay. It preserves
native predicted occupancy as the downstream Shape input; GT is scoring only.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .cascade_view_roles import file_sha256
from .evaluate_source_input_joint_archive import validate_captured
from .quick_metrics import binary_metrics
from .ss_pooled_support_metrics import artifact_support_metrics


def read_artifact(path,entry,record):
    if file_sha256(path)!=entry['artifact_SHA256']:
        raise ValueError('actual native SS artifact bytes changed')
    with np.load(path)as z:
        if (str(z['sample_id'])!=record['sample_id'] or int(z['seed'])!=record['seed']
                or z['occupancy_shape'].tolist()!=[64,64,64]):
            raise ValueError('native SS artifact sample/noise/shape differs')
        values=[]
        for key,hashkey in [('prediction_occupancy_packed','prediction_voxels_SHA256'),
                            ('target_occupancy_packed','target_voxels_SHA256')]:
            packed=z[key]
            if packed.dtype!=np.uint8 or packed.shape!=(64**3//8,):
                raise ValueError('complete actual packed SS64 occupancy required')
            if hashlib.sha256(packed.tobytes()).hexdigest()!=entry[hashkey]:
                raise ValueError('actual native prediction/GT grid hash differs')
            values.append(np.unpackbits(packed).reshape(64,64,64).astype(bool))
        pred,gt=values
    counts=dict(tp=int((pred & gt).sum()),fp=int((pred & ~gt).sum()),fn=int((~pred & gt).sum()))
    if counts!=entry['occupancy_counts']or any(counts[k]!=record[k]for k in counts):
        raise ValueError('native SS counts and actual archive prediction differ')
    return pred,gt


def select(parent_path,panel_path,seeds_path,output):
    parent_path,panel_path,seeds_path,output=map(Path,[parent_path,panel_path,seeds_path,output])
    if output.exists():raise FileExistsError('fresh DEV32 archive namespace required')
    bound={str(p):file_sha256(p)for p in [parent_path,panel_path,seeds_path]}
    parent=json.loads(parent_path.read_text());archive=parent['native_SS_artifacts']
    if (archive['status']!='PASS_ACTUAL_JOINT_SS128_SINGLE_EVALUATION_ARCHIVE'
            or archive['scope']!='development-validation128' or archive['archived_asset_count']!=128):
        raise ValueError('actual native development128 archive required; no final-test selection')
    validate_captured(parent,archive['artifacts'])
    ids=json.loads(panel_path.read_text())['sample_ids'];seeds=json.loads(seeds_path.read_text())
    if len(ids)!=32 or len(set(ids))!=32:
        raise ValueError('exact prescribed32 unique asset IDs required')
    by={r['sample_id']:r for r in parent['per_asset']};entries={r['sample_id']:r for r in archive['artifacts']}
    if not set(ids)<=set(by)or any(type(seeds.get(sid))is not int or seeds[sid]!=by[sid]['seed']for sid in ids):
        raise ValueError('prescribed DEV32 assets/noise must match actual native128 predictions')
    prepared=[];predictions=[];targets=[];parent_root=parent_path.parent.resolve()
    for sid in ids:
        entry=copy.deepcopy(entries[sid]);relative=Path(entry['artifact'])
        if relative.is_absolute()or'..'in relative.parts:
            raise ValueError('owned relative SS artifact path required')
        source=(parent_root/relative).resolve()
        if not source.is_relative_to(parent_root):raise ValueError('SS archive path escaped parent output')
        pred,gt=read_artifact(source,entry,by[sid]);predictions.append(pred);targets.append(gt)
        entry['native_SS32_support']=artifact_support_metrics(source)
        prepared.append((source,relative,entry))
    output.mkdir(parents=True,exist_ok=False)
    for source,relative,entry in prepared:
        destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        if file_sha256(destination)!=entry['artifact_SHA256']:
            raise ValueError('DEV32 copied prediction bytes differ from parent')
    if any(file_sha256(p)!=digest for p,digest in bound.items()):
        raise ValueError('parent evaluation/panel/noise selection changed')
    result=copy.deepcopy(parent);result.update(per_asset=[copy.deepcopy(by[sid])for sid in ids],quick_panel_ids=ids,
        panel_file=str(panel_path),seed_map_file=str(seeds_path),
        quick_metrics=binary_metrics(torch.from_numpy(np.stack(predictions)),torch.from_numpy(np.stack(targets))))
    result['originating_native128_execution']=result.pop('deterministic_evaluation',None)
    result['evaluation_execution']='CPU subset of existing native128 predictions; no new inference'
    result['native_SS_artifacts']=dict(status='PASS_ACTUAL_JOINT_SS32_CPU_SUBSET_OF_NATIVE128_ARCHIVE',
        scope='development-validation32',artifacts=[entry for _,_,entry in prepared],archived_asset_count=32,
        original_parent_evaluation=str(parent_path),original_parent_SHA256=bound[str(parent_path)],
        original_parent_asset_count=128,prescribed_panel=str(panel_path),prescribed_panel_SHA256=bound[str(panel_path)],
        parent_prediction_latent_and_occupancy_bytes_exact=True,actual_counts_GTnoise_and_grid_hashes_verified=True,
        subset_voxel_metrics_recomputed_without_reusing_parent128_aggregate=True,
        additional_neural_forward_or_inference_or_optimizer_updates=0,
        selection_without_quality_filtering=True,empty_predictions_preserved_without_GT_fallback=True,
        predicted_SS_is_available_for_full_cascade=True,GT_SS_is_for_scoring_only=True,
        native64_to32_shape_support_recorded=True,full_cascade_generated=False,
        model_adopted=False,full_goal_complete=False)
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-evaluation',type=Path,required=True)
    parser.add_argument('--panel-file',type=Path,required=True)
    parser.add_argument('--seed-map-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=select(args.parent_evaluation,args.panel_file,args.seed_map_file,args.output)
    print(json.dumps(dict(status=result['native_SS_artifacts']['status'],assets=32,
        additional_neural_inference=0,metrics=result['quick_metrics'])),flush=True)


if __name__=='__main__':main()
