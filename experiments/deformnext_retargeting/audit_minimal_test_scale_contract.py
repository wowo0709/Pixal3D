"""Audit the frozen TEST508 image-only scale contract in both cascade arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path:Path)->str:
    state=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            state.update(block)
    return state.hexdigest()


def rows(path:Path)->list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    campaign=json.loads(args.campaign.read_text())
    pairs={};paths=set()
    for entry in campaign['plans']:
        q=json.loads(Path(entry['shape_plan']).read_text())
        sid=entry['sample_id'];arm=entry['arm']
        if (sid!=q['row']['sample_id'] or arm not in {'V1','SELECTED'}
                or q['object_scale_policy']!='image-estimated-v1'
                or q['M1_derived_scale_supplied_to_generation'] is not False
                or q['input_states']!=['M1','M0','M0','M0']
                or len(q['RGB_paths'])!=4):
            raise ValueError('frozen input/scale policy differs: '+sid+'/'+arm)
        if arm in pairs.setdefault(sid,{}):raise ValueError('duplicate arm: '+sid)
        pairs[sid][arm]=q
        paths.add(q['image_scale_predictions'])
    if len(pairs)!=508 or any(set(v)!={'V1','SELECTED'} for v in pairs.values()) or len(paths)!=1:
        raise ValueError('exact paired TEST508 plans and one estimator output required')
    predictions_path=Path(next(iter(paths)));predictions=rows(predictions_path)
    pred={r['sample_id']:r for r in predictions}
    if len(pred)!=508 or len(set(pred))!=508 or set(pred)!=set(pairs):
        raise ValueError('image-only scale prediction population differs')
    digest=sha(predictions_path)
    completed={'V1':0,'SELECTED':0,'both':0};scale_checks=0;top1_checks=0
    for sid,arms in pairs.items():
        expected=float(pred[sid]['predicted_mesh_scale'])
        if (not expected>0 or pred[sid]['M0_or_M1_mesh_or_GT_scale_used_at_inference'] is not False):
            raise ValueError('object GT-scale leak: '+sid)
        v,s=arms['V1'],arms['SELECTED']
        for q in (v,s):
            if (abs(float(q['row']['mesh_scale'])-expected)>1e-10
                    or q['RGB_paths']!=v['RGB_paths']
                    or q['row']['conditioning_input_sha256']!=v['row']['conditioning_input_sha256']
                    or q['row']['conditioning_camera_ids']!=v['row']['conditioning_camera_ids']
                    or q['frozen_model_set']!=v['frozen_model_set']):
                raise ValueError('V1/Selected generation input differs: '+sid)
            receipt=Path(q['output'])/'archive-receipt.json'
            generation=Path(q['output'])/'generation.json'
            if receipt.is_file():
                if not generation.is_file():raise ValueError('archive without generation: '+sid)
                g=json.loads(generation.read_text())
                if (g['object_scale_policy']!='image-estimated-v1'
                        or abs(float(g['scale_used_in_generation'])-expected)>1e-10
                        or g['M1_derived_scale_supplied_to_generation'] is not False
                        or g['image_scale_predictions_sha256']!=digest):
                    raise ValueError('completed generation used other scale: '+sid)
                completed['V1' if q is v else 'SELECTED']+=1;scale_checks+=1
        if ((Path(v['output'])/'archive-receipt.json').is_file()
                and (Path(s['output'])/'archive-receipt.json').is_file()):
            completed['both']+=1
        selected_entry=next(e for e in campaign['plans'] if e['sample_id']==sid and e['arm']=='SELECTED')
        pbr=json.loads(Path(selected_entry['top1_pbr_plan']).read_text())
        if (pbr['frozen_TEST508_scale_provenance']!=str(predictions_path)
                or abs(float(pbr['row']['mesh_scale'])-expected)>1e-10
                or pbr['RGB_paths']!=v['RGB_paths']):
            raise ValueError('Top-1 PBR scale/input differs: '+sid)
        if (Path(pbr['output'])/'archive-receipt.json').is_file():
            top1_checks+=1
    result={'status':'PASS_TEST508_IMAGE_ONLY_SCALE_PLAN_AND_COMPLETED_GENERATION_AUDIT',
            'samples':508,'source_scale_predictions_sha256':digest,
            'completed':completed,'generation_scale_checks':scale_checks,
            'selected_top1_archive_checks':top1_checks,
            'all_cascade_results_complete':completed['both']==508 and top1_checks==508,
            'M0_M1_mesh_or_GT_scale_used_at_inference':False,
            'V1_SELECTED_equal_image_only_scale_and_RGB_camera_inputs':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix(args.output.suffix+'.tmp')
    temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(args.output)
    print(json.dumps(result))

if __name__=='__main__':main()
