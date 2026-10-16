"""Version an existing aligned DeformNeXt manifest with explicit RGB inputs.

Latents/cameras are retained verbatim. Image-dependent caches get new paths;
the depth experiment's outputs and caches cannot be resumed as RGB results.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np

SCHEMA = 'deformnext-edited-rgb-input-v1'


def image_fingerprint(row, resolve):
    record={key:row[key] for key in ('sample_id','camera_angle_x','transform_matrix')}
    record['schema']=SCHEMA
    record['images']=[]
    for path in [row['target_anchor'], *row['source_images']]:
        with resolve(path).open('rb') as stream:
            record['images'].append(hashlib.file_digest(stream,'sha256').hexdigest())
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()


def convert(row, published_root, frames):
    result=copy.deepcopy(row)
    target=row['target_rgb_supervision']
    if row.get('target_anchor_kind')!='geometry_depth':
        raise ValueError('migration requires the explicit legacy depth manifest')
    if target in row['source_images']:
        raise ValueError('edited RGB target cannot be one of the M0 source files')
    if len(frames)<5 or frames[0].get('mesh_state')!='M1' or any(frames[i].get('mesh_state')!='M0' for i in (1,2,3,4)):
        raise ValueError('requires edited anchor and four original RGB cameras')
    if frames[0]['camera_id']!=frames[1]['camera_id'] or len({frames[i]['camera_id'] for i in (1,2,3,4)})!=4:
        raise ValueError('source-only requires four distinct cameras including the original target camera')
    if not np.allclose(row['transform_matrix'][0],frames[0]['transform_matrix']) or not np.allclose(frames[0]['transform_matrix'],frames[1]['transform_matrix']):
        raise ValueError('edited/original anchor camera differs from the validated latent frame')
    if not np.isclose(frames[0]['camera_angle_x'],frames[1]['camera_angle_x']):
        raise ValueError('edited/original anchor FOV differs')
    render=Path(target).parent
    result.update(conditioning_input_schema=SCHEMA, target_anchor=target,
                  target_anchor_kind='rgb',target_rgb_input=target,
                  target_depth_auxiliary=row['target_anchor'],
                  source_anchor_rgb=str(render/frames[1]['file_path']),
                  source_anchor_transform_matrix=frames[1]['transform_matrix'],
                  source_anchor_camera_angle_x=frames[1]['camera_angle_x'],
                  source_images=[str(render/frames[i]['file_path']) for i in (2,3,4)],
                  camera_angle_x=[frames[i]['camera_angle_x'] for i in (0,2,3,4)],
                  transform_matrix=[frames[i]['transform_matrix'] for i in (0,2,3,4)],
                  conditioning_camera_ids=[frames[i]['camera_id'] for i in (0,2,3,4)],
                  input_target_camera_id=frames[0]['camera_id'])
    # Raw M1 render naming is part of the published DeformNeXt export.
    # Exclude the edited input camera from the held-out target-view list.
    input_stem=f"view_{frames[0]['camera_id']:02d}_M1"
    result['evaluation_target_images']=[p for p in row.get('evaluation_target_images',[]) if Path(p).stem!=input_stem]
    for field,folder in [('transport_cache','transport'),('transport_cache_pbr','transport_pbr')]:
        result['legacy_'+field]=row[field]
        result[field]=str(published_root/folder/f"{row['sample_id']}.npz")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--published-root',type=Path,required=True)
    p.add_argument('--path-map',action='append',default=[])
    p.add_argument('--splits',nargs='+',choices=['train','validation'],default=['validation','train'])
    args=p.parse_args();maps=[x.split('=',1) for x in args.path_map]
    def resolve(path):
        for a,b in maps:
            if path==a or path.startswith(a.rstrip('/')+'/'):return Path(b+path[len(a):])
        return Path(path)
    args.output_root.mkdir(parents=True,exist_ok=True)
    report={'schema':SCHEMA,'target_modality':'rgb','splits':{},'preserved_latents':True}
    for split in args.splits:
        rows=[json.loads(x) for x in (args.manifest_root/f'{split}.jsonl').read_text().splitlines()]
        converted=[]
        for old in rows:
            transform_path=resolve(str(Path(old['target_rgb_supervision']).parent/'transforms.json'))
            transforms=json.loads(transform_path.read_text());frames=transforms['frames']
            row=convert(old,args.published_root,frames)
            for path in [row['target_anchor'],row['source_anchor_rgb'],*row['source_images']]:
                with Image.open(resolve(path)) as im:
                    if im.mode!='RGBA' or im.size!=(512,512):
                        raise ValueError(f'expected original 512px RGBA render: {path}')
                    im.verify()
            row['conditioning_input_sha256']=image_fingerprint(row,resolve)
            converted.append(row)
        target=args.output_root/f'{split}.jsonl';tmp=target.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r)+'\n' for r in converted));os.replace(tmp,target)
        report['splits'][split]=len(converted)
        print(json.dumps({'split':split,'rgb_inputs_verified':len(converted)}),flush=True)
    (args.output_root/'rgb_input_audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
