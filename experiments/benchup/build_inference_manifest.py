"""Create an explicitly provisional GT-free BenchUp conditioning manifest."""
from __future__ import annotations
import argparse,itertools,json,math
from pathlib import Path
import numpy as np
from .fit_source_orthographic_pilot import camera_basis
from experiments.deformnext_retargeting.prepare_rgb_manifest import image_fingerprint,SCHEMA


def angular_distance(a,b):return abs((a-b+math.pi)%(2*math.pi)-math.pi)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--qa',type=Path,required=True);p.add_argument('--recovery-qa',type=Path,required=True);p.add_argument('--surfaces',type=Path,required=True);p.add_argument('--source-png',type=Path,required=True);p.add_argument('--target-root',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(x) for x in a.index.read_text().splitlines()];qa=[json.loads(x) for x in a.qa.read_text().splitlines()];rec=[json.loads(x) for x in a.recovery_qa.read_text().splitlines()]
    if len(rows)!=109 or len(qa)!=180 or len(rec)!=25:raise ValueError('complete frozen BenchUp inputs required')
    by={}
    for r in qa+rec:
        key=(r['object_id'],r['view_name']);old=by.get(key)
        if old is None or r['best']['mask_iou']>old['best']['mask_iou']:by[key]=r
    if len(by)!=180:raise ValueError('duplicate or missing view QA')
    conv=np.array([[1.,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
    def getcam(obj,view):
        result=by[obj,view];b=result['best'];job=json.loads(Path(b['job']).read_text().splitlines()[b['job_index']]);fit=json.loads(Path(b['fit']).read_text())
        with np.load(a.surfaces/f'{obj}.npz') as data:bounds=data['bounds'].copy()
        extent=float(np.max(bounds[1]-bounds[0]));center=bounds.mean(0);n=np.eye(4);n[:3,:3]/=extent;n[:3,3]=-center/extent;canonical=conv@n@conv.T
        transform=canonical@np.linalg.inv(np.asarray(job['to_world']));s=np.linalg.norm(transform[:3,0]);c=np.asarray(job['camera']['camera_to_world']);matrix=np.eye(4);matrix[:3,:3]=transform[:3,:3]@c[:3,:3]/s;matrix[:3,3]=transform[:3,:3]@c[:3,3]+transform[:3,3]
        if not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-4):raise ValueError('camera rotation invalid')
        return {'matrix':matrix.tolist(),'fov':job['camera']['camera_angle_x'],'azimuth':fit['parameters']['azimuth'],'qa_iou':b['mask_iou'],'qa_selection':'max_source_mask_iou_between_first_and_recovery','qa_job':b['job'],'qa_job_index':b['job_index']}
    output=[]
    for row in rows:
        obj=row['object_id'];target_view=Path(row['same_index_source_candidate']).stem;anchor=getcam(obj,target_view)
        choices=[f'render_{i:04d}' for i in (0,8,17,25) if f'render_{i:04d}'!=target_view]
        cams={v:getcam(obj,v) for v in choices}
        def score(group):
            az=[anchor['azimuth']]+[cams[v]['azimuth'] for v in group];dist=[angular_distance(az[i],az[j]) for i in range(4) for j in range(i+1,4)]
            return (min(dist),sum(dist),tuple(group))
        chosen=max(itertools.combinations(choices,3),key=score);sources=[cams[v] for v in chosen];target=a.target_root/row['target_image'];imgs=[a.source_png/obj/'source/mv'/f'{v}.png' for v in chosen];same=a.source_png/obj/'source/mv'/f'{target_view}.png'
        if not target.is_file() or not same.is_file() or not all(x.is_file() for x in imgs):raise FileNotFoundError(row['sample_id'])
        cameras=[anchor,*sources];r={'sample_id':row['sample_id'],'source_uid':obj,'source_cluster_id':obj,'split':'test','target_anchor':str(target.resolve()),'target_rgb_input':str(target.resolve()),'target_anchor_kind':'rgb','source_anchor_rgb':str(same.resolve()),'source_images':[str(x.resolve()) for x in imgs],'conditioning_camera_ids':[8,9,10,11],'source_view_names':list(chosen),'original_view_name':target_view,'transform_matrix':[x['matrix'] for x in cameras],'camera_angle_x':[x['fov'] for x in cameras],'source_anchor_transform_matrix':anchor['matrix'],'source_anchor_camera_angle_x':anchor['fov'],'source_camera_qa_iou':[x['qa_iou'] for x in cameras],'camera_provenance':'source_mesh_source_rgb_estimated_not_ground_truth','target_camera_assumption':'same_index_source_camera_unverified','camera_confidence':'LOW' if min(x['qa_iou'] for x in cameras)<.7 else 'provisional','conditioning_input_schema':SCHEMA,'transport_cache':str((a.cache_root/'transport'/f"{row['sample_id']}.npz").resolve()),'transport_cache_pbr':str((a.cache_root/'transport-pbr'/f"{row['sample_id']}.npz").resolve())}
        r['conditioning_input_sha256']=image_fingerprint(r,Path);output.append(r)
    if len(output)!=109 or len({r['sample_id'] for r in output})!=109:raise ValueError('population mismatch')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in output));print(json.dumps({'status':'PREPARED_PROVISIONAL_UNVALIDATED_CAMERAS','samples':len(output),'low_quality_samples':sum(x['camera_confidence']=='LOW' for x in output),'output':str(a.output)}))


if __name__=='__main__':main()
