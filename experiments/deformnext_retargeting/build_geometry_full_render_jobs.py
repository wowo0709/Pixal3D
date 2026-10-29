"""Build camera8/9/10 jobs for six common-501 geometry ablations."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np

def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--template-plan-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--chunk-samples',type=int,default=8)
    p.add_argument('--source-m0-fits',type=Path);a=p.parse_args()
    if a.output_root.exists():raise FileExistsError(a.output_root)
    campaign=json.loads(a.campaign.read_text());arms=campaign['arms']
    entries={(r['arm'],r['sample_id']):r for r in campaign['plans']};ids=[r['sample_id'] for r in campaign['plans'] if r['arm']==arms[0]]
    templates={}
    for path in a.template_plan_root.glob('chunk*/jobs.jsonl'):
        for job in rows(path):
            if job.get('method')=='mv_pixal3d_zero_shot_source_m0' and job.get('camera_id') in (8,9,10):templates[job['sample_id'],job['camera_id']]=job
    if len(arms)!=6 or len(ids)!=501 or len(templates)!=1503:raise ValueError('six-arm common501 templates required')
    a.output_root.mkdir(parents=True);total=0
    for start in range(0,len(ids),a.chunk_samples):
        group=ids[start:start+a.chunk_samples];chunk=a.output_root/f'chunk{start//a.chunk_samples:03d}';stage=chunk/'stage';stage.mkdir(parents=True)
        jobs=[];inputs=[]
        for sid in group:
            for arm in arms:
                root=Path(json.loads(Path(entries[arm,sid]['top1_pbr_plan']).read_text())['output']);glb=root/'benchmark.glb';report=glb.with_suffix('.json')
                if not glb.is_file() or not report.is_file():raise FileNotFoundError(glb)
                relative=Path('inputs')/arm/sid/'benchmark.glb';target=stage/relative;target.parent.mkdir(parents=True);os.link(glb,target);inputs.append(str(relative))
                transform=json.loads(report.read_text())['to_world']
                if a.source_m0_fits:
                    fit=json.loads((a.source_m0_fits/arm/(sid+'.json')).read_text())
                    if fit.get('status')!='PASS_REGENERATED_SOURCE_M0_SIM3' or fit.get('sample_id')!=sid or fit.get('arm')!=arm:
                        raise ValueError('invalid source-M0 fit')
                    transform=(np.asarray(fit['correction_sim3']) @ np.asarray(transform)).tolist()
                for camera in (8,9,10):
                    job=dict(templates[sid,camera],method='geometry_'+arm.lower(),mesh=str(relative),metric_mesh=str(relative),
                        to_world=transform,trusted_material_dump=None,output_rgba=f'renders/{arm}/{sid}/view{camera:02d}.png')
                    jobs.append(job)
        (chunk/'jobs.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in jobs));(chunk/'input-files.txt').write_text(''.join(x+'\n' for x in sorted(set(inputs))))
        (chunk/'plan.json').write_text(json.dumps({'status':'PREPARED_GEOMETRY_FULL_RENDER_CHUNK','sample_ids':group,'jobs':len(jobs)},indent=2)+'\n');total+=len(jobs)
    result={'status':'PASS_GEOMETRY_FULL_RENDER_PLAN','samples':501,'arms':arms,'jobs':total,'chunks':(len(ids)+a.chunk_samples-1)//a.chunk_samples,'cameras':[8,9,10]}
    (a.output_root/'campaign.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
