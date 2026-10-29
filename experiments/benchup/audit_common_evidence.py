"""Verify per-image provenance for all currently shared BenchUp cases."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('runtime','corrected-root','benchmark-root','manifest','aggregate','output'):p.add_argument('--'+x,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.benchmark_root))
    from deformnext_benchmark.io import canonical_hash
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    population={x['sample_id']:x for x in rows}
    if len(rows)!=109 or len(population)!=109:raise ValueError('unique frozen population required')
    aggregate_bytes=a.aggregate.read_bytes();aggregate=json.loads(aggregate_bytes);common=aggregate['common_sample_ids']
    if aggregate['population_manifest_sha256']!=sha(a.manifest) or len(common)!=aggregate['common_count'] or not set(common)<=population.keys():raise ValueError('aggregate manifest mismatch')
    methods={'V1':'v1','Selected':'selected','MV-zero':'mv-zero','Official-RGB1':'official-rgb1','TRELLIS2':'trellis2','ReconViaGen':'recon','ShapeUP':'shapeup','SpaceControl':'spacecontrol','MVAdapter-Recon':'mvadapter-recon','CorrAdapter-Recon':'corradapter-recon'}
    fixed={'TRELLIS2','ReconViaGen','SpaceControl','MVAdapter-Recon','CorrAdapter-Recon'}
    protocol=json.loads((a.benchmark_root/'protocol.json').read_text());protocol_hash=canonical_hash(protocol)
    checked=[];errors=[]
    for sid in common:
        row=population[sid];refs=[Path(row['target_anchor']),Path(row['source_images'][0]),Path(row['source_images'][1])];refhash=[sha(x) for x in refs]
        for name,method in methods.items():
            try:
                root=a.corrected_root if name in fixed else a.runtime
                chunk=root/(method+'-render-stream-v1')/'cases'/sid
                learned=root/(method+'-learned-metrics-v1')/sid
                metrics=json.loads((learned/'learned.json').read_text());basic=json.loads((chunk/'diagnostic-scores.json').read_text())
                if metrics['sample_id']!=sid or basic['sample_id']!=sid or metrics['status']!='PROVISIONAL_INPUT_IMAGE_METRICS' or basic['status']!='IMAGE_DIAGNOSTIC_PROVISIONAL_NOT_FINAL_BENCHMARK':raise ValueError('metric identity/status')
                if sha(learned/'inception.npz')!=metrics['inception_sha256']:raise ValueError('Inception feature hash')
                jobs=[json.loads(x) for x in (chunk/'jobs.jsonl').read_text().splitlines() if x];byout={j['output_rgba']:j for j in jobs}
                if len(jobs)!=6 or len(byout)!=6:raise ValueError('six distinct render jobs required')
                meshhash=sha(chunk/'stage/prediction.glb')
                candidates=[]
                for marker in root.glob(f'{method}-postprocess-shard-*/cases/{sid}/archive.json'):
                    archive=json.loads(marker.read_text())
                    if archive.get('status')=='PASS_FILE4_SHA256' and archive.get('sha256')==meshhash:
                        bake=json.loads((marker.parent/'benchmark.json').read_text())
                        reg=json.loads((marker.parent/'source-registration.json').read_text())
                        if bake['glb_sha256']==meshhash and bake['prediction_sha256']==reg['prediction_sha256'] and not reg['target_GT_used']:
                            candidates.append(reg)
                if not candidates:raise ValueError('no matching archived bake/source-only registration')
                raw_matrix=np.asarray(candidates[0]['to_world'],dtype=np.float64)
                aligned_matrix=np.asarray(candidates[0]['correction_sim3'],dtype=np.float64)@raw_matrix
                if name in {'TRELLIS2','ReconViaGen','MVAdapter-Recon','CorrAdapter-Recon'} and candidates[0]['native_frame']!='trellis_minus_y_front_plus_z_up':raise ValueError('uncorrected native front-axis')
                for mode in ('raw','source_m0'):
                    entries=metrics['metrics'][mode]
                    if len(entries)!=3:raise ValueError('three metric views required')
                    for i,entry in enumerate(entries):
                        png=chunk/'stage'/mode/f'view{i}.png';receipt=json.loads(png.with_suffix('.receipt.json').read_text());job=byout[f'{mode}/view{i}.png'];digest=sha(png)
                        if job['sample_id']!=sid or job['method']!=method+'_'+mode or job['protocol_sha256']!=protocol_hash:raise ValueError('job identity/protocol')
                        expected_camera={'camera_to_world':row['transform_matrix'][i],'camera_angle_x':row['camera_angle_x'][i],'convention':'Blender_OpenGL_camera_to_world'}
                        if canonical_hash(job['camera'])!=canonical_hash(expected_camera) or not np.allclose(job['to_world'],raw_matrix if mode=='raw' else aligned_matrix,atol=1e-10):raise ValueError('job camera or alignment does not match manifest/registration')
                        if receipt['output_sha256']!=digest or entry['render_sha256']!=digest or entry['reference_sha256']!=refhash[i]:raise ValueError('render/reference image hash')
                        if receipt['input_sha256']!=meshhash or receipt['metric_mesh_sha256']!=meshhash or receipt['job_sha256']!=canonical_hash(job) or receipt['camera_sha256']!=canonical_hash(job['camera']) or receipt['to_world_sha256']!=canonical_hash(job['to_world']) or receipt['protocol_sha256']!=protocol_hash:raise ValueError('mesh/job/camera receipt')
                        if receipt['blender_version']!=protocol['render']['blender_version']:raise ValueError('Blender version')
                checked.append({'method':name,'sample_id':sid,'mesh_sha256':meshhash})
            except Exception as e:errors.append({'method':name,'sample_id':sid,'error':f'{type(e).__name__}: {e}'})
    result={'status':'PASS_CURRENT_COMMON_EVIDENCE' if not errors else 'FAIL_CURRENT_COMMON_EVIDENCE','common_count':len(common),'checked_method_cases':len(checked),'expected_method_cases':len(common)*len(methods),'aggregate_sha256':hashlib.sha256(aggregate_bytes).hexdigest(),'manifest_sha256':sha(a.manifest),'scope':'hash and declared job provenance only; not camera ground-truth validation or a target-geometry metric','errors':errors,'checked':checked}
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(a.output)
    print(json.dumps({k:v for k,v in result.items() if k!='checked'}),flush=True)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
