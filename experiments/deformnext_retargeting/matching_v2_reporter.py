"""Publish common-panel voxel metrics/images and persist campaign completion."""
import argparse
import json
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--gpus',required=True)
    a=p.parse_args()
    import wandb
    from .matching_v2_queue import gpu_ids
    run=wandb.init(project='pixal3d-deformnext-retargeting',entity='youngwoojeon-unist',
        id='ss-matching-v2-evaluations',name='SS matching-v2 common-panel evaluations',resume='allow',
        dir=str(a.work_root),config={'physical_gpu_allowlist':[int(g) for g in gpu_ids(a.gpus)],'data':'DeformNeXt RGB','stage':'SS64','generation_batch':8})
    receipt=a.work_root/'evaluation-logging.json';logged=set(json.loads(receipt.read_text()).get('logged',[])) if receipt.exists() else set()
    while True:
        for directory in ['screening','evaluations','calibration-oracle','oracle-verified','retargeting-oracle','projection-audit','role-balance-probe','noise-confirmation','retarget-native-v3/evaluations','retarget-native-v3/noise-confirmation','retarget-native-v3/query-study/evaluations','retarget-native-v3/query-study/noise-confirmation','module-study-native-v4/evaluations','module-study-native-v4/noise-confirmation']:
            for f in (a.root/directory).glob('**/evaluation.json'):
                key=str(f.relative_to(a.root))
                if key in logged:continue
                image=f.parent/'panel.png'
                if not image.exists():continue
                value=json.loads(f.read_text());metrics=value.get('quick_metrics',{})
                name=str(f.parent.relative_to(a.root)).replace('/','_')
                payload={name+'/'+k:v for k,v in metrics.items()}
                payload[name+'/assets']=len(value.get('sample_ids',value.get('quick_panel_ids',[])))
                payload[name+'/GT_Pred_Diff']=wandb.Image(str(image))
                run.log(payload);logged.add(key)
                receipt.write_text(json.dumps({'logged':sorted(logged),'time':time.time()},indent=2)+'\n')
        comparisons=[a.root/'noise-confirmation/c1-v1-three-noise.json',
            a.root/'role-balance-probe/validation128/paired-v1.json',
            *(a.root/'retarget-native-v3/comparisons').glob('*.json'),
            *(a.root/'module-study-native-v4/comparisons').glob('*.json'),
            *(a.root/'retarget-native-v3/query-study/comparisons').glob('*.json')]
        for f in comparisons:
            key=str(f.relative_to(a.root))
            if key in logged or not f.exists():continue
            record=json.loads(f.read_text());prefix='paired/'+key.removesuffix('.json').replace('/','_')
            payload={prefix+'/'+k:v for k,v in record.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
            interval=record['paired_cluster_bootstrap_ci95']
            payload.update({prefix+'/ci95_lower':interval[0],prefix+'/ci95_upper':interval[1]})
            run.log(payload);logged.add(key)
            receipt.write_text(json.dumps({'logged':sorted(logged),'time':time.time()},indent=2)+'\n')
        state_path=a.work_root/'queue-state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        run.log({'campaign/running':sum(r['status']=='running' for r in state.values()),
                 'campaign/completed':sum(r['status']=='completed' for r in state.values()),
                 'campaign/held':sum(r['status']=='HELD' for r in state.values()),
                 'campaign/failed':sum(r['status']=='FAILED' for r in state.values())})
        if state and all(r['status'] in {'completed','HELD','FAILED'} for r in state.values()):
            result={'status':'finished_with_holds' if any(r['status']=='HELD' for r in state.values()) else 'finished',
                    'failed':[k for k,r in state.items() if r['status']=='FAILED'],
                    'held':{k:r.get('reason') for k,r in state.items() if r['status']=='HELD'},
                    'completed':[k for k,r in state.items() if r['status']=='completed'],'time':time.time()}
            if result['failed']:result['status']='requires_failure_review'
            (a.root/'campaign-result.json').write_text(json.dumps(result,indent=2)+'\n')
            run.summary['campaign_status']=result['status'];run.finish();break
        time.sleep(30)


if __name__=='__main__':main()
