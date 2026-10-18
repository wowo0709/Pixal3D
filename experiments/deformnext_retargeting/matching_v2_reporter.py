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
        from .summarize_ss_epoch_trends import record_epoch_contrasts
        record_epoch_contrasts(a.root)
        for directory in ['screening','evaluations','calibration-oracle','oracle-verified','retargeting-oracle','projection-audit','role-balance-probe','noise-confirmation','baseline-noise-confirmation','retarget-native-v3/evaluations','retarget-native-v3/noise-confirmation','retarget-native-v3/query-study/evaluations','retarget-native-v3/query-study/noise-confirmation','module-study-native-v4/evaluations','module-study-native-v4/noise-confirmation']:
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
            *(a.root/'retarget-native-v3/epoch-comparisons').glob('*.json'),
            *(a.root/'module-study-native-v4/comparisons').glob('*.json'),
            *(a.root/'module-study-native-v4/epoch-comparisons').glob('*.json'),
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
        report=a.root/'module-study-native-v4/final-primary-report.json'
        report_key=str(report.relative_to(a.root))
        if report.exists() and report.with_suffix('.md').exists() and report_key not in logged:
            value=json.loads(report.read_text())
            artifact=wandb.Artifact('ss-native-module-v4-primary-report',type='evaluation-report')
            artifact.add_file(str(report));artifact.add_file(str(report.with_suffix('.md')))
            run.log_artifact(artifact);run.summary['primary_report_status']=value['status']
            run.summary['primary_models']=len(value['models'])
            logged.add(report_key);receipt.write_text(json.dumps({'logged':sorted(logged),'time':time.time()},indent=2)+'\n')
        table_key=report_key+'#final14-tables-v1'
        if report.exists() and table_key not in logged:
            value=json.loads(report.read_text())
            if value['status']!='primary_matrix_complete':
                raise ValueError('primary matrix is not complete')
            models=wandb.Table(columns=['arm','micro_voxel_iou','voxel_f1','voxel_precision','voxel_recall'],
                data=[[name,*(row[key] for key in ['voxel_iou','voxel_f1','voxel_precision','voxel_recall'])]
                    for name,row in value['models'].items()])
            paired=wandb.Table(columns=['contrast','delta_pp','ci95_lower_pp','ci95_upper_pp','source_clusters','sampling_noises'],
                data=[[name,100*row['gain'],*(100*x for x in row['paired_cluster_bootstrap_ci95']),
                    row['source_clusters'],row['sampling_noise_seeds']] for name,row in value['contrasts'].items()])
            run.log({'primary14/fixed_final_models':models,'primary14/paired_contrasts':paired})
            run.summary['primary14/training_seed_replication']=False
            run.summary['primary14/unseen_test_evaluation']=False
            logged.add(table_key);receipt.write_text(json.dumps({'logged':sorted(logged),'time':time.time()},indent=2)+'\n')
        regional=a.root/'current-geometry-regions-v1/all-six-run-summary-v2.json'
        regional_key=str(regional.relative_to(a.root))
        if regional.exists() and regional.with_suffix('.md').exists() and regional_key not in logged:
            value=json.loads(regional.read_text())
            if value['status']!='PASS_COMPLETE_CURRENT_REGIONAL_PILOT' or value['training_runs']!=6:
                raise ValueError('regional pilot has not passed the complete six-run audit')
            artifact=wandb.Artifact('ss-current-regional-six-seed-runs-v2',type='evaluation-report')
            artifact.add_file(str(regional));artifact.add_file(str(regional.with_suffix('.md')))
            run.log_artifact(artifact)
            table=wandb.Table(columns=['contrast','metric','candidate_minus_baseline','ci95_lower','ci95_upper','source_clusters',
                'seed101','seed202','seed303'],data=[[name,metric,row['mean'],
                    *(row['ci95'] if row['ci95'] is not None else [None,None]),row['source_clusters'],
                    *((row['seed_means'] or {}).get(str(seed)) for seed in [101,202,303])]
                    for name,metrics in value['contrasts'].items() for metric,row in metrics.items()])
            run.log({'current_regional/fixed_final_contrasts':table})
            run.summary['current_regional/status']=value['status']
            run.summary['current_regional/production_promotion']=False
            logged.add(regional_key);receipt.write_text(json.dumps({'logged':sorted(logged),'time':time.time()},indent=2)+'\n')
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
