"""Same-epoch ray module contrasts from actual closed common32 snapshots."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path

from .audit_ray_group_startup import verify_initial
from .compare_frozen_utility_epochs import closed_snapshot, record_at
from .compare_orthogonal_utility_epochs import paired_records
from .ray_group_study import ARMS, STATIC, validate_profile
from .semantic_topk_search import write_new
from .summarize_frozen_utility_followup import zero_records


def read(path):
    return json.loads(path.read_text())


def verify_epoch_audit(audit, arm, epoch):
    expected = dict(status='PASS_ACTUAL_EPOCH_OPTIMIZER_LOGGING_AND_BATCH', epoch=epoch,
        step=96*epoch, training_seed=101, global_batch=96, world_size=2,
        microbatch_per_rank=48, accumulation=1, actual_optimizer_states=366 if arm in STATIC else 371,
        all_optimizer_steps=96*epoch, actual_remote_four_numeric_fields_verified=True,
        actual_remote_PNG_byte_exact=True)
    if any(audit.get(key) != value for key,value in expected.items()):
        raise ValueError('actual closed ray epoch/optimizer/logging contract differs: ' + arm)


def verify_matched_recipes(candidate, baseline):
    for key in ['models','dataset','validation','data_dir','training_seed','num_gpus']:
        if candidate[key] != baseline[key]:
            raise ValueError('ray comparison has unmatched nuisance: ' + key)
    left,right=copy.deepcopy(candidate['trainer']),copy.deepcopy(baseline['trainer'])
    left['args'].pop('deform_conditioner')
    right['args'].pop('deform_conditioner')
    if left != right:
        raise ValueError('ray comparison has unmatched trainer/continuation/learning rate')


def summarize(root, epoch, candidate, baseline):
    if epoch not in range(1,6) or candidate == baseline:
        raise ValueError('distinct declared ray arms and epoch1..5 required')
    configs={name:read(root/'configs'/(name+'.json')) for name in [candidate,baseline]}
    for cfg in configs.values():
        validate_profile(cfg,root)
    verify_matched_recipes(configs[candidate],configs[baseline])
    campaign=root.parent
    values,initial,inputs,folders={},{},[],{}
    for name in [candidate,baseline]:
        folder=root/'epoch-audits-v1'/('train-'+name)/f'epoch{epoch:02d}'
        folders[name]=folder
        receipt=closed_snapshot(folder,campaign)
        verify_epoch_audit(read(folder/'audit.json'),name,epoch)
        values[name]=read(folder/'validation-snapshot.json')
        initial[name]=read(folder/'initial-validation-snapshot.json')
        if values[name]['step']!=96*epoch:
            raise ValueError('actual snapshot step differs')
        inputs += [receipt,folder/'audit.json',folder/'validation-snapshot.json',
                   folder/'initial-validation-snapshot.json',root/'configs'/(name+'.json')]
    verify_initial(initial[candidate],initial[baseline])
    ids=values[candidate]['quick_panel_ids']
    if ids!=values[baseline]['quick_panel_ids'] or ids!=initial[candidate]['quick_panel_ids']:
        raise ValueError('actual same-epoch panel/order differs')
    pair=paired_records(values[baseline]['quick_per_asset'],values[candidate]['quick_per_asset'],ids)
    for name,field in [(candidate,'candidate_iou'),(baseline,'baseline_iou')]:
        if abs(pair[field]-values[name]['quick_metrics']['voxel_iou'])>1e-12:
            raise ValueError('actual voxel aggregation differs')
    context,temporal={},{}
    for name in [candidate,baseline]:
        context[name]={}
        for label,base,arm in [('V1',campaign.parent/'ss-seed-confirmation-v1','v1'),
                               ('old-Top1-R',campaign,'static-top1-c1')]:
            path=base/'runs_ss64'/arm/'validation.jsonl'
            other,start=record_at(path,96*epoch),record_at(path,0)
            verify_initial(initial[name],start)
            context[name][label]=paired_records(other['quick_per_asset'],values[name]['quick_per_asset'],ids)
            inputs.append(path)
        if epoch==1:
            previous=initial[name]
        else:
            folder=folders[name].parent/f'epoch{epoch-1:02d}'
            inputs += [closed_snapshot(folder,campaign),folder/'validation-snapshot.json']
            previous=read(folder/'validation-snapshot.json')
        if previous['step']!=96*(epoch-1) or previous['quick_panel_ids']!=ids:
            raise ValueError('actual previous epoch identity differs')
        temporal[name]=paired_records(previous['quick_per_asset'],values[name]['quick_per_asset'],ids)
    zeros,paths=zero_records(campaign)
    inputs+=paths
    for name in [candidate,baseline]:
        for label,repeats in zeros.items():
            lookup={row['sample_id']:row for row in repeats[0]['per_asset']}
            context[name][label]=paired_records([lookup[sid] for sid in ids],values[name]['quick_per_asset'],ids)
    inputs.append(Path(__file__))
    return dict(status='PASS_CLOSED_MATCHED_RAY_COMMON32_SAME_EPOCH_CONTRAST',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),epoch=epoch,step=96*epoch,
        candidate=candidate,baseline=baseline,
        metrics={name:values[name]['quick_metrics'] for name in [candidate,baseline]},
        matched_ray_contrast=pair,context_contrasts=context,temporal_contrasts=temporal,
        inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dict.fromkeys(inputs)],
        scope='Common32/noise0/trainingseed101;source-cluster95% bootstrap,exploratory intermediate trend. Fixedraw480/final128x3 and training-seed confirmation remain required.',
        historical_scope='OldTop1R uses a different query unit;context only. V1 and both zeros share inputs/desiredM1GT. Zero-source uses4M0RGB,zero-edited uses1M1+3M0RGB.',
        model_adopted=False,full_goal_complete=False)


def render(result):
    rows=['# Same-epoch ray module comparison','',result['scope'],'',
        '| Arm | IoU | F1 |','| --- | ---: | ---: |']
    rows += [f"| {name} | {m['voxel_iou']:.6f} | {m['voxel_f1']:.6f} |" for name,m in result['metrics'].items()]
    pair=result['matched_ray_contrast'];lo,hi=pair['paired_cluster_bootstrap_ci95']
    rows += ['',f"{result['candidate']} minus {result['baseline']}: {100*pair['gain']:+.3f} pp, 95% CI [{100*lo:+.3f},{100*hi:+.3f}] pp.",'',result['historical_scope']]
    return '\n'.join(rows)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--candidate',choices=list(ARMS),required=True)
    p.add_argument('--baseline',choices=list(ARMS),required=True)
    p.add_argument('--expected-sha',required=True)
    args=p.parse_args()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=args.expected_sha:
        raise ValueError('declared comparison source changed')
    result=summarize(args.root,args.epoch,args.candidate,args.baseline)
    folder=args.root/'epoch-pair-comparisons-v1'/(args.candidate+'-vs-'+args.baseline)
    output=folder/f'epoch{args.epoch:02d}.json'
    write_new(output,result)
    with output.with_suffix('.md').open('x') as stream:
        stream.write(render(result))
    print(json.dumps(dict(status=result['status'],epoch=args.epoch,contrast=result['matched_ray_contrast'])),flush=True)


if __name__=='__main__':main()
