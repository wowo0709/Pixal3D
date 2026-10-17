"""Complete three-noise coverage for zero-shot and legacy B baselines."""
import argparse
import json
import os
from pathlib import Path
import sys


def jobs(root,work,config_root,initializer):
    result=[]
    for arm in ['zero-edited-rgb','zero-source-only','b1','b2','b3']:
        native=arm.startswith('zero-')
        for noise in [1,2]:
            out=root/f'baseline-noise-confirmation/noise-{noise}/{arm}'
            checkpoint=initializer if native else work/f'evaluation-checkpoints/{arm}/denoiser_step0000480.pt'
            command=[sys.executable,'-u','-m','experiments.deformnext_retargeting.evaluate_native_checkpoint',
                '--stage','ss64','--config-root',str(config_root),'--checkpoint',str(checkpoint),
                '--step','0' if native else '480','--output',str(out),
                '--panel-file',str(work/'panels/panel-128.json'),
                '--seed-map-file',str(work/f'panels/seed-map-extra-noise-{noise}.json'),'--generation-batch','8']
            required=[checkpoint,work/f'panels/seed-map-extra-noise-{noise}.json']
            if native:
                required.append(config_root/'ss64-v1.json')
                command+=['--conditioning-mode',arm.removeprefix('zero-')]
            else:
                conditioner=work/f'evaluation-checkpoints/{arm}/deform_conditioner_step0000480.pt'
                config=config_root/f'ss64-{arm}.json';required += [conditioner,config]
                command+=['--arm-config',str(config),'--conditioner-checkpoint',str(conditioner)]
            result.append({'name':f'evaluate128-baseline-{arm}-noise{noise}','resource':'evaluation','priority':-1,
                'requires':[str(f) for f in required],'outputs':[str(out/'evaluation.json'),str(out/'panel.png')],
                'command':command})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','work-root','config-root','initializer']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();path=a.work_root/'queue-jobs.json';existing=json.loads(path.read_text());new=jobs(a.root,a.work_root,a.config_root,a.initializer)
    if {j['name'] for j in existing}&{j['name'] for j in new}:raise ValueError('baseline repeats already queued')
    if any(not Path(f).is_file() for j in new for f in j['requires']):raise ValueError('baseline checkpoint/config/seed map missing')
    dependencies=[j['name'] for j in existing if j['name'].startswith('compare-native-module-')]
    dependencies += ['compare-p1-vs-c1-three-noise','compare-p2-vs-c1-three-noise','compare-p2-vs-p1-three-noise']
    dependencies += [j['name'] for j in new]
    out=a.root/'module-study-native-v4'
    new.append({'name':'compile-ss-primary-module-report','resource':'cpu','priority':0,'depends_on':dependencies,
        'outputs':[str(out/'final-primary-report.json'),str(out/'final-primary-report.md')],
        'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.summarize_ss_module_study','--root',str(a.root)]})
    tmp=path.with_suffix('.baseline.tmp');tmp.write_text(json.dumps(existing+new,indent=2)+'\n');os.replace(tmp,path)
    print(json.dumps({'new_baseline_evaluations':10,'new_training_runs':0,'final_primary_report_job_added':True}))


if __name__=='__main__':main()
