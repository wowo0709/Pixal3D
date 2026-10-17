"""Append fixed-noise parent evaluation and the matched physical-query study."""
import argparse
import json
import os
from pathlib import Path
import sys

from .matching_v2_queue import gpu_ids


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--control-gpus',required=True);p.add_argument('--query-gpus',required=True)
    a=p.parse_args();control=gpu_ids(a.control_gpus);query=gpu_ids(a.query_gpus)
    if len(control)!=2 or len(query)!=2 or set(control)&set(query):raise ValueError('two disjoint assigned GPU pairs required')
    root=a.root;work=a.work_root;stage=root/'retarget-native-v3';study=stage/'query-study'
    jobs=json.loads((work/'queue-jobs.json').read_text());original={j['name'] for j in jobs};added=[]
    py=sys.executable
    def add(job):
        if job['name'] in original or job['name'] in {j['name'] for j in added}:raise ValueError('follow-up job already exists')
        added.append(job)
    def evaluation(name,config,run,out,noise,dependency):
        seed=work/'panels'/('seed-map.json' if noise==0 else f'seed-map-extra-noise-{noise}.json')
        if not seed.is_file():raise FileNotFoundError(seed)
        add({'name':name,'resource':'evaluation','priority':1,'depends_on':[dependency],
            'requires':[str(run/'completed.json')],'outputs':[str(out/'evaluation.json'),str(out/'panel.png')],
            'command':[py,'-u','-m','experiments.deformnext_retargeting.evaluate_native_checkpoint',
                '--stage','ss64','--config-root',str(config.parent),'--arm-config',str(config),
                '--checkpoint',str(run/'ckpts/denoiser_step0000480.pt'),
                '--conditioner-checkpoint',str(run/'ckpts/deform_conditioner_step0000480.pt'),
                '--step','480','--output',str(out),'--panel-file',str(work/'panels/panel-128.json'),
                '--seed-map-file',str(seed),'--generation-batch','8']})
    def compare(name,baseline,candidate,out,dependencies):
        path=work/'comparison-inputs'/(name+'.json');path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps([{'noise':str(n),'baseline':str(baseline[n]),'candidate':str(candidate[n])} for n in range(3)],indent=2)+'\n')
        add({'name':name,'resource':'cpu','priority':0,'depends_on':dependencies,
            'requires':[str(f) for f in [*baseline,*candidate]],'outputs':[str(out)],
            'command':[py,'-u','-m','experiments.deformnext_retargeting.compare_ss_noise_panels',
                '--input-list',str(path),'--output',str(out)]})
    def baseline_files(arm):
        return [root/('evaluations/v1/native/evaluation.json' if arm=='v1' else 'evaluations/c1/normal/evaluation.json'),
            *[root/f'noise-confirmation/noise-{n}/{arm}/evaluation.json' for n in [1,2]]]
    paths={};deps={}
    for arm in ['p1-native-grid','p2-matcher-grid']:
        paths[arm]=[stage/f'evaluations/{arm}/evaluation.json'];deps[arm]=['evaluate128-'+arm]
        for n in [1,2]:
            name=f'evaluate128-{arm}-noise{n}';out=stage/f'noise-confirmation/noise-{n}/{arm}'
            evaluation(name,stage/f'configs/{arm}.json',stage/'runs_ss64'/arm,out,n,'train-ss-'+arm)
            paths[arm].append(out/'evaluation.json');deps[arm].append(name)
    selections=[]
    for key,left,right in [('p1-vs-c1','c1','p1-native-grid'),('p2-vs-c1','c1','p2-matcher-grid'),('p2-vs-p1','p1-native-grid','p2-matcher-grid')]:
        name='compare-'+key+'-three-noise';selections.append(name)
        compare(name,baseline_files(left) if left=='c1' else paths[left],paths[right],
            stage/'comparisons'/(key+'-three-noise.json'),deps[right]+([] if left=='c1' else deps[left]))
    prepare='prepare-matched-query-study'
    add({'name':prepare,'resource':'cpu','priority':0,'depends_on':selections,
        'outputs':[str(study/'decision.json'),str(study/'configs/c1-control.json'),str(study/'configs/c2-query.json')],
        'command':[py,'-u','-m','experiments.deformnext_retargeting.prepare_query_study','--root',str(root)]})
    for arm,gpus in [('c1-control',control),('c2-query',query)]:
        train='train-query-study-'+arm;run=study/'runs_ss64'/arm;config=study/'configs'/(arm+'.json')
        add({'name':train,'resource':'exclusive','gpus':gpus,'priority':0,'depends_on':[prepare],
            'requires':[str(study/'decision.json'),str(study/'context-loader-preflight.json')],
            'outputs':[str(run/'completed.json')],
            'command':[py,'-u','-m','experiments.deformnext_retargeting.matching_v2_ss',
                '--mode','train','--root',str(study),'--config',str(config)]})
        paths[arm]=[];deps[arm]=[]
        for n in [0,1,2]:
            name=f'evaluate128-query-{arm}-noise{n}'
            out=study/f'evaluations/{arm}' if n==0 else study/f'noise-confirmation/noise-{n}/{arm}'
            evaluation(name,config,run,out,n,train);paths[arm].append(out/'evaluation.json');deps[arm].append(name)
    compare('compare-query-effect-three-noise',paths['c1-control'],paths['c2-query'],
        study/'comparisons/c2-vs-matched-c1.json',deps['c1-control']+deps['c2-query'])
    for arm in ['c1-control','c2-query']:
        for base in ['v1','c1']:
            compare(f'compare-query-{arm}-vs-{base}-three-noise',baseline_files(base),paths[arm],
                study/'comparisons'/(arm+'-vs-original-'+base+'.json'),deps[arm])
    tmp=work/'queue-jobs.followup.tmp';tmp.write_text(json.dumps(jobs+added,indent=2)+'\n');os.replace(tmp,work/'queue-jobs.json')
    print(json.dumps({'appended_jobs':len(added),'control_gpus':control,'query_gpus':query,
        'C1_and_C2_start_only_after_final128_three_noise_parent_selection':True}),flush=True)


if __name__=='__main__':main()
