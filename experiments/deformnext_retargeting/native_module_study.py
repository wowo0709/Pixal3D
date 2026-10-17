"""Restore all five SS module contrasts on the corrected native projection."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

from .matching_v2_queue import gpu_ids


ARMS=[('b2-top1','b2',False),('b2-top1-r','b2',True),('b3-attention-r','b3',True),
      ('c1-retarget-r','c1',True),('c2-query-r','c2',True)]


def configs(base,study,head):
    if base['experiment']['head_sha256']!=head['head_sha256']:
        raise ValueError('module ladder must use the frozen shared matcher')
    result={}
    for index,(name,arm,reject) in enumerate(ARMS):
        cfg=copy.deepcopy(base);run=study/'runs_ss64'/name
        cfg.update(default_output_dir=str(run),output_dir=str(run),load_dir=str(run),ckpt='latest',
            num_gpus=2,master_port=str(29951+index),wandb_id='ss-native-module-v4-'+name,
            wandb_name='SS native module '+name)
        for args in [cfg['dataset']['args'],cfg['validation']['dataset_args']]:
            args.update(transport_arm=arm);args.pop('deformation_root',None)
        module={'implementation':'matching-v2','arm':arm,'value_dim':1024,'query_dim':1024,
            'hidden_dim':256,'g_max':.25,'reject':reject,
            'reject_threshold':head['reliability']['threshold'] if reject else 0.,
            'projection_mode':'native-bilinear','initialization_rng':'r9-compatible'}
        if arm=='c2':module['deformation_dim']=2048
        cfg['trainer']['args'].update(deform_conditioner=module,batch_size_per_gpu=48,batch_split=1,
            max_steps=480,i_save=96,max_checkpoints=5)
        cfg['experiment'].update(arm=name,conditioning_schema='matching-v2-native-module-v4',
            release_status='pending-native-proof-and-memory-probe',recipe_path=str(study/'configs'/(name+'.json')),
            global_batch=96,world_size=2,microbatch_per_gpu=48,epochs=5,validate_at_start=True,
            auto_continue_after_pilot=False,initialization='original mixed385 base+LoRA; independent optimizer',
            source_comparison='shared H3 native-projection Top1 -> rejection -> attention -> adapter -> physical query',
            raw_transport_gate='historical nearest gate remains HOLD; native ladder is an explicit diagnostic study',
            inference_gt_inputs=False)
        result[name]=cfg
    return result


def build_jobs(root,work,study,first,second):
    py=sys.executable;jobs=[];paths={};evaluations={};inputs={}
    original_c1=[root/'evaluations/c1/normal/evaluation.json',
        *[root/f'noise-confirmation/noise-{n}/c1/evaluation.json' for n in [1,2]]]
    v1=[root/'evaluations/v1/native/evaluation.json',
        *[root/f'noise-confirmation/noise-{n}/v1/evaluation.json' for n in [1,2]]]
    previous={tuple(first):'p2-matcher-grid',tuple(second):'p1-native-grid'}
    for i,(name,arm,reject) in enumerate(ARMS):
        assigned=first if i%2==0 else second
        before=previous[tuple(assigned)]
        dependencies=([f'evaluate128-{before}',*[f'evaluate128-{before}-noise{n}' for n in [1,2]]]
            if before.startswith('p') else evaluations[before])
        config=study/'configs'/(name+'.json');run=study/'runs_ss64'/name
        train='train-native-module-'+name
        jobs.append({'name':train,'resource':'exclusive','gpus':assigned,'priority':0,
            'depends_on':dependencies,'requires':[str(study/'configs-ready.json'),str(study/'context-loader-preflight.json'),str(root/'retarget-native-v3/learned-evidence/cache-audit.json')],
            'outputs':[str(run/'completed.json')],
            'command':[py,'-u','-m','experiments.deformnext_retargeting.matching_v2_ss',
                '--mode','train','--root',str(study),'--config',str(config)]})
        paths[name]=[];evaluations[name]=[]
        for noise in [0,1,2]:
            out=study/f'evaluations/{name}' if noise==0 else study/f'noise-confirmation/noise-{noise}/{name}'
            seed=work/'panels'/('seed-map.json' if noise==0 else f'seed-map-extra-noise-{noise}.json')
            ev=f'evaluate128-native-module-{name}-noise{noise}';evaluations[name].append(ev);paths[name].append(out/'evaluation.json')
            jobs.append({'name':ev,'resource':'evaluation','priority':1,'depends_on':[train],
                'requires':[str(run/'completed.json')],'outputs':[str(out/'evaluation.json'),str(out/'panel.png')],
                'command':[py,'-u','-m','experiments.deformnext_retargeting.evaluate_native_checkpoint',
                    '--stage','ss64','--config-root',str(config.parent),'--arm-config',str(config),
                    '--checkpoint',str(run/'ckpts/denoiser_step0000480.pt'),
                    '--conditioner-checkpoint',str(run/'ckpts/deform_conditioner_step0000480.pt'),
                    '--step','480','--output',str(out),'--panel-file',str(work/'panels/panel-128.json'),
                    '--seed-map-file',str(seed),'--generation-batch','8']})
        previous[tuple(assigned)]=name
    contrasts=[('b2-top1','v1'),('b2-top1-r','b2-top1'),('b3-attention-r','b2-top1-r'),
        ('c1-retarget-r','b3-attention-r'),('c2-query-r','c1-retarget-r')]
    contrasts += [(name,baseline) for name,_,_ in ARMS for baseline in ['v1','original-c1']
        if (name,baseline) not in contrasts]
    for candidate,baseline in contrasts:
        left=v1 if baseline=='v1' else original_c1 if baseline=='original-c1' else paths[baseline]
        name=f'compare-native-module-{candidate}-vs-{baseline}'
        spec=work/'comparison-inputs'/(name+'.json')
        inputs[spec]=[{'noise':str(n),'baseline':str(left[n]),'candidate':str(paths[candidate][n])} for n in range(3)]
        jobs.append({'name':name,'resource':'cpu','priority':0,
            'depends_on':evaluations[candidate]+evaluations.get(baseline,[]),
            'requires':[str(f) for f in [*left,*paths[candidate]]],
            'outputs':[str(study/'comparisons'/(candidate+'-vs-'+baseline+'.json'))],
            'command':[py,'-u','-m','experiments.deformnext_retargeting.compare_ss_noise_panels',
                '--input-list',str(spec),'--output',str(study/'comparisons'/(candidate+'-vs-'+baseline+'.json'))]})
    return jobs,inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--first-gpus',required=True);p.add_argument('--second-gpus',required=True)
    a=p.parse_args();first=gpu_ids(a.first_gpus);second=gpu_ids(a.second_gpus)
    if len(first)!=2 or len(second)!=2 or set(first)&set(second):raise ValueError('two disjoint assigned GPU pairs required')
    root=a.root;work=a.work_root;study=root/'module-study-native-v4'
    head=json.loads((root/'selected-head.json').read_text())
    audit=json.loads((root/'retarget-native-v3/learned-evidence/cache-audit.json').read_text())
    if audit.get('status')!='PASS':raise ValueError('shared matcher cache audit is incomplete')
    base=json.loads((root/'retarget-native-v3/configs/p2-matcher-grid.json').read_text())
    recipes=configs(base,study,head)
    for name,cfg in recipes.items():
        path=study/'configs'/(name+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise ValueError('module recipe already exists: '+name)
        path.write_text(json.dumps(cfg,indent=2)+'\n')
    jobs,inputs=build_jobs(root,work,study,first,second)
    existing=json.loads((work/'queue-jobs.json').read_text());names={j['name'] for j in existing}
    if names&{j['name'] for j in jobs}:raise ValueError('module jobs already queued')
    for path,value in inputs.items():
        path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
    # Pending retarget-only parent selection would duplicate the new C1/C2
    # pair with a different rejection contract. Preserve its specification,
    # but supersede those jobs rather than running an incomplete replacement.
    state=json.loads((work/'queue-state.json').read_text());superseded=[]
    for job in existing:
        name=job['name']
        if name=='prepare-matched-query-study' or name.startswith(('train-query-study-','evaluate128-query-','compare-query-')):
            if state.get(name,{}).get('status') not in {'pending','HELD'}:raise ValueError('cannot supersede an active or completed query job')
            job['gate_file']=str(study/'superseded-query-gate.json');superseded.append(name)
            state[name]={'status':'HELD','resource':job['resource'],'reason':{'superseded_by':'complete native module ladder','old_definition_preserved':True}}
    (study/'superseded-query-gate.json').write_text(json.dumps({'status':'SUPERSEDED','reason':'C1/C2 are included with the shared rejection contract in module-study-native-v4'})+'\n')
    ready={'status':'PASS','meaning':'configuration ready, not a performance promotion','arms':list(recipes),
        'head_sha256':head['head_sha256'],'projection':'native-bilinear','head_temperature':head['temperature'],
        'reliability':head['reliability'],'epochs':5,'global_batch':96,'world_size':2,'microbatch_per_rank':48,
        'historical_nearest_transport_gate':'HOLD unchanged','superseded_query_jobs':superseded,
        'reason':'restore original mandatory Top1/rejection/attention comparisons and adapter/query additions; negative frozen GT probe is retained as evidence, not substituted for these trainable comparisons'}
    (study/'configs-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
    for path,value in [(work/'queue-jobs.json',existing+jobs),(work/'queue-state.json',state)]:
        tmp=path.with_suffix('.module.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
    print(json.dumps({'queued_arms':list(recipes),'new_jobs':len(jobs),'superseded_pending_jobs':len(superseded)}))


if __name__=='__main__':main()
