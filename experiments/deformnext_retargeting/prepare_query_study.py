"""Fix a final-evaluation parent, then prepare matched C1/C2 fresh SS runs."""
import argparse
import copy
import json
from pathlib import Path


def choose_parent(comparisons):
    positive=lambda key:comparisons[key]['paired_cluster_bootstrap_ci95'][0]>0
    if positive('p2-vs-c1') and positive('p2-vs-p1'):return 'p2-matcher-grid'
    if positive('p1-vs-c1'):return 'p1-native-grid'
    return 'original-c1'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();stage=a.root/'retarget-native-v3';out=stage/'query-study'
    context=json.loads((out/'context-population-audit.json').read_text())
    if context.get('status')!='PASS':raise ValueError('query context population has not passed')
    for name in ['p1-native-grid','p2-matcher-grid']:
        done=json.loads((stage/'runs_ss64'/name/'completed.json').read_text())
        if done.get('status')!='PASS' or done.get('updates')!=480 or done.get('epochs')!=5:
            raise ValueError('query selection cannot use incomplete parent training')
    comparisons={key:json.loads((stage/'comparisons'/(key+'-three-noise.json')).read_text())
        for key in ['p1-vs-c1','p2-vs-c1','p2-vs-p1']}
    if any(x['assets']!=128 or x['sampling_noise_seeds']!=3 for x in comparisons.values()):
        raise ValueError('query parent requires final128 and all3 fixed sampling repeats')
    parent=choose_parent(comparisons);source='p1-native-grid' if parent=='original-c1' else parent
    base=json.loads((stage/'configs'/(source+'.json')).read_text())
    if base['trainer']['args']['max_steps']!=480 or base['experiment']['global_batch']!=96:
        raise ValueError('query study requires the fixed five-epoch/global96 parent recipe')
    if parent=='original-c1':base['trainer']['args']['deform_conditioner']['projection_mode']='nearest'
    for name,arm in [('c1-control','c1'),('c2-query','c2')]:
        cfg=copy.deepcopy(base);run=out/'runs_ss64'/name;config=out/'configs'/(name+'.json')
        cfg.update(default_output_dir=str(run),output_dir=str(run),load_dir=str(run),num_gpus=2,
            master_port='29942' if arm=='c2' else '29941',wandb_id='ss-native-v3-'+name,
            wandb_name='SS matched '+name+' parent '+parent)
        cfg['trainer']['args'].update(batch_size_per_gpu=48,batch_split=1,max_steps=480)
        module=cfg['trainer']['args']['deform_conditioner'];module.update(arm=arm,reject=False,reject_threshold=0.)
        if arm=='c2':module['deformation_dim']=2048
        else:module.pop('deformation_dim',None)
        for args in [cfg['dataset']['args'],cfg['validation']['dataset_args']]:
            args['transport_arm']=arm;args.pop('deformation_root',None)
            if arm=='c2' and parent!='p2-matcher-grid':
                args['deformation_root']=str(a.root.parent/'rgb_v1/matches_r9_v5')
        cfg['experiment'].update(arm=name,world_size=2,microbatch_per_gpu=48,epochs=5,
            validate_at_start=True,auto_continue_after_pilot=False,release_status='pending-native-query-proof-memory-probe',
            source_parent=parent,source_comparison='C2 versus fresh matched C1; query channels physically absent in C1',
            initialization='original mixed385 complete base+LoRA; no trained parent weights reused',recipe_path=str(config))
        config.parent.mkdir(parents=True,exist_ok=True);config.write_text(json.dumps(cfg,indent=2)+'\n')
    decision={'status':'PASS','selected_parent':parent,'comparisons':comparisons,
        'selection':'P2 only if both gains over original C1 and P1 have positive source-cluster CI; else passing P1; otherwise original C1',
        'evaluation_scope':'development128, three sampling repeats; not unseen test or training-seed replication',
        'arms':['c1-control','c2-query'],'fresh_parent_replay':True,'physical_deformation_channels':{'c1':0,'c2':2048},
        'source_of_deformation':'Omega epoch7 RAW FRAME+GLOBAL context delta; no GT deformation input',
        'global_batch':96,'world_size':2,'microbatch_per_rank':48,'accumulation':1,'epochs':5}
    (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps({'status':'PASS','parent':parent,'query_channels':2048,'fresh_C1_control':True}),flush=True)


if __name__=='__main__':main()
