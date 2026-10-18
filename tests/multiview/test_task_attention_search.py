from experiments.deformnext_retargeting.extend_task_attention_search import configurations,jobs_for,ARMS


def base():
    return {'trainer':{'args':{'deform_conditioner':{'arm':'c1','score_temperature':.005},
        'continuation_ckpt':{'denoiser':'SS4210'},'max_steps':480}},
        'dataset':{'args':{'transport_arm':'c1'}},'validation':{'dataset_args':{'transport_arm':'c1'}},
        'experiment':{'head_sha256':'original4f'},'training_seed':101}


def test_task_recipe_controls_keep_budget_initializer_and_independent_switches(tmp_path):
    cfg=configurations(base(),tmp_path,.05)
    assert list(cfg)==ARMS
    specs={k:c['trainer']['args']['deform_conditioner'] for k,c in cfg.items()}
    assert 'task_attention' not in specs['static-c1']
    assert specs['task-full']['task_attention']=={'decoder':True,'score_prior':True,'chunk_size':64}
    assert specs['task-no-decoder']['task_attention']=={'decoder':False,'score_prior':True,'chunk_size':64}
    assert specs['task-no-prior']['task_attention']=={'decoder':True,'score_prior':False,'chunk_size':64}
    assert specs['task-native-spatial']['correspondence_mode']=='native-spatial'
    for name,c in cfg.items():
        assert c['training_seed']==101 and c['trainer']['args']['max_steps']==480
        assert c['trainer']['args']['continuation_ckpt']=={'denoiser':'SS4210'}
        assert c['experiment']['inference_gt_inputs'] is False
        assert c['experiment']['physical_deformation_query_channels']==0
        assert c['experiment']['rejection'] is False
        assert c['dataset']['args'].get('task_matching_features',False)==(name!='static-c1')


def test_all_five_trainings_and_fifteen_final_evaluations_are_required(tmp_path):
    recipes=configurations(base(),tmp_path,.05)
    jobs=jobs_for(recipes,tmp_path,tmp_path/'previous',[['0','1'],['2','3']])
    assert len(jobs)==20 and len({j['name'] for j in jobs})==20
    assert len([j for j in jobs if j['name'].startswith('train-')])==5
    assert len([j for j in jobs if j['name'].startswith('final-')])==15
    assert all(j['gpus'] in [['0','1'],['2','3']] for j in jobs if j['resource']=='exclusive')
    assert all('480' in j['command'] for j in jobs if j['resource']=='evaluation')
    assert all(len(j['depends_on'])>=1 for j in jobs)
