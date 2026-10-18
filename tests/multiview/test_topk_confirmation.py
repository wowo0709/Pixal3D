from experiments.deformnext_retargeting.semantic_topk_confirmation import seed_config,make_jobs


def test_confirmation_keeps_original_initializer_and_budget(tmp_path):
    original={'training_seed':101,'trainer':{'args':{'max_steps':480,'batch_size_per_gpu':48,'batch_split':1,
        'continuation_ckpt':{'denoiser':'SS4210'}}},'experiment':{'arm':'top1-reference'},'num_gpus':2}
    cfg=seed_config(original,tmp_path,'top1-reference',202)
    assert cfg['training_seed']==cfg['experiment']['training_seed']==202 and original['training_seed']==101
    assert cfg['trainer']==original['trainer'] and cfg['num_gpus']==2
    assert cfg['load_dir']=='' and cfg['ckpt']=='none'
    assert 'seed202' in cfg['wandb_id'] and str(tmp_path/'runs_ss64/top1-reference')==cfg['default_output_dir']


def test_confirmation_is_an_independent_five_epoch_run_with_three_final_noises(tmp_path):
    jobs=make_jobs(tmp_path,tmp_path/'old','top1-reference',['2','3'],['development-parent'],202)
    assert len(jobs)==4 and jobs[0]['gpus']==['2','3']
    assert jobs[0]['depends_on']==['development-parent']
    assert all('resume' not in j['command'] for j in jobs)
    assert len([j for j in jobs if j['resource']=='evaluation'])==3
    assert all('480' in j['command'] for j in jobs[1:])


def test_fresh_confirmation_clears_inherited_resume_fields(tmp_path):
    source={'load_dir':'previous-trained-run','ckpt':'480','experiment':{'arm':'task-full'},
            'trainer':{'args':{'continuation_ckpt':{'denoiser':'original-SS4210'}}}}
    cfg=seed_config(source,tmp_path,'task-full',202)
    assert cfg['load_dir']=='' and cfg['ckpt']=='none'
    assert cfg['trainer']['args']['continuation_ckpt']['denoiser']=='original-SS4210'
    assert source['load_dir']=='previous-trained-run'
