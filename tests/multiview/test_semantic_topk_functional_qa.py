from experiments.deformnext_retargeting.semantic_topk_functional_qa import job,attention_probe_jobs


def test_functional_qa_reads_actual_raw480_after_training_on_any_assigned_free_gpu(tmp_path):
    q=job(tmp_path,tmp_path,'task-no-prior',101,tmp_path/'panel.json')
    assert q['depends_on']==['train-task-no-prior'] and q['resource']=='evaluation'
    assert 'gpu' not in q and 'gpus' not in q
    assert q['command'][q['command'].index('--checkpoint')+1].endswith('denoiser_step0000480.pt')
    assert 'train.py' not in q['command'] and 'resume' not in q['command']


def test_selected_second_seed_has_its_own_functional_checkpoint_binding(tmp_path):
    q=job(tmp_path,tmp_path/'seed-confirmation-v1','task-full',202,tmp_path/'panel.json')
    assert q['name']=='functional-QA-s202-task-full'
    assert q['depends_on']==['confirm-seed202-train-task-full']
    assert '/seed-confirmation-v1/' in q['command'][q['command'].index('--conditioner-checkpoint')+1]


def test_missing_attention_probes_preserve_existing_qa_and_only_use_idle_assigned_gpu(tmp_path):
    probes=attention_probe_jobs(tmp_path,tmp_path/'panel.json')
    assert len(probes)==2
    for name,q in zip(['static-c1','task-full'],probes):
        assert q['name']=='attention-statistics-s101-'+name and q['priority']==2
        assert q['depends_on']==['train-task-no-decoder','train-'+name]
        assert q['resource']=='evaluation' and 'gpus' not in q and 'gpu' not in q
        assert q['outputs']==[str(tmp_path/'attention-statistics-v1'/name/'summary.json')]
        assert '/functional-QA-v1/' not in q['outputs'][0]
        assert q['command'][q['command'].index('--checkpoint')+1].endswith('denoiser_step0000480.pt')
