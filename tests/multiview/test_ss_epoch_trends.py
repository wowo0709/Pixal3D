import json

from experiments.deformnext_retargeting.summarize_ss_epoch_trends import completed_lines,record_epoch_contrasts


def test_inflight_jsonl_tail_is_not_a_completed_validation(tmp_path):
    path=tmp_path/'validation.jsonl';path.write_text('{"step":96}\n{"step":192')
    assert completed_lines(path)==[{'step':96}]


def test_only_matching_completed_epochs_produce_interim_comparisons(tmp_path):
    ids=[f'asset{i}' for i in range(32)]
    for name,steps,tp in [('p1-native-grid',[0,96,192],10),('p2-matcher-grid',[0,96],11)]:
        run=tmp_path/'retarget-native-v3/runs_ss64'/name;run.mkdir(parents=True)
        (run/'config.json').write_text(json.dumps({'validation':{'quick_metrics':{'sampling_steps':12}}}))
        rows=[{'step':step,'quick_per_asset':[{'sample_id':sid,'source_cluster_id':'source'+str(i),
            'seed':i,'tp':tp,'fp':5,'fn':20-tp} for i,sid in enumerate(ids)]} for step in steps]
        (run/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    outputs=record_epoch_contrasts(tmp_path)
    assert len(outputs)==1
    record=json.loads(outputs[0].read_text())
    assert record['step']==96 and record['assets']==32
    assert record['status']=='interim' and record['selection_use'] is False
    assert record['fixed_final128_three_noise'] is False
    assert record_epoch_contrasts(tmp_path)==[]
