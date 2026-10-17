from pathlib import Path

import pytest

from experiments.deformnext_retargeting.native_module_study import ARMS,build_jobs,configs


def base_recipe():
    return {'experiment':{'head_sha256':'shared-head'},
        'dataset':{'args':{'match_root':'same-cache','expected_q2c_sha256':'shared-head'}},
        'validation':{'dataset_args':{'match_root':'same-cache'},'quick_metrics':{'sampling_steps':12}},
        'trainer':{'args':{'continuation_ckpt':{'denoiser':'original-base-lora'},
            'optimizer':{'args':{'lr':1e-5}},'model_learning_rates':{'deform_conditioner':1e-4}}}}


def test_five_arms_isolate_selection_rejection_attention_adapter_and_query(tmp_path):
    head={'head_sha256':'shared-head','reliability':{'threshold':.2763}}
    original=base_recipe();recipes=configs(original,tmp_path,head)
    assert list(recipes)==[x[0] for x in ARMS]
    modules=[c['trainer']['args']['deform_conditioner'] for c in recipes.values()]
    left,right=modules[0],modules[1]
    assert {k for k in left if left[k]!=right[k]}=={'reject','reject_threshold'}
    assert {k for k in modules[1] if modules[1][k]!=modules[2][k]}=={'arm'}
    assert {k for k in modules[2] if modules[2][k]!=modules[3][k]}=={'arm'}
    assert modules[4]['deformation_dim']==2048
    assert all('deformation_dim' not in m for m in modules[:4])
    for i,cfg in enumerate(recipes.values()):
        assert cfg['trainer']['args']['continuation_ckpt']=={'denoiser':'original-base-lora'}
        assert cfg['trainer']['args']['optimizer']==original['trainer']['args']['optimizer']
        assert cfg['dataset']['args']['match_root']=='same-cache'
        assert cfg['dataset']['args']['expected_q2c_sha256']=='shared-head'
        assert 'deformation_root' not in cfg['dataset']['args']
        assert cfg['trainer']['args']['max_steps']==480
        assert cfg['trainer']['args']['batch_size_per_gpu']*cfg['num_gpus']==96
        assert cfg['trainer']['args']['batch_split']==1
        assert cfg['experiment']['inference_gt_inputs'] is False
        assert cfg['dataset']['args']['transport_arm']==ARMS[i][1]
    assert 'deform_conditioner' not in original['trainer']['args']


def test_changed_shared_matcher_cannot_create_module_ladder(tmp_path):
    with pytest.raises(ValueError,match='frozen shared matcher'):
        configs(base_recipe(),tmp_path,{'head_sha256':'another-head'})


def test_queue_preserves_every_contrast_and_requires_final_sampling_repeats(tmp_path):
    jobs,inputs=build_jobs(Path('/data'),tmp_path,Path('/data/study'),['0','1'],['2','3'])
    training=[j for j in jobs if j['name'].startswith('train-native-module-')]
    assert len(training)==5
    assert len({j['name'] for j in jobs})==len(jobs)
    assert all(j['resource']=='exclusive' and len(j['gpus'])==2 for j in training)
    assert [j['gpus'] for j in training]==[['0','1'],['2','3'],['0','1'],['2','3'],['0','1']]
    for j in training:
        assert len(j['depends_on'])==3
        assert 'gate_file' not in j
        assert j['command'][j['command'].index('--mode')+1]=='train'
    comparisons={j['name']:j for j in jobs if j['name'].startswith('compare-')}
    for candidate,baseline in [('b2-top1','v1'),('b2-top1-r','b2-top1'),('b3-attention-r','b2-top1-r'),
        ('c1-retarget-r','b3-attention-r'),('c2-query-r','c1-retarget-r')]:
        assert f'compare-native-module-{candidate}-vs-{baseline}' in comparisons
    assert len([j for j in jobs if j['resource']=='evaluation'])==15
    assert all(len(value)==3 for value in inputs.values())
