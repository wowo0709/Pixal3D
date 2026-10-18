import copy
import json
from pathlib import Path

import pytest

from experiments.deformnext_retargeting.prepare_ss_seed_confirmation import ARMS,make_config,check_matched,build_jobs


def recipes(root,work):
    base={'models':{'denoiser':{'name':'SS','args':{}}},'dataset':{'name':'DeformNeXtSS','args':{'transport_arm':'v1','expected_split':'train'}},
        'trainer':{'name':'Flow','args':{'deform_conditioner':None,'optimizer':{'lr':1e-5},'max_steps':480,'batch_split':1,'batch_size_per_gpu':96}},
        'validation':{'dataset_args':{'transport_arm':'v1','expected_split':'validation'},'quick_metrics':{'seed':42}},
        'data_dir':'fixed-train-manifest','experiment':{'training_assets':9149,'pilot_steps':96,'planned_total_steps':480,'pilot_baseline_file':'old'}}
    result={}
    for arm in ARMS:
        saved=copy.deepcopy(base)
        for args in [saved['dataset']['args'],saved['validation']['dataset_args']]:args['transport_arm']=arm
        if arm!='v1':
            saved['trainer']['args']['deform_conditioner']={'arm':arm,'implementation':'r9-native-v5','topk':64,'score_temperature':.005}
            saved['trainer']['args']['model_learning_rates']={'deform_conditioner':1e-4}
            for args in [saved['dataset']['args'],saved['validation']['dataset_args']]:
                args.update(transport_implementation='r9-native-v5',expected_q2c_sha256='frozen',match_root='same-R9-cache')
        result[arm]=make_config(saved,arm,root,work)
    return result


def test_all_arms_use_same_native_training_core_and_disable_pilot_resume(tmp_path):
    configs=recipes(tmp_path/'study',tmp_path/'work');check_matched(configs)
    for cfg in configs.values():
        assert cfg['load_dir']=='' and cfg['ckpt']=='none'
        assert cfg['experiment']['auto_continue_after_pilot'] is False
        assert cfg['experiment']['required_microbatch_per_gpu']==48
        assert not {'pilot_steps','planned_total_steps','pilot_baseline_file'}&cfg['experiment'].keys()
    changed=copy.deepcopy(configs);changed['c1']['trainer']['args']['optimizer']['lr']=2e-5
    with pytest.raises(ValueError,match='optimizer'):check_matched(changed)
    changed=copy.deepcopy(configs);changed['c1']['trainer']['args']['deform_conditioner']['score_temperature']=.1
    with pytest.raises(ValueError,match='same frozen matcher'):check_matched(changed)
    changed=copy.deepcopy(configs);changed['c1']['dataset']['args']['match_root']='other-cache'
    with pytest.raises(ValueError,match='caches differ'):check_matched(changed)
    changed=copy.deepcopy(configs)
    for cfg in changed.values():cfg['experiment'].pop('required_microbatch_per_gpu')
    with pytest.raises(ValueError,match='silently fall back'):check_matched(changed)


def test_confirmation_jobs_preserve_GPU_pairs_and_all_nine_final_evaluations(tmp_path):
    folder=tmp_path/'ss-seed-confirmation-v1';work=tmp_path/'work';configs=recipes(folder,work)
    for arm,cfg in configs.items():
        p=folder/'configs'/(arm+'.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(cfg))
    jobs=build_jobs(tmp_path,work);names={job['name']:job for job in jobs}
    train=[job for job in jobs if job['name'].startswith('train-')]
    assert len(train)==3 and all(job['resource']=='exclusive' and len(job['gpus'])==2 and set(job['gpus'])<=set(range(4)) for job in train)
    assert names['train-ss-original-seed101-v1']['gpus']==[2,3]
    assert 'final128-seed101-c1-noise-2' in names['train-ss-original-seed101-b3']['depends_on']
    final=[job for job in jobs if job['name'].startswith('final128-')]
    assert len(final)==9
    for job in final:
        arm=job['name'].split('-')[2]
        assert ('--conditioner-checkpoint' in job['command'])==(arm!='v1')
        assert '--generation-batch' in job['command']
    assert len(names['summarize-original-SS-seed101-confirmation']['depends_on'])==9
