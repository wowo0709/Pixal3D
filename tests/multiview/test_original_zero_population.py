from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting.original_zero_population import MODES, model_binding, original_weight_guard, validate_binding
from experiments.deformnext_retargeting.native_population_contract import ENGINEERING, FINAL
from test_native_population_contract import inputs as native_inputs


def inputs(scope=FINAL, alias='zero-source4'):
    original=native_inputs(scope)
    _,_,cfg,rows,panel,seeds,kw=original
    cfg=deepcopy(cfg);cfg['dataset']['name']='DeformNeXtSS64'
    cfg['validation']['dataset_args']['transport_arm']='v1'
    cfg['trainer']['args']['deform_conditioner']=None
    ckpt='/original/denoiser_step0004210.pt';config='/original/config.json'
    cfg['trainer']['args']['continuation_ckpt']={'denoiser':ckpt}
    hashes={ckpt:'a'*64,config:'b'*64}
    models={a:model_binding(a,config,ckpt,hashes) for a in MODES}
    binding=models[alias]
    plan=dict(population_scope=scope,expected_assets=len(panel),model_binding=binding,
        panel='/panel.json',seeds='/seeds.json',output='/output',dataset_roots=original[0]['dataset_roots'],
        population_inputs=deepcopy(original[0]['population_inputs']))
    values={'--stage':'ss64','--step':'0','--generation-batch':'8','--checkpoint':ckpt,'--arm-config':config,
        '--conditioning-mode':MODES[alias],'--panel-size':str(len(panel)),'--panel-file':plan['panel'],
        '--seed-map-file':plan['seeds'],'--output':plan['output'],'--expected-stage-root':'/repo',
        '--expected-entry-sha':'c'*64,'--cuda-memory-fraction':'.2','--config-root':'/configs'}
    plan['native_arguments']=[x for pair in values.items()for x in pair]
    controls=dict(status='FROZEN_ORIGINAL_ZERO_RGB_CONTROLS_BEFORE_TEST',heldout_model_quality_read=False,models=models)
    parent=kw['freeze'];parent['population']['generation_seeds']=seeds
    parent['original_zero_controls']=['source4-M0-M0-M0-M0','edited4-M1-M0-M0-M0']
    parity=dict(kw['parity'],model_binding=binding)
    args=dict(train=kw['train_rows'],validation=kw['validation_rows'],controls=controls,
              parent=parent,parity=parity,qa=kw['cache_QA'],qa_plan=kw['cache_QA_plan'])
    return plan,cfg,binding,rows,panel,seeds,args


@pytest.mark.parametrize('alias',MODES)
def test_original_four_RGB_inputs_and_zero_update_step_preserved(alias):
    p,c,b,r,ids,s,kw=inputs(alias=alias);before=deepcopy(c)
    projected=validate_binding(p,c,b,r,ids,s,**kw)
    assert c==before and projected['validation']['dataset_args']['expected_split']=='test'
    assert projected['trainer']==c['trainer'] and b['original_multiview_step']==4210
    assert p['native_arguments'][p['native_arguments'].index('--step')+1]=='0'
    assert b['conditioner_checkpoint']is None and b['conditioning_mode']==MODES[alias]


def test_engineering36_preserves_entire_original_validation_recipe():
    p,c,b,r,ids,s,kw=inputs(ENGINEERING)
    # Pass the same actual original validation data to both arguments.
    kw['validation']=r
    assert validate_binding(p,c,b,r,ids,s,**kw)==c


@pytest.mark.parametrize('damage',[
    'source3','wrong-anchor','not-original','source-weight','new-checkpoint','step4210','trained-source',
    'changed-config','one-load','tail','test-before-freeze','changed-native-population','different-noise',
    'GT-matching-switch','missing-assets','test-TRAIN-overlap','same-SS-different-config','panel-path',
])
def test_rejects_protocol_or_population_confounds(damage):
    p,c,b,r,ids,s,kw=inputs()
    def arg(flag,value):p['native_arguments'][p['native_arguments'].index(flag)+1]=value
    if damage=='source3':p['native_arguments']+=['--num-views','3']
    elif damage=='wrong-anchor':arg('--conditioning-mode','edited-rgb')
    elif damage=='not-original':b['schema']='trained'
    elif damage=='source-weight':b['conditioner_checkpoint']={'path':'/weight'}
    elif damage=='new-checkpoint':arg('--checkpoint','/trained/denoiser_step0000480.pt')
    elif damage=='step4210':arg('--step','4210')
    elif damage=='trained-source':c['trainer']['args']['deform_conditioner']={'name':'source'}
    elif damage=='changed-config':arg('--arm-config','/other/config.json')
    elif damage=='one-load':kw['parity']['independent_model_loads']=1
    elif damage=='tail':kw['parity']['tail_generation_batch']=0
    elif damage=='test-before-freeze':kw['controls']['heldout_model_quality_read']=True
    elif damage=='changed-native-population':kw['parent']['population']['rendered_cameras']=[8]
    elif damage=='different-noise':kw['parent']['population']['generation_seeds']={**s,ids[0]:999}
    elif damage=='GT-matching-switch':kw['qa']['matching_head_training_or_calibration_updates']=1
    elif damage=='missing-assets':r.pop()
    elif damage=='test-TRAIN-overlap':r[0]['source_uid']=kw['train'][0]['source_uid']
    elif damage=='same-SS-different-config':kw['controls']['models']['zero-edited4']['native_config']={'path':'/other','SHA256':'f'*64}
    elif damage=='panel-path':p['population_inputs']['panel']['path']='/different'
    with pytest.raises(ValueError):validate_binding(p,c,b,r,ids,s,**kw)


@pytest.mark.parametrize('damage',[None,'wrong-base','changed-LoRA','optimizer-step'])
def test_guard_checks_actual_loaded_tensor_values_and_evaluation_immutability(tmp_path,damage):
    import torch
    class SavedTensors:
        def __init__(self):self.values={'base':torch.tensor([1.]),'lora_B':torch.tensor([2.])}
        def state_dict(self):return self.values
    model=SavedTensors();path=tmp_path/'original.pt';torch.save(model.state_dict(),path)
    trainer=SimpleNamespace(models={'denoiser':model},optimizer=SimpleNamespace(state={}),step=0)
    def evaluate(t):
        if damage=='changed-LoRA':model.values['lora_B'].add_(1.)
        if damage=='optimizer-step':t.step=1
        return 'actual'
    quick=SimpleNamespace(evaluate_quick=evaluate);records=[]
    if damage=='wrong-base':model.values['base'].add_(1.)
    if damage is None:
        with original_weight_guard(quick,str(path),records):assert quick.evaluate_quick(trainer)=='actual'
        assert records[0]['original_full_base_and_LoRA_exact_before_evaluation']
        assert len(records[0]['actual_loaded_trainable_SHA256'])==64
    else:
        with pytest.raises(ValueError):
            with original_weight_guard(quick,str(path),records):quick.evaluate_quick(trainer)
        assert records==[]
    assert quick.evaluate_quick is evaluate
