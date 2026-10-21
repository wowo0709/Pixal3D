"""RAW metric-name compatibility must never replace dataset/model/sampling data."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting.member_raw_metric_dispatch import raw_ss_metric_dispatch, RAW, SS


def test_only_selector_is_copied_and_native_result_objects_are_preserved():
    trainer,dataset=object(),object()
    quick={'enabled':True,'sampling_steps':12,'generation_batch_size':8}
    config={'dataset_name':RAW,'quick_metrics':quick}
    result=({'voxel_iou':.4},object(),['example'])
    calls=[]
    def original(t,d,c):
        calls.append((t,d,c))
        return result
    module=SimpleNamespace(evaluate_quick=original)
    events=[]
    with raw_ss_metric_dispatch(events,quick_module=module):
        assert module.evaluate_quick(trainer,dataset,config) is result
    assert module.evaluate_quick is original
    assert calls[0][0] is trainer and calls[0][1] is dataset
    assert calls[0][2]['quick_metrics'] is quick and calls[0][2]['dataset_name']==SS
    assert config['dataset_name']==RAW and len(events)==1


def test_other_dataset_selectors_delegate_the_original_configuration_object():
    config={'dataset_name':'DeformNeXtPBR1024'}
    calls=[]
    def original(t,d,c):calls.append(c);return 'unchanged'
    module=SimpleNamespace(evaluate_quick=original)
    events=[]
    with raw_ss_metric_dispatch(events,quick_module=module):
        assert module.evaluate_quick(None,None,config)=='unchanged'
    assert calls[0] is config and not events


def test_dispatcher_is_restored_after_native_evaluation_error():
    def original(t,d,c):raise RuntimeError('native error')
    module=SimpleNamespace(evaluate_quick=original)
    with pytest.raises(RuntimeError,match='native error'):
        with raw_ss_metric_dispatch([],quick_module=module):
            module.evaluate_quick(None,None,{'dataset_name':RAW})
    assert module.evaluate_quick is original


def test_real_native_dispatch_routes_RAW_to_the_same_SS_metric_function(monkeypatch):
    from experiments.deformnext_retargeting import quick_metrics as quick
    import torch
    class Dataset:
        rows=[]
        def __len__(self):return 0
    dataset=Dataset()
    trainer=SimpleNamespace(mix_precision_dtype=torch.bfloat16)
    calls=[]
    result=({'voxel_iou':.45},'literal-panel')
    monkeypatch.setattr(torch,'autocast',lambda *args,**kwargs:nullcontext())
    def metrics(t,d,indices,c):calls.append((t,d,indices,c));return result
    monkeypatch.setattr(quick,'_ss_metrics',metrics)
    config=dict(dataset_name=RAW,quick_metrics=dict(enabled=True,panel_size=0,sampling_steps=12))
    # The old dispatcher never reached the native SS metric for this name.
    assert quick.evaluate_quick(trainer,dataset,config)==({},None,[])
    assert not calls
    events=[]
    with raw_ss_metric_dispatch(events):
        actual=quick.evaluate_quick(trainer,dataset,config)
    expected=quick.evaluate_quick(trainer,dataset,dict(config,dataset_name=SS))
    assert actual==expected==({'voxel_iou':.45},'literal-panel',[])
    assert calls[0][0] is calls[1][0] is trainer and calls[0][1] is calls[1][1] is dataset
    assert calls[0][3] is calls[1][3] is config['quick_metrics']
    assert len(events)==1 and config['dataset_name']==RAW
