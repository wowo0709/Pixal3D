from pathlib import Path
import numpy as np
import pytest
import torch

from pixal3d.datasets.deformnext import DeformNeXtSS64
from pixal3d.datasets.deformnext_member_raw import DeformNeXtMemberRawSS64


def cache_fixture(tmp_path,monkeypatch):
    path=tmp_path/'input-features.npz'
    target=np.zeros((1024,2048),np.float16);target[:,1024:]=2
    source=np.ones((3,1024,2048),np.float16);source[...,1024:]=3
    np.savez(path,target_keys=target,source_keys=source,conditioning_input_sha256=np.asarray('same-RGB'))
    pack=dict(transport_target_keys=torch.from_numpy(target[:,:1024].copy()),
        transport_source_keys=torch.from_numpy(source[...,:1024].copy()),cond=torch.ones(4,3,2,2))
    monkeypatch.setattr(DeformNeXtSS64,'_common',lambda self,row:dict(pack))
    dataset=object.__new__(DeformNeXtMemberRawSS64);dataset._resolve=lambda value:Path(value)
    row=dict(transport_cache=str(path),conditioning_input_sha256='same-RGB')
    return dataset,row,pack,path,target,source


def test_adds_actual_RAW_context_without_requiring_SS_latent_or_geometry(tmp_path,monkeypatch):
    dataset,row,pack,path,target,source=cache_fixture(tmp_path,monkeypatch)
    item=dataset._common(row)
    assert torch.equal(item['transport_target_context'],torch.from_numpy(target))
    assert torch.equal(item['transport_source_context'],torch.from_numpy(source))
    assert item['cond'] is pack['cond'] and item['transport_target_keys'] is pack['transport_target_keys']
    assert not {'x_0','transport_deformation','transport_semantic_indices'}&item.keys()


@pytest.mark.parametrize('failure',['input-ID','FRAME','GLOBAL-nonfinite','forbidden-match'])
def test_rejects_wrong_identity_features_or_hidden_matching_sidecar(failure,tmp_path,monkeypatch):
    dataset,row,pack,path,target,source=cache_fixture(tmp_path,monkeypatch)
    if failure=='input-ID':row['conditioning_input_sha256']='different-RGB'
    elif failure=='FRAME':
        source[...,:1024]=2
        np.savez(path,target_keys=target,source_keys=source,conditioning_input_sha256=np.asarray('same-RGB'))
    elif failure=='GLOBAL-nonfinite':
        target[:,1024:]=np.nan
        np.savez(path,target_keys=target,source_keys=source,conditioning_input_sha256=np.asarray('same-RGB'))
    else:pack['transport_deformation']=torch.ones(1024,2048)
    with pytest.raises(ValueError):dataset._common(row)


@pytest.mark.parametrize('extra',[dict(match_root='/old-match'),dict(task_matching_features=True),
    dict(deformation_root='/GT-sidecar'),dict(deformation_query_channels=1024)])
def test_RAW_dataset_preserves_original_uncached_prediction_input_contract(extra,monkeypatch):
    arguments=dict(deformation_query_channels=2048,transport_arm='c1',transport_implementation='r9-native-v5',
        task_matching_features=False,match_root=None)
    called=[]
    monkeypatch.setattr(DeformNeXtSS64,'__init__',lambda self,*a,**kw:called.append(kw))
    dataset=DeformNeXtMemberRawSS64('manifest.jsonl',**arguments)
    assert dataset.deformation_query_channels==2048 and called[0]['match_root'] is None
    with pytest.raises(ValueError):DeformNeXtMemberRawSS64('manifest.jsonl',**dict(arguments,**extra))
