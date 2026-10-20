from types import SimpleNamespace
import copy
import hashlib

import numpy as np
import pytest
import torch

from experiments.deformnext_retargeting.evaluate_ss_artifact_subset import capture_ss_artifacts,validate_subset_reference


def reference_pair():
    row=dict(sample_id='asset-a',seed=47,target_voxels_sha256='gt-sha',tp=3,fp=2,fn=1)
    meta=dict(stage='ss64',optimizer_updates_during_evaluation=0,checkpoint='NN480.pt',conditioner_checkpoint='source480.pt',
        arm_config='frozen.json',conditioning_mode='edited-rgb',target_modality='rgb',input_states=['M1','M0','M0','M0'],
        input_camera_ids=[8,9,10,11],manifest_sha256='manifest-sha',generation_batch_size=8,sampling_steps=12,guidance=3.)
    actual=dict(meta,per_asset=[row]);reference=dict(meta,per_asset=[row,dict(row,sample_id='asset-b')])
    captured=[dict(sample_id='asset-a',seed=47,target_voxels_SHA256='gt-sha',occupancy_counts=dict(tp=3,fp=2,fn=1))]
    return actual,reference,captured


def test_prefix_records_and_protocol_are_checked_without_equating_panel_aggregates():
    actual,reference,captured=reference_pair()
    actual['quick_metrics']={'voxel_iou':.5};reference['quick_metrics']={'voxel_iou':.7}
    validate_subset_reference(actual,reference,captured,count=1)
    changed=copy.deepcopy(actual);changed['generation_batch_size']=1
    with pytest.raises(ValueError,match='generation_batch_size'):validate_subset_reference(changed,reference,captured,count=1)
    changed=copy.deepcopy(reference);changed['per_asset'].reverse()
    with pytest.raises(ValueError,match='prefix records'):validate_subset_reference(actual,changed,captured,count=1)


@pytest.mark.parametrize('field,value',[('seed',48),('target_voxels_SHA256','other-GT'),('occupancy_counts',dict(tp=2,fp=2,fn=1))])
def test_archive_rejects_unpaired_noise_GT_or_prediction_counts(field,value):
    actual,reference,captured=reference_pair();captured[0][field]=value
    with pytest.raises(ValueError,match='occupancy or sample/noise'):validate_subset_reference(actual,reference,captured,count=1)


def test_capture_preserves_native_return_records_actual_prediction_bytes_and_decoder(tmp_path):
    latent=torch.tensor([[3.],[4.]])
    decoded=torch.full((2,1,64,64,64),-1.)
    decoded[0].flatten()[:5]=1;decoded[1].flatten()[2:6]=1
    original_decode=lambda value:decoded
    dataset=SimpleNamespace(rows=[dict(sample_id='asset-a')],decode_latent=original_decode)
    def metrics(trainer,ds,indices,config):
        assert torch.equal(ds.decode_latent(latent),decoded)
        return {'native':'unchanged'},np.zeros((2,2,3),dtype=np.uint8)
    quick=SimpleNamespace(_ss_metrics=metrics,panel_seeds=lambda *args:[47]);captured=[]
    with capture_ss_artifacts(quick,tmp_path,captured):
        result=quick._ss_metrics(None,dataset,[0],{})
    assert result[0]=={'native':'unchanged'} and quick._ss_metrics is metrics and dataset.decode_latent is original_decode
    assert captured[0]['occupancy_counts']==dict(tp=3,fp=2,fn=1)
    with np.load(tmp_path/captured[0]['artifact']) as saved:
        np.testing.assert_array_equal(saved['prediction_latent'],latent[0].numpy())
        np.testing.assert_array_equal(saved['target_latent'],latent[1].numpy())
        packed=np.packbits((decoded[0,0].numpy()>0).reshape(-1))
        np.testing.assert_array_equal(saved['prediction_occupancy_packed'],packed)
        assert hashlib.sha256(packed.tobytes()).hexdigest()==captured[0]['prediction_voxels_SHA256']


def test_failed_native_decode_restores_both_hooks_and_exports_no_artifact(tmp_path):
    class DecodeFailure(Exception):pass
    def broken(*args):raise DecodeFailure()
    dataset=SimpleNamespace(decode_latent=broken)
    def metrics(trainer,ds,indices,config):return ds.decode_latent(None)
    quick=SimpleNamespace(_ss_metrics=metrics);captured=[]
    with pytest.raises(DecodeFailure):
        with capture_ss_artifacts(quick,tmp_path,captured):quick._ss_metrics(None,dataset,[0],{})
    assert dataset.decode_latent is broken and quick._ss_metrics is metrics and not captured
    assert not (tmp_path/'ss-artifacts').exists()
