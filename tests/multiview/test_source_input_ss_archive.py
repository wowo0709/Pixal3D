import copy
import hashlib
import json

import numpy as np
import pytest

from experiments.deformnext_retargeting.cascade_view_roles import file_sha256
from experiments.deformnext_retargeting.evaluate_source_input_joint_archive import validate_arguments,validate_captured
from experiments.deformnext_retargeting.select_source_input_ss_archive import select


def fixture(tmp_path):
    """Synthetic binary grids only; no model prediction or campaign receipt."""
    root=tmp_path/'unit-parent';folder=root/'ss-artifacts';folder.mkdir(parents=True)
    rows=[];entries=[]
    for i in range(128):
        sid=f'asset-{i}';seed=1000+i
        pred=np.zeros((64,64,64),dtype=bool);gt=pred.copy();gt[2,4,6]=True
        if i:pred[2,4,6]=True
        if i%2:pred[3,5,7]=True
        pp,pg=np.packbits(pred.flatten()),np.packbits(gt.flatten())
        tp=int((pred & gt).sum());fp=int((pred & ~gt).sum());fn=int((~pred & gt).sum())
        path=folder/(sid+'.npz')
        np.savez_compressed(path,sample_id=np.asarray(sid),seed=np.int64(seed),occupancy_shape=np.array([64,64,64]),
            prediction_occupancy_packed=pp,target_occupancy_packed=pg,
            prediction_latent=np.array([i],dtype=np.float32),target_latent=np.array([i+1],dtype=np.float32))
        gtsha=hashlib.sha256(pg.tobytes()).hexdigest()
        rows.append(dict(sample_id=sid,seed=seed,source_cluster_id=f'cluster-{i}',tp=tp,fp=fp,fn=fn,target_voxels_sha256=gtsha))
        entries.append(dict(sample_id=sid,seed=seed,artifact='ss-artifacts/'+path.name,artifact_SHA256=file_sha256(path),
            prediction_voxels_SHA256=hashlib.sha256(pp.tobytes()).hexdigest(),target_voxels_SHA256=gtsha,occupancy_counts=dict(tp=tp,fp=fp,fn=fn)))
    result=dict(status='PASS',stage='ss64',step=480,optimizer_updates_during_evaluation=0,diagnostic_only=False,
        generation_batch_size=8,sampling_steps=12,guidance=3.,target_modality='rgb',
        input_states=['M1','M0','M0','M0'],input_camera_ids=[8,9,10,11],per_asset=rows,
        checkpoint='unit-NN480.pt',conditioner_checkpoint='unit-source480.pt',
        quick_metrics=dict(voxel_iou=.123),quick_panel_ids=[r['sample_id']for r in rows],
        source_input_joint=dict(variant='source-contrast-NoDQ',completed_native_joint_updates=480,source_tensors=19,
            actual_joint_SS_LoRA_and_source_finetuned=True,source_and_denoiser_same_completed_run=True,
            actual_joint_source_state_exact_after_evaluation=True),
        native_SS_artifacts=dict(status='PASS_ACTUAL_JOINT_SS128_SINGLE_EVALUATION_ARCHIVE',
            scope='development-validation128',archived_asset_count=128,artifacts=entries))
    parent=root/'evaluation.json';parent.write_text(json.dumps(result))
    panel=tmp_path/'unit-panel32.json';ids=[f'asset-{i}'for i in reversed(range(32))];panel.write_text(json.dumps(dict(sample_ids=ids)))
    seeds=tmp_path/'unit-seeds.json';seeds.write_text(json.dumps({r['sample_id']:r['seed']for r in rows}))
    return parent,panel,seeds,result


def test_cpu_subset_preserves_real_grid_and_latent_bytes_and_recomputes32_metrics(tmp_path):
    parent,panel,seeds,result=fixture(tmp_path);out=tmp_path/'unit-subset';v=select(parent,panel,seeds,out)
    ids=json.loads(panel.read_text())['sample_ids'];assert [r['sample_id']for r in v['per_asset']]==ids
    assert v['source_input_joint']==result['source_input_joint']
    assert v['quick_metrics']['voxel_iou']==pytest.approx(31/48)
    assert v['quick_metrics']!=result['quick_metrics']
    for entry in v['native_SS_artifacts']['artifacts']:
        assert file_sha256(out/entry['artifact'])==file_sha256(parent.parent/entry['artifact'])
        if entry['sample_id']=='asset-0':assert not entry['native_SS32_support']['nonempty_generation_input']
        else:assert entry['native_SS32_support']['native_shape512_SS32_tokens']==1
    assert v['native_SS_artifacts']['additional_neural_forward_or_inference_or_optimizer_updates']==0
    assert json.loads(parent.read_text())==result


@pytest.mark.parametrize('damage',['file-bytes','prediction-grid-hash','wrong-seed','duplicate-panel','not-joint','final-test'])
def test_refuses_corrupt_or_unmatched_selection_before_creating_output(tmp_path,damage):
    parent,panel,seeds,v=fixture(tmp_path)
    if damage=='file-bytes':
        with(parent.parent/v['native_SS_artifacts']['artifacts'][0]['artifact']).open('ab')as stream:stream.write(b'corrupt')
    elif damage=='prediction-grid-hash':v['native_SS_artifacts']['artifacts'][0]['prediction_voxels_SHA256']='0'*64
    elif damage=='wrong-seed':
        s=json.loads(seeds.read_text());s['asset-0']+=1;seeds.write_text(json.dumps(s))
    elif damage=='duplicate-panel':
        p=json.loads(panel.read_text());p['sample_ids'][-1]=p['sample_ids'][0];panel.write_text(json.dumps(p))
    elif damage=='not-joint':v['source_input_joint']['completed_native_joint_updates']=112
    else:v['native_SS_artifacts']['scope']='final-test508'
    if damage in {'prediction-grid-hash','not-joint','final-test'}:parent.write_text(json.dumps(v))
    out=tmp_path/'unit-invalid-subset'
    with pytest.raises(ValueError):select(parent,panel,seeds,out)
    assert not out.exists()


def test_native_archive_does_not_accept_mismatched_capture_counts(tmp_path):
    _,_,_,v=fixture(tmp_path);captured=copy.deepcopy(v['native_SS_artifacts']['artifacts'])
    validate_captured(v,captured);captured[0]['occupancy_counts']['fn']+=1
    with pytest.raises(ValueError,match='decode/archive'):validate_captured(v,captured)


def test_archive_cli_cannot_change_population_sampler_or_combine_controls():
    args=['--stage','ss64','--step','480','--expected-assets','128','--panel-size','128',
        '--generation-batch','8','--source-value-control','normal']
    for flag in ['--arm-config','--checkpoint','--conditioner-checkpoint','--completed-training-root',
                 '--source-input-design','--panel-file','--seed-map-file','--output']:args.extend([flag,'unit'])
    validate_arguments(args)
    with pytest.raises(ValueError):validate_arguments(args+['--generation-batch','1'])
    with pytest.raises(ValueError):validate_arguments(args+['--conditioning-mode','source-only'])
