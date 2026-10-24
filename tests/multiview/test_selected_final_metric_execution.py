"""Real archived pixels/feature NPZ and entry AST checks; no neural forwards."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.deformnext_retargeting import assigned_gpu_entry,extract_official_inception,selected_inception_entry
from experiments.deformnext_retargeting import prepare_selected_final_metrics as prepare
from experiments.deformnext_retargeting import collect_selected_final_metrics as collect
from experiments.deformnext_retargeting.cascade_view_roles import ClosedInputs,file_sha256 as sha
from experiments.deformnext_retargeting.selected_campaign_resources import process_identity
from experiments.deformnext_retargeting.selected_final_metric_inputs import (expected_feature_observations,feature_observations,read_pixels,validate_images)
from experiments.deformnext_retargeting.run_selected_final_metric import verify_gpu_proof,verify_report
from test_selected_final_metric_summary import inputs


def archived(tmp_path):
    path=tmp_path/'shared/camera08.npz';path.parent.mkdir()
    prediction=np.stack([np.full((512,512),x,np.float32) for x in [.25,.6,.8]])
    reference=np.stack([np.full((512,512),x,np.float32) for x in [.3,.4,.5]])
    mask=np.zeros((512,512),np.float32);mask[100:400,100:350]=1
    np.savez(path,prediction=prediction,prediction_alpha=np.ones((512,512),np.float32),GT_shared_renderer=reference,GT_alpha=mask)
    receipt=dict(status='PASS',local_equals_actual_file4_SHA256=True,files=[dict(relative_path=str(path.relative_to(tmp_path)),sha256=sha(path),bytes=path.stat().st_size)])
    (path.parent/'file4-closure.json').write_text(json.dumps(receipt));return path,prediction,reference,mask


def test_real_native512_pixels_bind_both_CHW_feature_and_HWC_perceptual_conversions(tmp_path):
    import hashlib
    path,pred,reference,mask=archived(tmp_path);row=read_pixels(path,alias='V1',sid='sample',cluster='source',check=ClosedInputs(tmp_path))
    ch=(np.clip(pred,0,1)*255).round().astype(np.uint8);hw=ch.transpose(1,2,0).copy()
    assert row['pixel_observation']['prediction_CHW_uint8_SHA256']==hashlib.sha256(ch.tobytes()).hexdigest()
    assert row['image_audit']['converted_image_SHA256']['prediction_full']==hashlib.sha256(hw.tobytes()).hexdigest()
    assert row['pixel_observation']['prediction_CHW_uint8_SHA256']!=row['image_audit']['converted_image_SHA256']['prediction_full']
    assert row['image_audit']['GT_only_crop_box']==[70,70,380,430]
    observations=feature_observations([row]);expected=expected_feature_observations([row])
    assert observations[0]['array']=='prediction' and 'converted_CHW_uint8_SHA256' not in observations[0]
    assert expected[0]['converted_CHW_uint8_SHA256']==hashlib.sha256(ch.tobytes()).hexdigest()
    assert feature_observations([row],reference=True)[0]['array']=='GT_shared_renderer'


def test_changed_actual_archive_bytes_fail_before_any_metric_forward(tmp_path):
    path,*_=archived(tmp_path);original=path.read_bytes();path.write_bytes(original+b'altered archive bytes')
    with pytest.raises(ValueError,match='identity mismatch'):read_pixels(path,alias='V1',sid='s',cluster='c',check=ClosedInputs(tmp_path))


def test_small_or_missing_camera_population_cannot_masquerade_as_final508(tmp_path):
    path,*_=archived(tmp_path);row=read_pixels(path,alias='V1',sid='s',cluster='c',check=ClosedInputs(tmp_path))
    with pytest.raises(ValueError,match='1524'):validate_images([row],alias='V1',parent=dict(population=dict(sample_ids=['s'],source_clusters_by_sample={'s':'c'})),check=ClosedInputs(tmp_path))


def proof(kind):
    devices=['0','1','2','3','6','7'];source=Path(extract_official_inception.__file__) if kind=='inception' else Path(assigned_gpu_entry.__file__).with_name('evaluate_official_perceptual.py')
    adapter=selected_inception_entry.adapt_guard if kind=='inception' else assigned_gpu_entry.adapt_guard
    _,body=adapter(source.read_text(),devices);body.update(original_source=str(source),original_source_SHA256=sha(source))
    value=dict(actual_entry_success=True,physical_GPU='6',assigned_GPUs=devices,original_source_files_unmodified=True)
    if kind=='inception':value.update(body)
    else:value.update(numerical_AST_unchanged=True,adapted_modules={'experiments.deformnext_retargeting.evaluate_official_perceptual':body})
    context=dict(kind=kind,assigned_GPUs=devices,source_SHA256={str(source):sha(source)})
    return context,value,source


@pytest.mark.parametrize('kind',['perceptual','inception'])
def test_actual_original_metric_AST_changes_only_one_GPU_membership_literal(kind):
    context,value,source=proof(kind);before=source.read_bytes();verify_gpu_proof(context,value,'6');assert source.read_bytes()==before
    if kind=='inception':compile(selected_inception_entry.adapt_guard(source.read_text(),context['assigned_GPUs'])[0],str(source),'exec')


@pytest.mark.parametrize('kind',['perceptual','inception'])
@pytest.mark.parametrize('damage',['GPU','scope','failed','numerical','source','membership-count'])
def test_forged_or_wrong_device_numerical_execution_proof_is_rejected(kind,damage):
    context,value,source=proof(kind)
    if damage=='GPU':value['physical_GPU']='7'
    elif damage=='scope':value['assigned_GPUs']=['0']
    elif damage=='failed':value['actual_entry_success']=False
    else:
        body=value if kind=='inception' else value['adapted_modules']['experiments.deformnext_retargeting.evaluate_official_perceptual']
        if damage=='numerical':body['all_other_AST_statements_unchanged']=False
        elif damage=='source':body['original_source_SHA256']='f'*64
        else:body['changed_membership_literals']=2
    with pytest.raises(ValueError):verify_gpu_proof(context,value,'6')


@pytest.mark.parametrize('damage',['no-guard','duplicate-guard','other-expression'])
def test_inception_literal_adapter_refuses_ambiguous_or_unrecognized_original_source(damage):
    source="if visible[0] not in {'0','1','2','3'}: raise ValueError()"
    if damage=='no-guard':source='x=1'
    elif damage=='duplicate-guard':source=source+'\n'+source
    else:source=source.replace('visible[0]','other[0]')
    with pytest.raises(ValueError):selected_inception_entry.adapt_guard(source,['6','7'])


def feature_fixture(tmp_path):
    args=inputs(tmp_path,'V1','vanilla');parent=args['parent'];alias='V1';rows=[]
    for case,score in zip(args['cases'][alias].values(),[None]*508):
        for p in case['scored_image_observations']:
            rows.append(dict(arm=alias,sample_id=p['sample_id'],source_cluster_id=p['source_cluster_id'],camera_id=p['camera_id'],pixel_observation=p))
    context=dict(alias=alias,kind='inception',model_binding=case['frozen_model_binding'],native_plan={'SHA256':'a'*64})
    native=dict(weights_SHA256='b'*64,pool_role='generated-prediction');manifest=dict(images={alias:rows})
    path=tmp_path/'features.npz';np.savez(path,features=np.ones((1524,2048),np.float32),sample_id=[r['sample_id'] for r in rows],
        camera_id=[r['camera_id'] for r in rows],source_cluster_id=[r['source_cluster_id'] for r in rows])
    report=dict(status='PASS_ACTUAL_OFFICIAL_INCEPTION2048_FEATURE_POOL',scope='final-test508',images=1524,assets=508,
        package='torch-fidelity0.3.0',feature_layer='2048',feature_dtype='float32',fullframe=True,TF32=False,autocast=False,
        same_batch_repeat_exact=True,optimizer_updates=0,model_binding=context['model_binding'],weights_SHA256='b'*64,
        observations=expected_feature_observations(rows),features_SHA256=sha(path),input_plan_SHA256='a'*64,pool_role='generated-prediction')
    return context,native,manifest,parent,report,path


@pytest.mark.parametrize('damage',[None,'feature-bytes','row-order','image-conversion','weights','model','reference-role'])
def test_actual_feature_archive_rows_hashes_model_and_prediction_reference_roles(tmp_path,damage):
    context,native,manifest,parent,report,path=feature_fixture(tmp_path)
    if damage=='feature-bytes':path.write_bytes(path.read_bytes()+b'altered features')
    elif damage=='row-order':
        with np.load(path) as bank:values={k:bank[k].copy() for k in bank.files}
        values['camera_id']=values['camera_id'][::-1];np.savez(path,**values);report['features_SHA256']=sha(path)
    elif damage=='image-conversion':report['observations'][0]['converted_CHW_uint8_SHA256']='other-image'
    elif damage=='weights':report['weights_SHA256']='wrong-weight'
    elif damage=='model':report['model_binding']={}
    elif damage=='reference-role':report['pool_role']='shared-original-material-GT'
    if damage:
        with pytest.raises(ValueError):verify_report(context,native,manifest,parent,report,features=path)
    else:verify_report(context,native,manifest,parent,report,features=path)


def test_completed_queue_file_cannot_override_actual_live_wrapper(tmp_path):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True);identity=process_identity(child.pid)
    context=dict(python=sys.executable);record=dict(plan={'path':str(tmp_path/'plan.json')},states=str(tmp_path/'states.json'),jobs=str(tmp_path/'jobs.json'),job_name='metric')
    command=[sys.executable,'-u','-m','experiments.deformnext_retargeting.run_selected_final_metric','--plan',record['plan']['path']]
    Path(record['states']).write_text(json.dumps({'metric':dict(status='completed',pid=child.pid,returncode=0,command=command,gpu='6')}))
    audit=dict(worker_identity=identity,physical_GPU='6')
    try:
        with pytest.raises(ValueError,match='absent'):collect.completed_queue_worker(record,audit,context)
    finally:child.terminate();child.wait()


def test_metric_cross_queue_partial_append_leaves_activation_absent(tmp_path,monkeypatch):
    a=tmp_path/'a.json';b=tmp_path/'b.json';old={'name':'keep-old','command':['preserved']};a.write_text(json.dumps([old]));b.write_text(json.dumps([old]))
    activation=tmp_path/'not-yet-published/file4-closure.json';jobs=[dict(name='new-a',queue=str(a),requires=[str(activation)]),dict(name='new-b',queue=str(b),requires=[str(activation)])]
    actual=prepare.atomic
    def fail_second(path,value):
        if str(path)==str(b):raise OSError('fixture filesystem failure')
        actual(path,value)
    # Input order to append_jobs is intentionally independent of set order.
    # Either no queue or just the first successful queue may be installed.
    monkeypatch.setattr(prepare,'atomic',fail_second)
    with pytest.raises(OSError):prepare.append_jobs({},jobs)
    assert not activation.exists()
    for p in [a,b]:assert json.load(open(p))[0]==old
