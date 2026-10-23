import copy
import hashlib
import json

import pytest

from experiments.deformnext_retargeting import evaluate_raw_correspondence_checkpoint as raw
from experiments.deformnext_retargeting import evaluate_unified_design_checkpoint as design
from test_raw_correspondence_native import control as raw_control
from test_unified_design_controls import control as design_control


@pytest.mark.parametrize('arm',['C_RAW','C_SPATIAL','C_D1_COS',
    'FULL_QK_ONLY','FULL_VOXEL_Q','FULL_MEAN_MATCH','C_DECODER_NO_R'])
def test_actual_checkpoint_identity_is_specific_to_the_physical_control(tmp_path,arm):
    evaluator,factory=(raw,raw_control) if arm in {'C_RAW','C_SPATIAL','C_D1_COS'} else (design,design_control)
    cfg,reference=factory(arm)
    path=tmp_path/'reference.json';path.write_text(json.dumps(reference))
    cfg['experiment'].update(reference_science_config=str(path),
        reference_science_config_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
    root='/actual/'+arm+'/native';cfg['output_dir']=root
    bundle=dict(schema=evaluator.BUNDLE_SCHEMA,variant=arm,step=480,
        checkpoint=root+'/ckpts/denoiser_step0000480.pt',
        conditioner_checkpoint=root+'/ckpts/deform_conditioner_step0000480.pt',
        native_config=root+'/config.json',source_teacher_updates=0,
        native_task_objective='SS_flow_mse',native_validation=dict(step=480))
    assert evaluator.validate_bundle(bundle,cfg)==arm
    for field,wrong in [('conditioner_checkpoint','other-model.pt'),('schema','other-family'),('variant','FULL')]:
        changed=copy.deepcopy(bundle);changed[field]=wrong
        with pytest.raises(ValueError):evaluator.validate_bundle(changed,cfg)
