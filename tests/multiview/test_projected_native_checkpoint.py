import hashlib
import json

import pytest

from experiments.deformnext_retargeting.evaluate_projected_native_checkpoint import BUNDLE_SCHEMA,validate_bundle
from test_projected_native_recipe import control


@pytest.mark.parametrize('arm',['C_TOP1_V','C_FIXED_TOPK'])
def test_saved_projected_model_requires_own_actual_config_and_checkpoint_pair(tmp_path,arm):
    cfg,reference=control(arm)
    path=tmp_path/'reference.json';path.write_text(json.dumps(reference))
    cfg['experiment'].update(reference_science_config=str(path),
        reference_science_config_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
    cfg['output_dir']='/actual/'+arm+'/native'
    root=cfg['output_dir']
    bundle=dict(schema=BUNDLE_SCHEMA,variant=arm,step=480,
        checkpoint=root+'/ckpts/denoiser_step0000480.pt',
        conditioner_checkpoint=root+'/ckpts/deform_conditioner_step0000480.pt',
        native_config=root+'/config.json',source_teacher_updates=0,
        native_task_objective='SS_flow_mse',native_validation=dict(step=480))
    assert validate_bundle(bundle,cfg)==arm
    bundle['conditioner_checkpoint']='/actual/other-model/deform_conditioner_step0000480.pt'
    with pytest.raises(ValueError):validate_bundle(bundle,cfg)
