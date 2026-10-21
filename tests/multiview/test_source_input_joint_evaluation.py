import json
from pathlib import Path

import pytest

from experiments.deformnext_retargeting.evaluate_source_input_joint import validate_completed_joint_inputs
from experiments.deformnext_retargeting.source_input_joint_runtime import TEACHER_INITIAL_FUNCTION


def write(path, value):
    path.write_text(json.dumps(value))


def completed(tmp_path):
    """Tiny file-binding unit fixture; never a real model/campaign artifact."""
    root = tmp_path/'unit-run'; (root/'ckpts').mkdir(parents=True)
    variant = 'source-contrast-NoDQ'
    cfg = dict(output_dir=str(root), experiment=dict(source_input_design=variant,
        additional_source_teacher_supervision=True, initial_function=TEACHER_INITIAL_FUNCTION))
    write(root/'config.json', cfg)
    write(root/'progress.json', dict(stage='completed',current=480,total=480))
    write(root/'startup.json', dict(status='PASS_ACTUAL_FRESH_MEMBER_FEATURE_SOURCE_RAY_SS_STARTUP',
        initial_function=TEACHER_INITIAL_FUNCTION,
        global_batch=96,microbatch=96,accumulation=1,world_size=1,
        trainable_NN_LoRA_tensors=360,trainable_source_tensors=19,optimizer_states=0))
    write(root/'source-input-joint-runtime.json', dict(status='PASS_COMPLETED_NATIVE_SOURCE_INPUT_PHASE',
        phase='science',source_input_design=variant,initialization=[dict(variant=variant,
            source_tensors=19,additional_TRAIN_M1_source9_10_supervision=True)]))
    denoiser, source = root/'ckpts/denoiser_step0000480.pt', root/'ckpts/deform_conditioner_step0000480.pt'
    # This helper checks file provenance. The actual native entry separately
    # performs strict tensor loading, source state identity and optimizer checks.
    denoiser.write_bytes(b'unit checkpoint binding');source.write_bytes(b'unit source binding')
    return root, cfg, denoiser, source


def test_binds_both_checkpoints_and_completion_files_read_only(tmp_path):
    root,cfg,d,s=completed(tmp_path)
    result, hashes=validate_completed_joint_inputs(root,root/'config.json',d,s,
        variant='source-contrast-NoDQ',step=480)
    assert result==cfg and len(hashes)==6
    assert str(d)in hashes and str(s)in hashes


@pytest.mark.parametrize('damage',['running','wrong-source','wrong-denoiser','teacher-step','prototype',
                                   'different-variant','wrong-budget','undeclared-supervision'])
def test_rejects_partial_pilot_mixed_run_and_unmatched_protocol_before_gpu(tmp_path,damage):
    root,cfg,d,s=completed(tmp_path); step=480
    if damage=='running':write(root/'progress.json',dict(stage='running',current=479,total=480))
    elif damage=='wrong-source':s=tmp_path/'conditioner_step0000112.pt'
    elif damage=='wrong-denoiser':d=tmp_path/'denoiser_step0004210.pt'
    elif damage=='teacher-step':step=112
    elif damage=='prototype':
        p=root/'source-input-joint-runtime.json';v=json.loads(p.read_text());v['phase']='capacity';write(p,v)
    elif damage=='different-variant':
        cfg['experiment']['source_input_design']='original-NoDQ';write(root/'config.json',cfg)
    elif damage=='wrong-budget':
        p=root/'startup.json';v=json.loads(p.read_text());v['global_batch']=48;write(p,v)
    else:
        cfg['experiment']['additional_source_teacher_supervision']=False;write(root/'config.json',cfg)
    with pytest.raises(ValueError):
        validate_completed_joint_inputs(root,root/'config.json',d,s,variant='source-contrast-NoDQ',step=step)
