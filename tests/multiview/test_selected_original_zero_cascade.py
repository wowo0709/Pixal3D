"""Original numerical statements are untouched; only metadata/device guards vary."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting.selected_original_zero_cascade_population import invoke_original,validate
from experiments.deformnext_retargeting.selected_final_population import compose_models
from test_selected_final_case_plans import inputs
from test_selected_final_population import evaluation


def case_inputs(tmp_path,alias='zero-source4'):
    from experiments.deformnext_retargeting.selected_final_case_plans import case_plans
    args=inputs(tmp_path,alias=alias);case=case_plans(**args);plan=case['phases'][0]['plan']
    models=compose_models(args['parent'],args['controls'])
    record=evaluation(models[alias],args['parent'],alias)
    return plan,record,dict(parent=args['parent'],controls=args['controls'])


@pytest.mark.parametrize('alias',['zero-source4','zero-edited4'])
def test_actual_selected_parent_keys_keep_original_zero_cascade_inputs_and_steps(tmp_path,alias):
    plan,record,kwargs=case_inputs(tmp_path,alias);before=deepcopy((plan,record,kwargs))
    result=validate(plan,record,**kwargs)
    assert (plan,record,kwargs)==before
    assert result['input_states']==(['M0']*4 if alias=='zero-source4' else ['M1','M0','M0','M0'])
    assert result['visibility_mask_used'] is False


@pytest.mark.parametrize('damage',['wrong-alias','fake-prefix','original-weights-not-checked','source3','GT-depth',
    'stage','parent-original-stage','pose','input-state','different-SS','render-step','frozen-population'])
def test_wrong_original_protocol_cannot_use_a_selected_parent(tmp_path,damage):
    plan,record,kwargs=case_inputs(tmp_path)
    if damage=='wrong-alias':plan['original_control_alias']='V1'
    elif damage=='fake-prefix':record['native_SS_artifacts']['archived_asset_count']=32
    elif damage=='original-weights-not-checked':record['actual_original_full_base_and_LoRA_validation']['original_full_base_and_LoRA_exact_before_evaluation']=False
    elif damage=='source3':plan['row']['conditioning_camera_ids']=[9,10,11]
    elif damage=='GT-depth':plan['row']['target_anchor_kind']='depth'
    elif damage=='stage':plan['stages']['shape1024']['checkpoint']='/other.pt'
    elif damage=='parent-original-stage':kwargs['parent']['original_stages']['pbr1024']['checkpoint']['SHA256']='f'*64
    elif damage=='pose':plan['row']['source_anchor_transform_matrix'][0][3]=.2
    elif damage=='input-state':plan['input_states']=['M1','M0','M0','M0']
    elif damage=='different-SS':plan['original_SS_checkpoint']='/other/denoiser_step0004210.pt'
    elif damage=='render-step':plan['stages']['pbr1024']['original_multiview_step']=1900
    else:plan['row']['sample_id']='validation-sample'
    with pytest.raises(ValueError):validate(plan,record,**kwargs)


@pytest.mark.parametrize('behavior',['success','skip','twice','original-failure'])
def test_scoped_metadata_callback_and_device_AST_restore_on_success_or_failure(tmp_path,behavior):
    plan,record,kwargs=case_inputs(tmp_path)
    ss=tmp_path/'evaluation.json';ss.write_text(json.dumps(record));plan['SS_evaluation']=str(ss)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));source=tmp_path/'original.py'
    call = "validate_zero_shot_cascade(p,e)\n    events.append('original-body')"
    if behavior=='skip':call="events.append('original-body')"
    elif behavior=='twice':call="validate_zero_shot_cascade(p,e)\n    validate_zero_shot_cascade(p,e)"
    elif behavior=='original-failure':call="raise RuntimeError('original-body-failed')"
    source.write_text("import json,sys\nfrom pathlib import Path\ndef validate_zero_shot_cascade(p,e):\n    raise AssertionError('unscoped old32 guard')\n"
        "def main():\n    gpu='6'\n    if gpu not in {'0','1','2','3'}:\n        raise AssertionError('unassigned')\n"
        "    p=json.loads(Path(sys.argv[-1]).read_text())\n    e=json.loads(Path(p['SS_evaluation']).read_text())\n    "+call+'\n')
    old=lambda *_:None;events=[];core=SimpleNamespace(__file__=str(source),__name__='owned_original',
        validate_zero_shot_cascade=old,events=events);old_argv=sys.argv;before=source.read_bytes()
    if behavior=='success':
        proof=invoke_original(core,path,plan,record,devices=['0','6'],**kwargs)
        assert proof['all_other_AST_statements_unchanged'] and events==['original-body']
    else:
        with pytest.raises(RuntimeError if behavior=='original-failure' else ValueError):
            invoke_original(core,path,plan,record,devices=['0','6'],**kwargs)
    assert core.validate_zero_shot_cascade is old and sys.argv is old_argv and source.read_bytes()==before
