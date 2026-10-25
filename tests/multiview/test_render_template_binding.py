"""Synthetic metadata and a tiny file; no render, geometry or model execution."""
from copy import deepcopy
import hashlib

import pytest

from experiments.deformnext_retargeting.render_template_binding import bind_render_template_inputs


def inputs(tmp_path):
    path=tmp_path/'environment.exr';path.write_bytes(b'synthetic environment identity')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    template=dict(environment_map=str(path),AO_seed=42,prediction_render_face_budget=200000,
                  views=[dict(camera_id=8)],scope='unchanged native rendering')
    original=dict(input_SHA256={str(path):digest})
    return template,original,digest


def test_real_legacy_schema_gets_original_pin_without_changing_render_settings(tmp_path):
    template,original,digest=inputs(tmp_path);before=deepcopy((template,original))
    bound=bind_render_template_inputs(template,original)
    assert (template,original)==before
    assert bound['input_SHA256']=={template['environment_map']:digest}
    assert {k:v for k,v in bound.items() if k!='input_SHA256'}==template
    assert bind_render_template_inputs(bound,original)==bound


@pytest.mark.parametrize('damage',['missing-original','changed-environment','conflicting-template'])
def test_original_environment_identity_cannot_be_replaced(tmp_path,damage):
    template,original,_=inputs(tmp_path)
    if damage=='missing-original':original['input_SHA256']={}
    elif damage=='changed-environment':
        from pathlib import Path
        Path(template['environment_map']).write_bytes(b'changed')
    else:template['input_SHA256']={template['environment_map']:'0'*64}
    with pytest.raises(ValueError):bind_render_template_inputs(template,original)
