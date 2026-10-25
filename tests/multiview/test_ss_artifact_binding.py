"""Synthetic nested files and real ClosedInputs checks; no neural generation."""
from copy import deepcopy
import hashlib
import json

import pytest

from experiments.deformnext_retargeting.cascade_view_roles import ClosedInputs
from experiments.deformnext_retargeting.ss_artifact_binding import closed_ss_artifact_spec


def inputs(tmp_path):
    native = tmp_path / 'native'
    nested = native / 'ss-artifacts'
    nested.mkdir(parents=True)
    evaluation = native / 'evaluation.json'
    evaluation.write_text('{"fixture":true}\n')
    artifact = nested / 'example.npz'
    artifact.write_bytes(b'synthetic artifact, not an actual occupancy archive')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = native / 'file4-closure.json'
    receipt.write_text(json.dumps(dict(status='PASS', local_equals_actual_file4_SHA256=True,
        files=[dict(relative_path=str(p.relative_to(tmp_path)),sha256=digest(p),bytes=p.stat().st_size)
               for p in [evaluation,artifact]])))
    spec = dict(path=str(evaluation),SHA256=digest(evaluation),receipt=str(receipt))
    row = dict(artifact='ss-artifacts/example.npz',artifact_SHA256=digest(artifact))
    return spec,row,ClosedInputs(tmp_path)


def test_nested_artifact_uses_actual_evaluation_receipt_without_mutation(tmp_path):
    spec,row,check=inputs(tmp_path)
    before=deepcopy((spec,row))
    out=closed_ss_artifact_spec(spec,row,check=check)
    assert (spec,row)==before
    assert out['path']==str(tmp_path/'native/ss-artifacts/example.npz')
    assert out['receipt']==spec['receipt'] and out['SHA256']==row['artifact_SHA256']
    assert not (tmp_path/'native/ss-artifacts/file4-closure.json').exists()
    assert set(check.inputs)=={spec['path'],out['path']}


@pytest.mark.parametrize('damage',['record-hash','evaluation-hash','changed-bytes','receipt-only-artifact'])
def test_changed_or_unjoined_inputs_are_rejected(tmp_path,damage):
    spec,row,check=inputs(tmp_path)
    if damage=='record-hash':row['artifact_SHA256']='0'*64
    elif damage=='evaluation-hash':spec['SHA256']='0'*64
    elif damage=='changed-bytes':(tmp_path/'native/ss-artifacts/example.npz').write_bytes(b'changed')
    else:
        p=tmp_path/'native/file4-closure.json'
        receipt=json.loads(p.read_text());receipt['files']=receipt['files'][1:]
        p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):closed_ss_artifact_spec(spec,row,check=check)


@pytest.mark.parametrize('path',['../../foreign.npz','/foreign.npz','ss-artifacts/example.txt'])
def test_artifact_paths_cannot_escape_or_change_type(tmp_path,path):
    spec,row,check=inputs(tmp_path);row['artifact']=path
    with pytest.raises(ValueError,match='relative recorded'):
        closed_ss_artifact_spec(spec,row,check=check)
