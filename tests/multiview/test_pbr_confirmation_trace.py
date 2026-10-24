"""Native PBR trace shape and separate raw-byte/parsed-row identities."""
from copy import deepcopy
import hashlib
import json

import pytest

from experiments.deformnext_retargeting.collect_selected_pbr_seed_confirmation import (
    training_trace_digest,validate_trace,
)


def trace():
    return [dict(step=i,asset_ids=[f'asset{j}' for j in range(6)],
        x_t_SHA256=f'{i:064x}',t_SHA256=f'{i+1:064x}',global_SHA256=f'{i+2:064x}')
        for i in range(1,1901)]


def test_actual_native_flat_micro6_records_are_accepted():
    rows=trace()
    validate_trace(rows)
    assert len(training_trace_digest(rows))==64


def test_raw_file_hash_and_parsed_record_identity_have_distinct_roles():
    rows=trace()
    compact='\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n'
    spaced='\n'.join(json.dumps(dict(reversed(list(r.items())))) for r in rows)+'\n'
    assert hashlib.sha256(compact.encode()).hexdigest()!=hashlib.sha256(spaced.encode()).hexdigest()
    first=[json.loads(r) for r in compact.splitlines()]
    second=[json.loads(r) for r in spaced.splitlines()]
    assert training_trace_digest(first)==training_trace_digest(second)
    assert training_trace_digest(first)!=hashlib.sha256(compact.encode()).hexdigest()


@pytest.mark.parametrize('field',['asset_ids','x_t_SHA256','t_SHA256','global_SHA256'])
def test_actual_training_input_changes_cannot_keep_the_snapshot_identity(field):
    rows=trace();changed=deepcopy(rows)
    if field=='asset_ids':changed[437][field][2]='other-actual-asset'
    else:changed[437][field]='a'*64
    assert training_trace_digest(rows)!=training_trace_digest(changed)


@pytest.mark.parametrize('damage',['nested','string','short-batch','empty-ID','wrong-ID-type',
    'incomplete','duplicate-step','boolean-step','invalid-digest'])
def test_incomplete_or_non_native_trace_records_are_rejected(damage):
    rows=trace()
    if damage=='nested':rows[0]['asset_ids']=[rows[0]['asset_ids']]
    elif damage=='string':rows[0]['asset_ids']='abcdef'
    elif damage=='short-batch':rows[0]['asset_ids'].pop()
    elif damage=='empty-ID':rows[0]['asset_ids'][0]=''
    elif damage=='wrong-ID-type':rows[0]['asset_ids'][0]=1
    elif damage=='incomplete':rows.pop()
    elif damage=='duplicate-step':rows[1]['step']=1
    elif damage=='boolean-step':rows[0]['step']=True
    else:rows[0]['x_t_SHA256']='z'*64
    with pytest.raises(ValueError):validate_trace(rows)
