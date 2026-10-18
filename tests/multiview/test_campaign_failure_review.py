import copy
import hashlib
import json

import pytest

from experiments.deformnext_retargeting.review_campaign_failures import review_historical_failures


def fixture(root):
    folder = root / 'provenance'
    folder.mkdir()
    log, output = folder/'old.log', folder/'replacement.json'
    log.write_text('Traceback: explicit old error\n')
    output.write_text(json.dumps({'status':'PASS'}))
    def ref(path):
        return {'relative_path':str(path.relative_to(root)),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    review = {'failed_job':'old','failure_log':ref(log),'successor_jobs':['replacement'],
              'evidence':[{**ref(output),'status':'PASS'}]}
    path = folder/'resolved-historical-queue-failures-v1.json'
    value = {'status':'PASS_RESOLVED_HISTORICAL_FAILURES','reviews':[review]}
    path.write_text(json.dumps(value))
    state = {'old':{'status':'FAILED'}, 'replacement':{'status':'completed'},
             'new-unreviewed':{'status':'FAILED'}}
    return path, value, state


def test_without_a_review_no_failure_is_implicitly_resolved(tmp_path):
    assert review_historical_failures(tmp_path,{'old':{'status':'FAILED'}})['unresolved_failures']==['old']


def test_review_preserves_history_and_never_resolves_a_new_failure(tmp_path):
    _,_,state=fixture(tmp_path)
    before=copy.deepcopy(state)
    assert review_historical_failures(tmp_path,state)=={
        'resolved_historical_failures':{'old':['replacement']},'unresolved_failures':['new-unreviewed']}
    assert state==before


@pytest.mark.parametrize('broken',['hash','replacement','status','duplicate','nonfailed'])
def test_review_rejects_unverified_resolution(tmp_path,broken):
    path,value,state=fixture(tmp_path)
    if broken=='hash':(tmp_path/'provenance/replacement.json').write_text('{}')
    elif broken=='replacement':state['replacement']['status']='pending'
    elif broken=='status':value['reviews'][0]['evidence'][0]['status']='FAILED'
    elif broken=='duplicate':value['reviews']*=2
    else:state['old']['status']='completed'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):review_historical_failures(tmp_path,state)
