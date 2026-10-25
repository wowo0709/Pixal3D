"""One owner per case, NN-priority preservation and cross-queue metric gates."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.final_fixed_cascade_placement import place_jobs
from test_final_fixed_ablation_cascade import pbr


def inputs():
    pre='/prepared/file4-closure.json'; gate='/active/file4-closure.json'
    cases=[dict(name=f'case{i}',resource='exclusive',priority=30,depends_on=['PBR202'],
        requires=[pre],outputs=[f'/results/{i}/file4-closure.json'],command=['native',str(i)])
        for i in range(128)]
    phases=[dict(name=name,resource=resource,requires=[pre],depends_on=deps,
        outputs=[f'/metrics/{name}/file4-closure.json'],command=['metric',name])
        for name,resource,deps in [('coverage','cpu',[j['name'] for j in cases]),
                                 ('official','exclusive',['coverage']),('summary','cpu',['official'])]]
    return cases,phases,dict(main_training_job='PBR202',common_PBR=pbr(),
        main_devices=['10','11'],auxiliary_devices=['12'],
        prelaunch_receipt=pre,activation_receipt=gate)


def test_same128_cases_have_single_owner_and_unchanged_native_commands():
    cases,phases,kwargs=inputs();before=deepcopy((cases,phases,kwargs))
    result=place_jobs(cases,phases,**kwargs)
    assert (cases,phases,kwargs)==before
    assert len(result['main'])==66 and len(result['auxiliary'])==65
    actual={j['name']:j for j in result['main']+result['auxiliary']}
    assert len(actual)==131 and set(actual)=={j['name'] for j in cases+phases}
    for row in cases+phases:
        assert actual[row['name']]['command']==row['command']
        assert actual[row['name']]['outputs']==row['outputs']
        assert kwargs['activation_receipt'] in actual[row['name']]['requires']
    assert all(j['depends_on']==['PBR202'] for j in result['main'][:64])
    assert all(j['depends_on']==[] and j['priority']==30 for j in result['auxiliary'][:64])
    assert actual['official']['priority']==25 and actual['official'] in result['auxiliary']
    assert actual['coverage']['requires'][2:]==[j['outputs'][0] for j in cases]
    assert actual['summary']['requires'][-1]==actual['official']['outputs'][0]


@pytest.mark.parametrize('damage',['partial-panel','duplicate-case','duplicate-output',
    'foreign-model','unfinished-common-PBR','overlap-GPU','implicit-GPU','wrong-case-dependency',
    'dropped-case-receipt','partial-coverage','summary-before-official'])
def test_placement_cannot_change_the_experiment_or_let_metrics_skip_work(damage):
    cases,phases,kwargs=inputs()
    if damage=='partial-panel':cases.pop()
    elif damage=='duplicate-case':cases[-1]['name']=cases[0]['name']
    elif damage=='duplicate-output':cases[-1]['outputs']=cases[0]['outputs']
    elif damage=='foreign-model':kwargs['common_PBR']['mode']='vanilla'
    elif damage=='unfinished-common-PBR':kwargs['common_PBR']['step']=380
    elif damage=='overlap-GPU':kwargs['auxiliary_devices']=['10']
    elif damage=='implicit-GPU':cases[0]['gpu']='12'
    elif damage=='wrong-case-dependency':cases[0]['depends_on']=['unrelated-science']
    elif damage=='dropped-case-receipt':cases[0]['requires']=[]
    elif damage=='partial-coverage':phases[0]['depends_on'].pop()
    else:phases[2]['depends_on']=[]
    with pytest.raises(ValueError):place_jobs(cases,phases,**kwargs)
