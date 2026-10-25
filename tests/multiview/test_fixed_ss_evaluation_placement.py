"""Keep experiment identity and one-producer admission while moving pending SS work."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.fixed_ss_evaluation_placement import (
    EVALUATOR, VARIANTS, relocate_plan, successor_job,
)
from experiments.deformnext_retargeting.matching_v2_queue import choose_gpu, dependencies_ready


def inputs(variant='M_FIXED_SPATIAL', noise=0):
    old=f'/prepared/{variant}-noise{noise}.json'
    new=f'/placement/{variant}-noise{noise}.json'
    root=f'/results/{variant}/noise-{noise}'
    plan=dict(phase='evaluation',mode='final128',variant=variant,expected_assets=128,
        generation_batch_size=8,deterministic_algorithms=True,python='/python',
        assigned_GPUs=['6','7'],CPU_affinity_per_GPU={'6':[76,77],'7':[78,79]},
        command=['/python','-u','-m',EVALUATOR,'--plan',old],bundle=f'/model/{variant}/bundle.json',
        panel='/data/panel128.json',seeds=f'/data/noise{noise}.json',output=root+'/native',
        native_output=root+'/native',wrapper_output=root+'/wrapper',source_SHA256={'/source.py':'a'*64},
        input_SHA256={'/data.json':'b'*64},closed_inputs={'/data.json':'/data/file4-closure.json'},
        minimum_free_GPU_bytes=32*2**30,arbitrary_future_numerical_field={'temperature':.05})
    job=dict(name=variant+f'-old{noise}',resource='exclusive',priority=10,
        depends_on=['seal-'+variant],requires=['/prepared/file4-closure.json',f'/model/{variant}/file4-closure.json'],
        outputs=[root+'/native/file4-closure.json',root+'/wrapper/file4-closure.json'],
        command=['/python','-u','/worker.py','--plan',old,'--expected-self-sha','c'*64])
    return plan,job,dict(original_path=old,replacement_path=new,devices=['0','1','2','3'],
        cpu_affinity={str(i):[84+2*i,85+2*i] for i in range(4)})


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('noise',[0,1,2])
def test_all_nine_keep_model_population_noise_math_and_existing_collector_paths(variant,noise):
    plan,job,kw=inputs(variant,noise);before=deepcopy((plan,job))
    moved=relocate_plan(plan,**kw)
    actual=successor_job(job,state={'status':'pending'},name=variant+f'-main{noise}',
        replacement_path=kw['replacement_path'],relocated_plan=moved,
        producer_outputs={'seal-'+variant:[f'/model/{variant}/file4-closure.json']},
        main_training_job='PBR1900',prelaunch_receipt='/placement/file4-closure.json',
        activation_receipt='/activation/file4-closure.json')
    assert (plan,job)==before
    restored=deepcopy(moved)
    for k in ['assigned_GPUs','CPU_affinity_per_GPU','command']:restored[k]=plan[k]
    assert restored==plan
    assert actual['outputs']==job['outputs'] and set(job['requires'])<=set(actual['requires'])
    assert {'/placement/file4-closure.json','/activation/file4-closure.json'}<=set(actual['requires'])
    assert actual['command'][-2:]==job['command'][-2:]
    active={'PBR1900':dict(status='running',gpu='0,1,2,3',resource='exclusive')}
    assert not dependencies_ready(actual,active)
    assert choose_gpu(actual,['0','1','2','3'],active) is None
    active['PBR1900']['status']='completed'
    assert dependencies_ready(actual,active) and choose_gpu(actual,['0','1','2','3'],active)=='0'


@pytest.mark.parametrize('damage',['science','epoch-replay','partial-panel','batch-change',
    'nondeterministic','foreign-model','wrong-self-plan','overlapping-CPU','implicit-devices'])
def test_refuse_experiment_changes_and_implicit_or_overlapping_resources(damage):
    plan,job,kw=inputs()
    if damage=='science':plan['phase']='science'
    elif damage=='epoch-replay':plan['mode']='epoch-replay'
    elif damage=='partial-panel':plan['expected_assets']=32
    elif damage=='batch-change':plan['generation_batch_size']=4
    elif damage=='nondeterministic':plan['deterministic_algorithms']=False
    elif damage=='foreign-model':plan['variant']='E_FIXED_RAW'
    elif damage=='wrong-self-plan':plan['command'][-1]='/unrelated.json'
    elif damage=='overlapping-CPU':kw['cpu_affinity']['1']=[84,85]
    else:kw['devices']=[];kw['cpu_affinity']={}
    with pytest.raises(ValueError):relocate_plan(plan,**kw)


@pytest.mark.parametrize('state',['running','completed','FAILED','HELD'])
def test_never_move_an_active_finished_or_failed_producer(state):
    plan,job,kw=inputs();moved=relocate_plan(plan,**kw)
    with pytest.raises(ValueError):successor_job(job,state={'status':state},name='main',
        replacement_path=kw['replacement_path'],relocated_plan=moved,
        producer_outputs={'seal-M_FIXED_SPATIAL':['/model/M_FIXED_SPATIAL/file4-closure.json']},
        main_training_job='PBR1900',prelaunch_receipt='/placement/file4-closure.json',
        activation_receipt='/activation/file4-closure.json')


@pytest.mark.parametrize('damage',['missing-seal-receipt','different-output','sharing','attached'])
def test_successor_cannot_skip_provenance_change_output_or_share_with_training(damage):
    plan,job,kw=inputs();moved=relocate_plan(plan,**kw)
    if damage=='missing-seal-receipt':job['requires'].pop()
    elif damage=='different-output':job['outputs'][0]='/other/file4-closure.json'
    elif damage=='sharing':job['resource']='evaluation'
    else:job['attached_pid']=123
    with pytest.raises(ValueError):successor_job(job,state={'status':'pending'},name='main',
        replacement_path=kw['replacement_path'],relocated_plan=moved,
        producer_outputs={'seal-M_FIXED_SPATIAL':['/model/M_FIXED_SPATIAL/file4-closure.json']},
        main_training_job='PBR1900',prelaunch_receipt='/placement/file4-closure.json',
        activation_receipt='/activation/file4-closure.json')
