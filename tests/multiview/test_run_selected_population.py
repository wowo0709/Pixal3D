"""Execution admission only: no CUDA context, neural model or GPU launch."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.deformnext_retargeting import run_selected_population as runner
from experiments.deformnext_retargeting.pipeline_population_contract_v2 import ARMS, ENGINEERING, FINAL
from experiments.deformnext_retargeting.selected_population_plan_factory import make_plan
from test_selected_population_plan_factory import inputs


def plans(tmp_path, arm='E_LEARNED_RAW', scope=ENGINEERING):
    request, kwargs = inputs(tmp_path,arm,scope)
    request.update(repo=str(Path(runner.__file__).resolve().parents[2]), canonical_local_root='/outputs')
    population=make_plan(request,**kwargs)
    context=dict(schema=runner.SCHEMA,name='actual-assigned-population',repo=population['repo'],python=population['python'],
        canonical_local_root=population['canonical_local_root'],population_plan=dict(path='/plans/population.json'),
        assigned_GPUs=['0','6'],CPU_affinity_per_GPU={'0':[10,11],'6':[12,13]},
        closure_command=[population['python'],'/machine/closure.py'],minimum_free_GPU_bytes=64*2**30,
        manifest_directory='/machine/manifests',source_SHA256={p:'a'*64 for p in runner.required_sources(arm)})
    context['source_SHA256']['/machine/closure.py']='a'*64
    return context,population


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('scope',[ENGINEERING,FINAL])
def test_all_families_use_the_same_original_population_evaluator_with_explicit_assignment(tmp_path,arm,scope):
    context,population=plans(tmp_path,arm,scope);before=deepcopy((context,population))
    command=runner.validate_execution(context,population,gpu='6',allowed='0,6')
    assert command==[context['python'],'-u','-m',runner.MODULE,'--plan','/plans/population.json']
    assert (context,population)==before and population['model_binding']['module_arm']==arm


@pytest.mark.parametrize('damage',['GPU-group','not-authorized','not-in-plan','implicit-affinity','wrong-affinity',
    'duplicate-affinity','bool-affinity','duplicate-GPU','float-size','micro4','scope','no-determinism','legacy-mode',
    'different-repo','outside-output','same-output','unpinned-closure','wrong-python','source','missing-parity','source-conflict'])
def test_incomplete_or_different_execution_cannot_launch_a_child(tmp_path,damage):
    c,p=plans(tmp_path,scope=FINAL);gpu='6';allowed='0,6'
    if damage=='GPU-group':gpu='0,6'
    elif damage=='not-authorized':allowed='0'
    elif damage=='not-in-plan':p['assigned_GPUs']=['0']
    elif damage=='implicit-affinity':c['CPU_affinity_per_GPU']={}
    elif damage=='wrong-affinity':c['CPU_affinity_per_GPU']['7']=c['CPU_affinity_per_GPU'].pop('6')
    elif damage=='duplicate-affinity':c['CPU_affinity_per_GPU']['6']=[1,1]
    elif damage=='bool-affinity':c['CPU_affinity_per_GPU']['6']=[True]
    elif damage=='duplicate-GPU':c['assigned_GPUs']=['6','6']
    elif damage=='float-size':p['expected_assets']=508.
    elif damage=='micro4':p['generation_batch_size']=4
    elif damage=='scope':p['population_scope']='unchecked-test'
    elif damage=='no-determinism':p['deterministic_algorithms']=False
    elif damage=='legacy-mode':p['mode']='archive32'
    elif damage=='different-repo':c['repo']='/different/repo'
    elif damage=='outside-output':p['output']='/outside/native'
    elif damage=='same-output':p['wrapper_output']=p['output']
    elif damage=='unpinned-closure':c['source_SHA256'].pop('/machine/closure.py')
    elif damage=='wrong-python':c['python']='/other/python'
    elif damage=='source':c['source_SHA256'].pop(runner.required_sources('E_LEARNED_RAW')[0])
    elif damage=='missing-parity':p['engineering_parity'].pop('receipt')
    else:
        c['source_SHA256'][next(iter(p['source_SHA256']))]='b'*64
    with pytest.raises(ValueError):runner.validate_execution(c,p,gpu=gpu,allowed=allowed)
