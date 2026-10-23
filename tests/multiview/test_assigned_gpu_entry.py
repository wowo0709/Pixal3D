import ast
from pathlib import Path

import pytest

from experiments.deformnext_retargeting.assigned_gpu_entry import adapt_guard,validate_assignment


def test_explicit_assignment_requires_both_plan_and_scheduler():
    assert validate_assignment({'assigned_GPUs':['6','7']},'7','6,7')==['6','7']
    for visible,allowed in [('4','6,7'),('6','0,1,2,3'),('6,7','6,7'),('','6,7')]:
        with pytest.raises(ValueError):validate_assignment({'assigned_GPUs':['6','7']},visible,allowed)


@pytest.mark.parametrize('devices',[[],['6','6'],[6,7],['6','x'],None])
def test_rejects_invalid_plan_assignment(devices):
    with pytest.raises(ValueError):validate_assignment({'assigned_GPUs':devices},'6','6,7')


def test_guard_change_does_not_change_environment_check_or_body():
    source="""def check(gpu,allowed):
    if gpu not in {'0','1','2','3'} or gpu not in allowed:
        raise ValueError('assignment')
    return (gpu, 11 * 13)
"""
    tree,proof=adapt_guard(source,['6','7']);scope={};exec(compile(tree,'<guard-test>','exec'),scope)
    assert scope['check']('6',['6','7'])==('6',143)
    for gpu,allowed in [('0',['0','6']),('6',['0','1']),('4',['4','6'])]:
        with pytest.raises(ValueError):scope['check'](gpu,allowed)
    assert proof['all_other_AST_statements_unchanged']


@pytest.mark.parametrize('name',[
    'replay_r_pbr_on_generated_shape.py','render_cascade_artifact_v2.py',
    'score_cascade_fixed_reference.py','evaluate_official_perceptual.py',
])
def test_real_pinned_entry_AST_is_identical_except_the_guard(name):
    source=(Path(__file__).parents[2]/'experiments/deformnext_retargeting'/name).read_text()
    changed,proof=adapt_guard(source,['6','7'])
    for node in ast.walk(changed):
        if (isinstance(node,ast.Compare)and isinstance(node.left,ast.Name)and node.left.id=='gpu'
                and isinstance(node.comparators[0],ast.Set)
                and {x.value for x in node.comparators[0].elts}=={'6','7'}):
            original=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Compare)
                and isinstance(n.left,ast.Name)and n.left.id=='gpu'and isinstance(n.comparators[0],ast.Set))
            node.comparators[0]=original.comparators[0]
    assert ast.dump(changed,include_attributes=False)==ast.dump(ast.parse(source),include_attributes=False)
    assert proof['changed_membership_literals']==1


@pytest.mark.parametrize('source',['x=1',"if other not in {'0','1','2','3'}: pass", "if gpu not in {'0','1'}: pass"])
def test_refuses_unknown_source_guards(source):
    with pytest.raises(ValueError):adapt_guard(source,['6','7'])
