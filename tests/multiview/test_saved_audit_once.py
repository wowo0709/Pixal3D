import ast
import hashlib
import json
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting.run_saved_audit_once import audit_prefix, run


def test_prefix_preserves_every_audit_function_and_only_removes_terminal_loop():
    source = 'import json\nN="old"\ndef epoch(job,root,cfg,n):\n    assert n==1\n    return True\nwhile True:\n    raise RuntimeError("watcher must not run")\n'
    prefix, digest = audit_prefix(source, 'epoch')
    original = ast.parse(source)
    assert ast.dump(prefix.body[-1]) == ast.dump(original.body[-2])
    assert digest == hashlib.sha256(ast.get_source_segment(source, original.body[-2]).encode()).hexdigest()
    scope = {}; exec(compile(prefix, 'synthetic-owned-template.py', 'exec'), scope)
    assert scope['epoch']({}, None, {}, 1)
    with pytest.raises(AssertionError): scope['epoch']({}, None, {}, 2)


@pytest.mark.parametrize('source', ['def epoch(): pass', 'while True: pass\ndef epoch(): pass\nwhile True: pass', 'def wrong(): pass\nwhile True: pass'])
def test_unexpected_template_structure_is_rejected(source):
    with pytest.raises(ValueError): audit_prefix(source, 'epoch')


def test_cpu_only_and_source_hash_guards_run_before_any_template_execution(tmp_path, monkeypatch):
    template = tmp_path / 'owned.py'; template.write_text('raise RuntimeError("must not execute")')
    args = SimpleNamespace(implementation=template, expected_sha='wrong')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='CPU-only'): run(args)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='immutable'): run(args)


@pytest.mark.parametrize('include_binding', [False, True])
def test_completion_requires_closed_driver_binding_not_just_a_pass_label(tmp_path, monkeypatch, include_binding):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    root = tmp_path / 'synthetic-study'; config = root / 'configs/synthetic.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'experiment': {'arm': 'synthetic'},
        'default_output_dir': str(root / 'runs_ss64/synthetic')}))
    work = tmp_path / 'work'; work.mkdir()
    job = {'name': 'train-synthetic', 'command': ['python', '-m', 'synthetic', '--mode', 'train',
        '--root', str(root), '--config', str(config)]}
    (work / 'queue-jobs.json').write_text(json.dumps([job]))
    # Synthetic receipt only under pytest tmp; this does not model or call
    # the real checkpoint/W&B/file4 audit implementation.
    source = f'''from pathlib import Path
import json,hashlib
D=Path({str(tmp_path)!r})
def epoch(job,root,cfg,n):
    folder=N/'epoch-audits-v1'/job['name']/f'epoch{{n:02d}}'
    binding=folder/'once-driver-binding-v1.json'
    rows=[{{'relative_path':str(binding.relative_to(D)),'sha256':hashlib.sha256(binding.read_bytes()).hexdigest()}}] if {include_binding!r} else []
    (folder/'file4-closure.json').write_text(json.dumps({{'status':'PASS','local_equals_actual_file4_SHA256':True,'files':rows}}))
    return True
while True:
    raise RuntimeError('synthetic watcher must never execute')
'''
    template = tmp_path / 'synthetic-template.py'; template.write_text(source)
    args = SimpleNamespace(kind='epoch', implementation=template,
        expected_sha=hashlib.sha256(template.read_bytes()).hexdigest(), root=root,
        work_root=work, config=config, job_name=job['name'], epoch=1)
    output = root / 'epoch-audits-v1/train-synthetic/epoch01/once-driver-result-v1.json'
    if include_binding:
        run(args)
        result = json.loads(output.read_text())
        assert result['status'] == 'PASS_CLOSED_SAVED_CHECKPOINT_AUDIT_ONCE'
        assert result['watcher_loop_not_executed'] and result['training_restarts'] == 0
    else:
        with pytest.raises(ValueError, match='lifecycle/source binding'): run(args)
        assert not output.exists()
