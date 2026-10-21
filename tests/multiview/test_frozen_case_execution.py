"""Deferred archive bindings must use actual completed producer manifests."""
import json
from pathlib import Path

import pytest

from experiments.deformnext_retargeting.execute_frozen_cascade_case import io_binding, execute
from experiments.deformnext_retargeting.evaluate_bound_population import sha


def test_remote_PBR_and_render_bind_the_actual_produced_manifest_bytes(tmp_path):
    producer=tmp_path/'generation';producer.mkdir()
    path=producer/'remote-npz.json'
    path.write_text(json.dumps(dict(schema='native-cascade-remote-npz-v1',remote_root='/file4/owned/raw')))
    phase=dict(remote_stage='PBR-replay',input_archive_roots=[str(producer)],
        native_plan=dict(path='/plans/PBR.json',SHA256='plan'),native_source={'SHA256':'source'})
    store=dict(remote_root='/file4/owned/raw',ssh_argv=['ssh','explicit-assigned-host'])
    actual,module=io_binding(phase,{},store=store,prefix='case/PBR',compressed=False)
    assert module.endswith('.pbr_remote_results')
    assert actual['input_manifests']=={str(producer):sha(path)}
    assert not actual['compressed']
    path.write_text(json.dumps(dict(schema='native-cascade-remote-npz-v1',remote_root='/file4/other')))
    with pytest.raises(ValueError,match='store/schema'):io_binding(phase,{},store=store,prefix='case/render',compressed=False)


def test_missing_producer_and_unsafe_prefix_cannot_launch_a_native_phase(tmp_path):
    phase=dict(remote_stage='render',input_archive_roots=[str(tmp_path/'absent')],native_plan={'path':'/native','SHA256':'p'},native_source={'SHA256':'s'})
    with pytest.raises(FileNotFoundError):io_binding(phase,{},store={'remote_root':'/file4/raw'},prefix='case/render',compressed=False)
    phase['input_archive_roots']=[]
    with pytest.raises(ValueError,match='relative object key'):
        io_binding(phase,{},store={'remote_root':'/file4/raw'},prefix='../escape',compressed=False)


def test_full508_zero_uses_the_explicit_metadata_guard_adapter(tmp_path):
    phase=dict(remote_stage='zero-generation',input_archive_roots=[],native_plan={'path':'/native','SHA256':'p'},native_source={'SHA256':'s'})
    actual,module=io_binding(phase,{'bound_zero_shot':{}},store={'remote_root':'/file4/raw'},prefix='zero/generation',compressed=True)
    assert module.endswith('.bound_zero_shot_cascade')
    path=Path(__file__).parents[2]/'experiments/deformnext_retargeting/bound_zero_shot_cascade.py'
    assert actual['bound_zero_adapter_source_SHA256']==sha(path)


def test_unassigned_GPU_rejected_before_any_plan_or_weight_read(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7');monkeypatch.setenv('MATCHING_V2_ALLOWED_GPUS','0,1,2,3')
    with pytest.raises(ValueError,match='explicitly assigned'):
        execute({'path':'/must-not-read'},{'allowed_physical_GPUs':['0','1','2','3']})


@pytest.mark.parametrize('fail_render',[False,True])
def test_native_phase_sequence_and_failure_preservation_in_CPU_control_harness(tmp_path,monkeypatch,fail_render):
    # This harness fabricates no canonical receipt and runs no native model.
    # Only subprocess scheduling/output checks are exercised on unit metadata.
    from types import SimpleNamespace
    from test_frozen_cascade_case_plans import fixture
    from experiments.deformnext_retargeting import prepare_frozen_cascade_cases as planner
    from experiments.deformnext_retargeting import execute_frozen_cascade_case as executor
    plan=fixture(tmp_path)
    prepared=planner.prepare(plan,tmp_path/'compiled')
    case_spec=prepared['cases']['v1'][0]['plan']
    case=json.loads(Path(case_spec['path']).read_text())
    first=json.loads(Path(case['phases'][0]['native_plan']['path']).read_text())
    root=Path(first['output']).parent
    commands=[];closures=[]
    class ControlChild:
        pid=123456
        def __init__(self,command,**kwargs):
            commands.append(command)
            io=json.loads(Path(command[-1]).read_text())
            native=json.loads(Path(io['native_plan']).read_text())
            output=Path(native['output']);output.mkdir()
            self.code=int(fail_render and io['stage']=='render')
            result={'status':'PASS_UNIT_CONTROL_HARNESS_NOT_A_NATIVE_RESULT'}
            if io['stage']=='generation':
                result.update(actual_resolution=1024,sample_id=case['sample_id'],GT_geometry_or_coords_used_in_generation=False,per_output_alignment=False)
                filename='generation.json'
            else:
                filename='render-evaluation.json'
                if io['stage']=='render':
                    result.update(per_view=[{'GT_geometry_RGBA_containment':1.}]*3,prediction_geometry_postprocess={'render_faces':1000000})
            (output/filename).write_text(json.dumps(result))
            (output/'remote-npz.json').write_text(json.dumps(dict(schema='native-cascade-remote-npz-v1',remote_root='/file4/unit-only')))
        def wait(self):return self.code
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setenv('MATCHING_V2_ALLOWED_GPUS','0,1,2,3')
    monkeypatch.setattr(executor.subprocess,'Popen',ControlChild)
    monkeypatch.setattr(executor.psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:123.))
    monkeypatch.setattr(executor,'close',lambda folder,**kwargs:closures.append(folder))
    source=Path(executor.__file__)
    execution=dict(allowed_physical_GPUs=['0','1','2','3'],compressed=False,source_root=str(source.parents[2]),
        executor_source=dict(path=str(source),SHA256=sha(source)),canonical_local_root=str(tmp_path),
        python='unit-control-harness',raw_prefix='unit',store=dict(remote_root='/file4/unit-only',ssh_argv=[]))
    if fail_render:
        with pytest.raises(RuntimeError,match='outputs preserved'):executor.execute(case_spec,execution)
        assert (root/'failure.json').exists() and not (root/'case-completed.json').exists()
        assert len(commands)==2 and len(closures)==1
    else:
        result=executor.execute(case_spec,execution)
        assert result['sample_id']==case['sample_id'] and len(commands)==3 and len(closures)==4
        assert all(c[c.index('-m')+1].endswith('.cascade_remote_results') for c in commands)
        assert (root/'case-completed.json').exists()
