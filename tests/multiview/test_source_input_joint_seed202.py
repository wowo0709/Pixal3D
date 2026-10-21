"""CPU seed guard/routing tests; mock phase outputs are never GPU evidence."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting import source_input_joint_seed202 as seed


def config(variant='original-NoDQ'):
    return dict(training_seed=202,load_dir='',load_ckpt=None,experiment=dict(
        source_input_design=variant,additional_source_teacher_supervision=True))


@pytest.mark.parametrize('value',[101,42,True,'202',202.0,None])
def test_requires_actual_integer_joint_seed202(value):
    cfg=config();cfg['training_seed']=value
    with pytest.raises(ValueError,match='training_seed202'):seed.validate_config(cfg,'original-NoDQ')


@pytest.mark.parametrize('damage',['variant','cold','resume'])
def test_rejects_variant_supervision_or_resume_change(damage):
    cfg=config()
    if damage=='variant':cfg['experiment']['source_input_design']='source-contrast-NoDQ'
    elif damage=='cold':cfg['experiment']['additional_source_teacher_supervision']=False
    else:cfg['load_dir']='/different/run'
    with pytest.raises(ValueError):seed.validate_config(cfg,'original-NoDQ')


def test_capacity_guard_changes_only_in_memory_and_restores_after_failure(tmp_path):
    path=tmp_path/'unit-capacity.py'
    path.write_text("def main():\n    if cfg.get('training_seed') != 101: raise ValueError('wrong seed')\n    return 101\n")
    namespace=dict(__name__='unit_capacity',cfg={'training_seed':202})
    exec(compile(path.read_text(),str(path),'exec'),namespace)
    module=SimpleNamespace(__file__=str(path),**namespace);original=module.main;before=path.read_bytes()
    with pytest.raises(RuntimeError,match='unit phase failure'):
        with seed.capacity_seed202_binding(module) as record:
            assert module.main()==101 # Unrelated numeric constants remain exact.
            assert record['actual_seed_substitutions']==dict(seed_guard=1,initial_RNG=0)
            assert record['reverse_AST_equals_original']
            raise RuntimeError('unit phase failure')
    assert module.main is original and path.read_bytes()==before


def test_real_sealed_capacity_has_one_reversible_seed_guard():
    path=Path(seed.__file__).with_name('probe_member_feature_fullbatch.py');before=path.read_bytes()
    tree,counts=seed.seed202_ast(before.decode(),native_probe=False)
    compile(tree,'<CPU-capacity-seed202-binding>','exec')
    assert counts==dict(seed_guard=1,initial_RNG=0) and path.read_bytes()==before


@pytest.mark.parametrize('phase',['warm-parity','capacity','science'])
@pytest.mark.parametrize('variant',['original-NoDQ','source-contrast-NoDQ'])
def test_phase_routing_records_joint_seed_without_claiming_independent_teacher(tmp_path,monkeypatch,phase,variant):
    from experiments.deformnext_retargeting import source_input_joint_runtime as joint
    from experiments.deformnext_retargeting import probe_source_teacher_native_initialization as warm
    from experiments.deformnext_retargeting import probe_member_feature_fullbatch as capacity
    cfg=config(variant);cfg['output_dir']=str(tmp_path/'unit-output')
    cfgpath=tmp_path/'unit-config.json';cfgpath.write_text(json.dumps(cfg))
    spec=dict(variant=variant,checkpoint_SHA256='unit-checkpoint-not-real',audit_SHA256='unit-audit-not-real')
    specpath=tmp_path/'unit-teacher.json';specpath.write_text(json.dumps(spec))
    calls=[];original_capacity=capacity.main;output=Path(cfg['output_dir'])
    def fake_native_entry():
        calls.append(sys.argv[1:])
        if phase=='capacity':assert capacity.main is not original_capacity
        output.mkdir()
        if phase=='science':(output/'startup.json').write_text(json.dumps(dict(seed=202,unit_only=True)))
    monkeypatch.setattr(warm if phase=='warm-parity'else joint,'main',fake_native_entry)
    args=['unit-seed202','--phase',phase,'--source-input-design',variant,
        '--source-teacher-init-spec',str(specpath),'--config',str(cfgpath)]
    if phase=='warm-parity':args+=['--sample-id','unit-sample','--expected-conditioning-sha256','unit-RGB','--output',str(output)]
    monkeypatch.setattr(sys,'argv',args);seed.main()
    assert sys.argv is args and capacity.main is original_capacity and len(calls)==1
    expected=('--initialization-spec'if phase=='warm-parity'else'--source-teacher-init-spec')
    assert calls[0][calls[0].index(expected)+1]==str(specpath)
    if phase!='warm-parity':assert calls[0][calls[0].index('--phase')+1]==phase
    report=json.loads((output/'source-input-joint-seed202-runtime.json').read_text())
    assert report['actual_config_training_seed']==202 and report['phase']==phase
    assert report['fixed_teacher_training_seed_reused']and not report['independent_teacher_training_seed']
    assert len(report['probe_seed_bindings'])==int(phase=='capacity')
    assert report['scientific_quality_improvement_not_proven']is True and not report['full_goal_complete']


def test_failed_phase_restores_args_and_capacity_without_creating_success_report(tmp_path,monkeypatch):
    from experiments.deformnext_retargeting import source_input_joint_runtime as joint
    from experiments.deformnext_retargeting import probe_member_feature_fullbatch as capacity
    cfg=config();cfg['output_dir']=str(tmp_path/'no-output')
    cfgpath=tmp_path/'config.json';cfgpath.write_text(json.dumps(cfg))
    specpath=tmp_path/'teacher.json';specpath.write_text(json.dumps(dict(variant='original-NoDQ')))
    original=capacity.main
    def fail():raise RuntimeError('native failure')
    monkeypatch.setattr(joint,'main',fail)
    args=['unit','--phase','capacity','--source-input-design','original-NoDQ',
        '--source-teacher-init-spec',str(specpath),'--config',str(cfgpath)]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(RuntimeError,match='native failure'):seed.main()
    assert sys.argv is args and capacity.main is original and not Path(cfg['output_dir']).exists()


@pytest.mark.parametrize('damage',['startup-seed101','changed-config'])
def test_native_completion_cannot_be_relabeled_as_unchanged_seed202(tmp_path,monkeypatch,damage):
    from experiments.deformnext_retargeting import source_input_joint_runtime as joint
    cfg=config();cfg['output_dir']=str(tmp_path/'unit-output')
    cfgpath=tmp_path/'config.json';cfgpath.write_text(json.dumps(cfg))
    specpath=tmp_path/'teacher.json';specpath.write_text(json.dumps(dict(variant='original-NoDQ')))
    output=Path(cfg['output_dir'])
    def wrong_native_result():
        output.mkdir()
        (output/'startup.json').write_text(json.dumps(dict(seed=101 if damage=='startup-seed101' else 202)))
        if damage=='changed-config':cfgpath.write_text('{}')
    monkeypatch.setattr(joint,'main',wrong_native_result)
    args=['unit','--phase','science','--source-input-design','original-NoDQ',
        '--source-teacher-init-spec',str(specpath),'--config',str(cfgpath)]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(ValueError):seed.main()
    assert sys.argv is args and not (output/'source-input-joint-seed202-runtime.json').exists()
