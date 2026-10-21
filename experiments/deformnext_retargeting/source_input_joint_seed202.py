"""Second joint-training seed with the same fixed TRAIN1536 teacher.

The native warm probe and scientific trainer already read the config seed.
Only the sealed disposable capacity probe's single seed guard is adapted in
memory. This is not a second teacher-training seed or an admission decision.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

from .native_source_seed_runtime import seed202_ast


@contextmanager
def capacity_seed202_binding(module):
    path=Path(module.__file__);raw=path.read_bytes()
    tree,counts=seed202_ast(raw.decode(),native_probe=False)
    namespace=dict(module.__dict__)
    exec(compile(tree,str(path)+'[joint-seed202]','exec'),namespace)
    original=module.main
    record=dict(module=module.__name__,original_source_SHA256=hashlib.sha256(raw).hexdigest(),
        actual_seed_substitutions=counts,reverse_AST_equals_original=True,
        substituted_AST_SHA256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest())
    try:
        module.main=namespace['main']
        yield record
    finally:
        module.main=original
        if path.read_bytes()!=raw:
            raise ValueError('sealed capacity probe source changed during seed202 execution')


def validate_config(config,variant):
    exp=config['experiment']
    if (type(config.get('training_seed'))is not int or config['training_seed']!=202
            or variant not in {'original-NoDQ','source-contrast-NoDQ'}
            or exp.get('source_input_design')!=variant
            or exp.get('additional_source_teacher_supervision')is not True
            or config.get('load_dir')!='' or config.get('load_ckpt')is not None):
        raise ValueError('fresh teacher-warmed declared source-input design and actual joint training_seed202 required')


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--phase',choices=['warm-parity','capacity','science'],required=True)
    parser.add_argument('--source-input-design',choices=['original-NoDQ','source-contrast-NoDQ'],required=True)
    parser.add_argument('--source-teacher-init-spec',type=Path,required=True)
    local,remaining=parser.parse_known_args()
    # The native entry retains its full original parser and numerical recipe.
    if remaining.count('--config')!=1:
        raise ValueError('one explicit seed202 config required')
    config_path=Path(remaining[remaining.index('--config')+1])
    config_bytes=config_path.read_bytes();teacher_bytes=local.source_teacher_init_spec.read_bytes()
    config=json.loads(config_bytes);validate_config(config,local.source_input_design)
    teacher=json.loads(teacher_bytes)
    if teacher['variant']!=local.source_input_design:
        raise ValueError('same declared source-input teacher variant required')
    argv=sys.argv;bindings=[]
    try:
        if local.phase=='warm-parity':
            from . import probe_source_teacher_native_initialization as entry
            sys.argv=[sys.argv[0],'--initialization-spec',str(local.source_teacher_init_spec),*remaining]
            entry.main()
        else:
            from . import source_input_joint_runtime as entry
            sys.argv=[sys.argv[0],'--source-input-design',local.source_input_design,
                '--source-teacher-init-spec',str(local.source_teacher_init_spec),'--phase',local.phase,*remaining]
            if local.phase=='capacity':
                from . import probe_member_feature_fullbatch as capacity
                with capacity_seed202_binding(capacity) as binding:
                    entry.main();bindings.append(binding)
            else:entry.main()
    finally:sys.argv=argv
    if config_path.read_bytes()!=config_bytes or local.source_teacher_init_spec.read_bytes()!=teacher_bytes:
        raise ValueError('joint seed202 config or teacher specification changed during execution')
    if local.phase=='warm-parity':
        if remaining.count('--output')!=1:raise ValueError('one warm parity output required')
        output=Path(remaining[remaining.index('--output')+1])
    else:output=Path(config['output_dir'])
    if local.phase=='science':
        startup=json.loads((output/'startup.json').read_text())
        if startup.get('seed')!=202:
            raise ValueError('actual scientific startup must record joint training seed202')
    report=dict(status='PASS_COMPLETED_SOURCE_INPUT_JOINT_SEED202_PHASE',phase=local.phase,
        source_input_design=local.source_input_design,actual_config_training_seed=202,
        config_SHA256=hashlib.sha256(config_bytes).hexdigest(),
        fixed_teacher_checkpoint_SHA256=teacher['checkpoint_SHA256'],
        fixed_teacher_audit_SHA256=teacher['audit_SHA256'],
        fixed_teacher_training_seed_reused=True,independent_teacher_training_seed=False,
        second_joint_training_seed_not_generation_noise=True,probe_seed_bindings=bindings,
        original_entry_files_not_modified=True,native_optimizer_sampler_AMP_LoRA_data_and_epoch_validation_unchanged=True,
        GPU_capacity_and_quality_require_their_own_actual_reports=True,
        scientific_quality_improvement_not_proven=True,model_adopted=False,full_goal_complete=False)
    with (output/'source-input-joint-seed202-runtime.json').open('x')as stream:
        json.dump(report,stream,indent=2);stream.write('\n')


if __name__=='__main__':main()
