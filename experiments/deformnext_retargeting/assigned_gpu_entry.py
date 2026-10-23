"""Run pinned evaluation entries on an explicitly assigned physical GPU.

The historical entries hard-code GPU0-3. Adapt only that membership literal
in memory; preserve the environment allow-list and every numerical statement.
The on-disk, checksum-bound original source remains unchanged.
"""
import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys


def validate_assignment(assignment, visible, allowed):
    devices = assignment.get('assigned_GPUs')
    if (not isinstance(devices,list) or not devices or len(set(devices)) != len(devices)
            or any(not isinstance(x,str) or not x.isdigit() for x in devices)
            or visible not in devices or visible not in allowed.split(',')):
        raise ValueError('one physical GPU in both the explicit plan and scheduler assignment required')
    return devices


def adapt_guard(source, devices):
    """Change one exact legacy GPU membership literal, with a reversible proof."""
    tree = ast.parse(source)
    targets=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='gpu'
                and len(node.ops)==1 and isinstance(node.ops[0],ast.NotIn)
                and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Set)):
            values=node.comparators[0].elts
            if len(values)==4 and all(isinstance(x,ast.Constant)and isinstance(x.value,str)for x in values):
                if {x.value for x in values}=={'0','1','2','3'}:targets.append(node)
    if len(targets)!=1:
        raise ValueError('exactly one original GPU0-3 membership guard required')
    target=targets[0];original=target.comparators[0]
    before=ast.dump(tree,include_attributes=False)
    target.comparators[0]=ast.Set(elts=[ast.Constant(x)for x in devices])
    after=ast.dump(tree,include_attributes=False)
    target.comparators[0]=original
    if ast.dump(tree,include_attributes=False)!=before:
        raise ValueError('non-device source statements changed')
    target.comparators[0]=ast.Set(elts=[ast.Constant(x)for x in devices])
    ast.fix_missing_locations(tree)
    proof=dict(changed_membership_literals=1,all_other_AST_statements_unchanged=True,
        original_AST_SHA256=hashlib.sha256(before.encode()).hexdigest(),
        assigned_AST_SHA256=hashlib.sha256(after.encode()).hexdigest())
    return tree,proof


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entry',required=True,choices=['fixed-support','render-replay','shared-score','official-perceptual'])
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--assignment',type=Path,required=True)
    p.add_argument('--assignment-sha256',required=True)
    p.add_argument('--runtime-proof',type=Path,required=True)
    args=p.parse_args();raw=args.assignment.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.assignment_sha256:
        raise ValueError('unchanged closed assignment document required')
    assignment=json.loads(raw)
    devices=validate_assignment(assignment,os.environ.get('CUDA_VISIBLE_DEVICES',''),os.environ.get('MATCHING_V2_ALLOWED_GPUS',''))
    proofs={}

    def adapted(module_name):
        module=importlib.import_module(module_name);path=Path(module.__file__)
        source=path.read_text();digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if assignment['source_SHA256'].get(str(path))!=digest:
            raise ValueError('explicit original entry source checksum required: '+str(path))
        tree,proof=adapt_guard(source,devices);proof.update(original_source=str(path),original_source_SHA256=digest)
        namespace=dict(module.__dict__)
        exec(compile(tree,str(path),'exec'),namespace)
        proofs[module_name]=proof
        return namespace

    previous_argv=sys.argv;sys.argv=[args.entry,'--plan',str(args.plan)]
    success=False
    try:
        if args.entry=='fixed-support':
            from . import replay_r_pbr_on_generated_shape as native
            from . import replay_r_pbr_fixed_predicted_support as fixed
            fn=adapted(native.__name__)['run'];previous=native.run;native.run=fn
            try:fixed.main()
            finally:native.run=previous
        elif args.entry=='render-replay':
            from . import render_pbr_replay
            original_run_module=runpy.run_module
            target='experiments.deformnext_retargeting.render_cascade_artifact_v2'
            def run_assigned(name,*pos,**kw):
                if name!=target:return original_run_module(name,*pos,**kw)
                if pos or kw!={'run_name':'__main__'}:
                    raise ValueError('original native renderer call signature required')
                namespace=adapted(target);namespace['main']();return namespace
            runpy.run_module=run_assigned
            try:render_pbr_replay.main()
            finally:runpy.run_module=original_run_module
        else:
            names={'shared-score':'score_cascade_fixed_reference','official-perceptual':'evaluate_official_perceptual'}
            namespace=adapted('experiments.deformnext_retargeting.'+names[args.entry]);namespace['main']()
        success=True
    finally:
        sys.argv=previous_argv
        args.runtime_proof.parent.mkdir(parents=True,exist_ok=True)
        args.runtime_proof.write_text(json.dumps(dict(entry=args.entry,physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
            assigned_GPUs=devices,actual_entry_success=success,original_source_files_unmodified=True,
            numerical_AST_unchanged=True,adapted_modules=proofs),indent=2)+'\n')


if __name__=='__main__':main()
