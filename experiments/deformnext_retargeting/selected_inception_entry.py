"""Run original Inception extraction with only its GPU membership adapted."""
import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

from .assigned_gpu_entry import validate_assignment
from .cascade_view_roles import file_sha256 as sha
from .selected_pipeline_model_contract import require


def adapt_guard(source,devices):
    tree=ast.parse(source);targets=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.Compare) and len(node.ops)==1 and isinstance(node.ops[0],ast.NotIn)
            and isinstance(node.left,ast.Subscript) and isinstance(node.left.value,ast.Name) and node.left.value.id=='visible'
            and isinstance(node.left.slice,ast.Constant) and node.left.slice.value==0 and len(node.comparators)==1
            and isinstance(node.comparators[0],ast.Set) and len(node.comparators[0].elts)==4
            and all(isinstance(v,ast.Constant) and isinstance(v.value,str) for v in node.comparators[0].elts)
            and {v.value for v in node.comparators[0].elts}=={'0','1','2','3'}):targets.append(node)
    require(len(targets)==1,'exactly one original visible[0] GPU membership literal required')
    require(devices and len(devices)==len(set(devices)) and all(isinstance(g,str) and g.isdigit() for g in devices),'explicit assigned devices required')
    node=targets[0];original=node.comparators[0];before=ast.dump(tree,include_attributes=False)
    node.comparators[0]=ast.Set(elts=[ast.Constant(g) for g in devices]);after=ast.dump(tree,include_attributes=False)
    node.comparators[0]=original;require(ast.dump(tree,include_attributes=False)==before,'all other numerical AST statements must remain unchanged')
    node.comparators[0]=ast.Set(elts=[ast.Constant(g) for g in devices]);ast.fix_missing_locations(tree)
    return tree,dict(changed_membership_literals=1,all_other_AST_statements_unchanged=True,
        original_AST_SHA256=hashlib.sha256(before.encode()).hexdigest(),assigned_AST_SHA256=hashlib.sha256(after.encode()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assignment',type=Path,required=True);p.add_argument('--assignment-sha256',required=True);p.add_argument('--runtime-proof',type=Path,required=True)
    a=p.parse_args();require(sha(a.assignment)==a.assignment_sha256,'actual source-bound assignment required')
    context=json.loads(a.assignment.read_text());devices=validate_assignment(context,os.environ.get('CUDA_VISIBLE_DEVICES',''),os.environ.get('MATCHING_V2_ALLOWED_GPUS',''))
    module=importlib.import_module('experiments.deformnext_retargeting.extract_official_inception');source=Path(module.__file__)
    require(context['source_SHA256'].get(str(source))==sha(source)
        and context['source_SHA256'].get(str(Path(__file__).resolve()))==sha(__file__),'pin both original numerical entry and literal adapter required')
    tree,proof=adapt_guard(source.read_text(),devices);namespace=dict(module.__dict__);exec(compile(tree,str(source),'exec'),namespace)
    previous=sys.argv;success=False
    try:
        sys.argv=[str(source),'--plan',str(a.plan),'--output',str(a.output)];namespace['main']();success=True
    finally:
        sys.argv=previous;proof.update(original_source=str(source),original_source_SHA256=sha(source),
            actual_entry_success=success,physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),assigned_GPUs=devices,original_source_files_unmodified=True)
        a.runtime_proof.write_text(json.dumps(proof,indent=2)+'\n')


if __name__=='__main__':main()
