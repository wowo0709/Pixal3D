"""Run the unchanged native renderer on one explicitly assigned GPU."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from .assigned_gpu_entry import adapt_guard,validate_assignment
from .cascade_view_roles import ClosedInputs,file_sha256 as sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--assignment',type=Path,required=True)
    p.add_argument('--assignment-sha256',required=True)
    p.add_argument('--runtime-proof',type=Path,required=True)
    a=p.parse_args();assignment=json.loads(a.assignment.read_text())
    if sha(a.assignment)!=a.assignment_sha256:raise ValueError('bound original assignment required')
    devices=validate_assignment(assignment,os.environ.get('CUDA_VISIBLE_DEVICES',''),
                                os.environ.get('MATCHING_V2_ALLOWED_GPUS',''))
    check=ClosedInputs(assignment['canonical_local_root'])
    check.check(a.assignment,a.assignment.parent/'file4-closure.json')
    check.check(a.plan,a.plan.parent/'file4-closure.json')
    module=importlib.import_module('experiments.deformnext_retargeting.render_cascade_artifact_v2')
    path=Path(module.__file__)
    if assignment['source_SHA256'].get(str(path))!=sha(path):raise ValueError('pinned original native renderer required')
    tree,proof=adapt_guard(path.read_text(),devices)
    namespace=dict(module.__dict__);exec(compile(tree,str(path),'exec'),namespace)
    argv=sys.argv;sys.argv=[str(path),'--plan',str(a.plan)];success=False
    try:
        namespace['main']();success=True
    finally:
        sys.argv=argv
        a.runtime_proof.write_text(json.dumps(dict(actual_entry_success=success,
            original_source_SHA256=sha(path),physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
            assigned_GPUs=devices,original_source_files_unmodified=True,
            numerical_entry_AST_device_literal_only=proof),indent=2)+'\n')


if __name__=='__main__':main()
