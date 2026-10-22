"""Protect raw prediction consumers before retiring closed local duplicates."""
import json
from pathlib import Path


RAW_NAMES={'prediction.npz','geometry-evaluation.npz'}


def pending_raw_reads(jobs,states,root):
    root=Path(root);files={str(root/name)for name in RAW_NAMES};readers=set()
    for job in jobs:
        if states.get(job['name'],{}).get('status','pending')not in{'pending','running'}:
            continue
        command=job.get('command',[])
        if '--plan'not in command:continue
        path=Path(command[command.index('--plan')+1])
        if not path.is_file():raise ValueError('unreadable pending plan prevents raw-cache retirement: '+job['name'])
        document=json.loads(path.read_text())
        metadata_summary=document.get('phase')=='summary'or document.get('worker_mode')=='summary'
        def scan(value,key=None):
            if isinstance(value,str):
                metadata_only=metadata_summary and key=='generation'
                if value in files or value==str(root)and not metadata_only:
                    readers.add(job['name'])
            elif isinstance(value,list):
                for item in value:scan(item,key)
            elif isinstance(value,dict):
                for k,item in value.items():
                    if k in{'input_SHA256','closed_inputs','future_closed_inputs'}and isinstance(item,dict):
                        for raw in item:scan(raw)
                    elif k not in{'verified_inputs','files','source_SHA256'}:scan(item,k)
        scan(document)
    return sorted(readers)


def validate_raw_candidate(path,*,root,owned_outputs_root,terminal,generation,render_audit,
                           canonical_entry,actual_local_SHA256,readers):
    path,root,owned=Path(path),Path(root),Path(owned_outputs_root)
    if (path.parent!=root or path.name not in RAW_NAMES or not root.is_absolute()or not owned.is_absolute()
            or root==owned or not root.is_relative_to(owned)or root.resolve()!=root or path.is_symlink()
            or path.resolve().parent!=root or not path.is_file()or terminal.get('returncode')!=0
            or generation.get('status')!='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
            or generation.get('checkpoint_kind')!='trained-native1900'or generation.get('engineering_only')is not False
            or render_audit.get('status')!='PASS_ACTUAL_NATIVE1900_PBR_CASE_RENDERED_AND_SHARED_SCORED'
            or generation.get('sample_id')!=render_audit.get('sample_id')
            or generation.get('PBR_arm')!=render_audit.get('mode')
            or not generation.get('source_weights_unchanged')or not all(generation['geometry_array_byte_identity'].values())
            or not render_audit.get('same_raw_geometry_and_native_LOD')or not render_audit.get('fixed_original_material_M1_GT')
            or readers):
        raise ValueError('only own completed generated raw duplicates after actual rendering and all readers may retire')
    if canonical_entry.get('sha256')!=actual_local_SHA256 or canonical_entry.get('bytes')!=path.stat().st_size:
        raise ValueError('actual raw bytes must equal canonical file4 archive')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=actual_local_SHA256,local_duplicate_only=True)
