"""Separate the new369-state scientific gates from the old367-state arm."""
from pathlib import Path
import hashlib,json


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
    return h.hexdigest()


def closed_json(path,receipt,campaign_root):
    path=Path(path);r=json.loads(Path(receipt).read_text())
    relative=str(path.relative_to(Path(campaign_root).parent))
    if (r['status']!='PASS' or not r['local_equals_actual_file4_SHA256']
            or not any(x['relative_path']==relative and x['sha256']==digest(path) for x in r['files'])):
        raise ValueError('actual file4 proof differs: '+str(path))
    return json.loads(path.read_text())


def validate_capacity(report):
    if (report.get('status')!='PASS_ACTUAL_TASK_RAY369_DDP2_CAPACITY'
            or (report.get('world_size'),report.get('microbatch'),report.get('global_batch'),report.get('accumulation'))!=(2,48,96,1)
            or report.get('initial_optimizer_states')!=366 or report.get('actual_optimizer_states')!=369
            or report.get('changed_trainable_tensors')!={'default':360,'deform_conditioner':6,'deformation_query':1,'task_ray_attention':2}
            or report.get('optimizer_steps')!={'default':[99],'deform_conditioner':[99],'deformation_query':[3],'task_ray_attention':[3]}
            or not report.get('all_frozen_parameters_and_buffers_exact')
            or not report.get('post_update_DDP_equal') or not report.get('first97_common_loss_fields_exact')):
        raise ValueError('own actual369-state full-batch DDP capacity required')


def validate_initial32(report):
    if (report.get('status')!='PASS_ACTUAL_NATIVE_TASK_RAY_INITIAL32_PARITY'
            or report.get('assets')!=32 or report.get('generation_batch')!=8
            or report.get('reference_step')!=96 or report.get('optimizer_updates')!=0
            or not report.get('all32_baseline_record_fields_and_metrics_exact')
            or report.get('conditioner_state_tensors')!=46):
        raise ValueError('own native generated32 initial identity required')
