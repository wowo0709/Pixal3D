"""Dataset-only populations for the two original four-RGB SS controls.

Original checkpoint metadata stays step0 (zero deformation updates), with
multi-view pretraining step4210 stated separately. No previous test prediction
or fabricated prefix replay is required for a new held-out population.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .evaluate_bound_population import argument, bound_native_entry, project_dataset, validate_population
from .native_population_contract import ENGINEERING, FINAL, validate_capture
from .native_final_freeze import validate_spec


MODES = {'zero-source4': 'source-only', 'zero-edited4': 'edited-rgb'}
SCHEMA = 'original-MV-SS4210-population-binding-v1'


@contextmanager
def original_weight_guard(quick, checkpoint, records):
    """Check loaded values and evaluation immutability without CPU forwards."""
    original = quick.evaluate_quick
    def evaluate(trainer, *args, **kwargs):
        import torch
        if trainer.optimizer.state or trainer.step != 0:
            raise ValueError('original zero-shot evaluation requires an empty optimizer and zero updates')
        model = trainer.models['denoiser']
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        state = model.state_dict()
        if (set(state) != set(loaded) or any(state[k].dtype != loaded[k].dtype
                or state[k].shape != loaded[k].shape
                or not torch.equal(state[k].detach().cpu(), loaded[k]) for k in state)):
            raise ValueError('actual original full base and LoRA values differ from SS4210 checkpoint')
        names = sorted(k for k in state if 'lora_' in k)
        if not names:
            raise ValueError('original fine-tuned LoRA tensors required')
        def digest():
            value=hashlib.sha256()
            for name in names:
                tensor=model.state_dict()[name].detach().cpu().contiguous()
                value.update((name+str(tensor.dtype)+repr(tuple(tensor.shape))).encode())
                value.update(tensor.view(torch.uint8).numpy().tobytes())
            return value.hexdigest()
        before=digest();result=original(trainer,*args,**kwargs)
        if trainer.optimizer.state or trainer.step != 0 or digest()!=before:
            raise ValueError('original zero-shot optimizer or loaded LoRA changed during evaluation')
        records.append(dict(actual_loaded_trainable_SHA256=before,
            original_full_base_and_LoRA_exact_before_evaluation=True,
            actual_original_LoRA_tensors=len(names),optimizer_updates=0))
        return result
    quick.evaluate_quick=evaluate
    try:yield
    finally:quick.evaluate_quick=original


def model_binding(alias, cfg_path, checkpoint, hashes):
    if alias not in MODES or not checkpoint.endswith('/denoiser_step0004210.pt'):
        raise ValueError('one original SS4210 four-RGB control required')
    def spec(path):
        value = dict(path=path, SHA256=hashes[path]); validate_spec(value)
        return value
    return dict(schema=SCHEMA, alias=alias, conditioning_mode=MODES[alias],
        original_multiview_step=4210, deformation_finetuning_updates=0,
        native_config=spec(cfg_path), checkpoint=spec(checkpoint), conditioner_checkpoint=None)


def validate_arguments(arguments, plan, cfg, binding):
    expected = {'--stage':'ss64', '--step':'0', '--generation-batch':'8',
        '--checkpoint':binding['checkpoint']['path'], '--arm-config':binding['native_config']['path'],
        '--conditioning-mode':binding['conditioning_mode'], '--panel-size':str(plan['expected_assets']),
        '--panel-file':plan['panel'], '--seed-map-file':plan['seeds'], '--output':plan['output']}
    for flag, value in expected.items():
        if argument(arguments, flag) != value:
            raise ValueError('unchanged original zero-shot argument required: '+flag)
    allowed = set(expected) | {'--expected-stage-root','--expected-entry-sha','--cuda-memory-fraction','--config-root'}
    if (len(arguments) != 2*len(allowed) or set(arguments[::2]) != allowed
            or cfg['trainer']['args'].get('deform_conditioner') is not None
            or cfg['trainer']['args']['continuation_ckpt']['denoiser'] != binding['checkpoint']['path']
            or cfg['dataset']['name'] != 'DeformNeXtSS64'
            or cfg['validation']['dataset_args'].get('transport_arm') != 'v1'
            or plan.get('model_binding') != binding):
        raise ValueError('explicit vanilla original SS config and read-only arguments required')
    fraction = float(argument(arguments, '--cuda-memory-fraction'))
    if not 0 < fraction <= 1:
        raise ValueError('explicit valid CUDA memory fraction required')


def validate_binding(plan, cfg, binding, rows, panel, seeds, *, train, validation,
                     controls, parent=None, parity=None, qa=None, qa_plan=None):
    if (binding.get('schema') != SCHEMA or binding.get('alias') not in MODES
            or binding.get('conditioning_mode') != MODES[binding['alias']]
            or binding.get('original_multiview_step') != 4210
            or binding.get('deformation_finetuning_updates') != 0
            or binding.get('conditioner_checkpoint') is not None
            or not binding['checkpoint']['path'].endswith('/denoiser_step0004210.pt')):
        raise ValueError('original zero-update SS4210 binding required')
    for key in ['native_config', 'checkpoint']: validate_spec(binding[key])
    validate_arguments(plan['native_arguments'], plan, cfg, binding)
    expected = 36 if plan['population_scope'] == ENGINEERING else 508 if plan['population_scope'] == FINAL else 0
    if (not expected or len(panel) != expected or len(set(panel)) != expected
            or plan['expected_assets'] != expected or set(seeds) != set(panel)
            or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds.values())
            or controls.get('status') != 'FROZEN_ORIGINAL_ZERO_RGB_CONTROLS_BEFORE_TEST'
            or controls.get('heldout_model_quality_read') is not False
            or controls.get('models', {}).get(binding['alias']) != binding
            or set(controls['models']) != set(MODES)
            or plan['panel'] != plan['population_inputs']['panel']['path']
            or plan['seeds'] != plan['population_inputs']['seeds']['path']):
        raise ValueError('complete explicit original control population and pre-test binding required')
    for alias, other in controls['models'].items():
        if (other.get('schema') != SCHEMA or other.get('conditioning_mode') != MODES[alias]
                or other.get('checkpoint') != binding['checkpoint']
                or other.get('native_config') != binding['native_config']
                or other.get('conditioner_checkpoint') is not None
                or other.get('deformation_finetuning_updates') != 0):
            raise ValueError('the two original controls must differ only in four-RGB input mode')
    if plan['population_scope'] == ENGINEERING:
        if (plan['population_inputs']['manifest']['path'] != cfg['validation']['roots_file']
                or rows != validation or len(rows) != 508
                or not set(panel).issubset(r['sample_id'] for r in rows)):
            raise ValueError('original complete validation data required for engineering')
        rebound = project_dataset(cfg, manifest=cfg['validation']['roots_file'], expected_split='validation',
                                  **plan['dataset_roots'])
        if rebound != cfg:
            raise ValueError('engineering must preserve the full original config')
        return rebound
    validate_population(rows, scope=FINAL, panel=panel, seeds=seeds, train_rows=train, validation_rows=validation)
    if (parent is None or parent.get('status') != 'FROZEN_FROM_DEVELOPMENT_ONLY'
            or parent.get('heldout_quality_read_before_freeze') is not False
            or parent['population']['sample_ids'] != panel
            or parent['population']['generation_seeds'] != seeds
            or parent['population']['source_clusters_by_sample'] != {r['sample_id']:r['source_cluster_id'] for r in rows}
            or parent['population']['rendered_cameras'] != [8,9,10]
            or parent.get('original_zero_controls') != ['source4-M0-M0-M0-M0','edited4-M1-M0-M0-M0']):
        raise ValueError('the native campaign and original controls must share a frozen complete test population')
    if (parity is None or parity.get('status') != 'PASS_ACTUAL_NATIVE_BOUND_VALIDATION36_REPEATABILITY'
            or parity.get('model_binding') != binding or parity.get('independent_model_loads') != 2
            or parity.get('reference_prefix32_records_exact') is not True
            or parity.get('all36_records_and_archived_occupancy_exact') is not True
            or parity.get('tail_generation_batch') != 4 or parity.get('optimizer_updates') != 0):
        raise ValueError('actual own original-zero36 two-load parity required before test')
    manifest = plan['population_inputs']['manifest']
    if (qa is None or qa_plan is None or qa.get('status') != 'PASS_ACTUAL_TEST508_NATIVE_CACHE_AND_LOADER_QA'
            or qa.get('published_manifest') != manifest['path']
            or qa.get('published_manifest_SHA256') != manifest['SHA256']
            or not qa.get('SS_PBR_cache_ready') or not qa.get('all508_paired_coverage')
            or not qa.get('no_test_asset_dropped') or not qa.get('no_candidate_model_forward_or_test_quality_metric')
            or qa.get('Omega_completed_updates') != 23597
            or qa.get('matching_head_training_or_calibration_updates') != 0
            or plan['dataset_roots'] != {k:qa_plan[k] for k in ['match_root','projection_root']}):
        raise ValueError('actual complete native test cache QA and dataset binding required')
    return project_dataset(cfg, manifest=manifest['path'], expected_split='test', **plan['dataset_roots'])


def execute(plan_path):
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text())
    check=ClosedInputs(plan['canonical_local_root'])
    def guard():
        check.check(plan_path,plan_path.parent/'file4-closure.json')
        for p,d in {**plan['source_SHA256'],**plan['input_SHA256']}.items():
            if sha(p)!=d:raise ValueError('original control source/input changed: '+p)
        for p,r in plan['closed_inputs'].items():check.check(p,r)
    def read(spec):
        validate_spec(spec)
        if sha(spec['path'])!=spec['SHA256']:raise ValueError('bound control input changed')
        return json.loads(Path(spec['path']).read_text())
    guard(); binding=plan['model_binding'];cfg=read(binding['native_config'])
    if sha(binding['checkpoint']['path'])!=binding['checkpoint']['SHA256']:
        raise ValueError('original complete SS4210 checkpoint changed')
    controls=read(plan['control_freeze']);inputs=plan['population_inputs']
    for spec in inputs.values():
        if sha(spec['path'])!=spec['SHA256']:raise ValueError('population bytes changed')
    rows=[json.loads(l)for l in Path(inputs['manifest']['path']).read_text().splitlines()]
    train=[json.loads(l)for l in Path(inputs['train_manifest']['path']).read_text().splitlines()]
    validation=[json.loads(l)for l in Path(inputs['validation_manifest']['path']).read_text().splitlines()]
    panel=read(inputs['panel'])['sample_ids'];seeds=read(inputs['seeds'])
    kwargs=dict(train=train,validation=validation,controls=controls)
    extra=[]
    if plan['population_scope']==FINAL:
        for key,target in [('parent_freeze','parent'),('engineering_parity','parity'),('cache_QA','qa'),('cache_QA_plan','qa_plan')]:
            kwargs[target]=read(plan[key]);extra.append(plan[key])
        for evidence in kwargs['parent']['development_evidence']:read(evidence);extra.append(evidence)
        reference=None
    else:
        reference=read(plan['reference_evaluation']);extra.append(plan['reference_evaluation'])
        if panel!=reference['quick_panel_ids'][:36]:raise ValueError('original reference128 prefix36 required')
    rebound=validate_binding(plan,cfg,binding,rows,panel,seeds,**kwargs)
    from . import quick_metrics as quick
    from . import evaluate_deterministic_checkpoint as native
    from .evaluate_ss_artifact_subset import capture_ss_artifacts,validate_subset_reference
    from .member_raw_metric_dispatch import raw_ss_metric_dispatch
    output=Path(plan['output']);captured=[];events=[];calls=[];weights=[];argv=sys.argv
    entry=Path(__file__).with_name('evaluate_native_checkpoint.py')
    def projector(actual):
        if actual!=cfg:raise ValueError('native loader read another original config')
        calls.append(1);return copy.deepcopy(rebound)
    try:
        with raw_ss_metric_dispatch(events),bound_native_entry(entry,projector),capture_ss_artifacts(quick,output,captured), \
                original_weight_guard(quick,binding['checkpoint']['path'],weights):
            sys.argv=[native.__file__,*plan['native_arguments']];native.main()
    finally:sys.argv=argv
    result=json.loads((output/'evaluation.json').read_text())
    if (calls!=[1] or len(weights)!=1 or result['step']!=0 or result['checkpoint']!=binding['checkpoint']['path']
            or result['conditioner_checkpoint'] is not None or result['diagnostic_only']
            or result['conditioning_mode']!=binding['conditioning_mode']
            or result['input_states']!=(['M0']*4 if binding['alias']=='zero-source4' else ['M1','M0','M0','M0'])
            or result['input_camera_ids']!=[8,9,10,11] or result['manifest_sha256']!=inputs['manifest']['SHA256']
            or result['quick_panel_ids']!=panel):
        raise ValueError('actual original four-RGB numerical execution differs')
    validate_capture(result,captured,len(panel))
    import numpy as np
    for artifact in captured:
        path=output/artifact['artifact']
        if sha(path)!=artifact['artifact_SHA256']:raise ValueError('actual SS archive changed')
        with np.load(path)as arrays:
            for key in ['prediction_latent','target_latent']:
                value=arrays[key];digest=hashlib.sha256((str(value.dtype)+repr(value.shape)).encode());digest.update(value.tobytes())
                artifact[key+'_SHA256']=digest.hexdigest()
    prefix=False
    if reference is not None:
        validate_subset_reference(dict(result,per_asset=result['per_asset'][:32],quick_panel_ids=panel[:32]),reference,captured[:32],count=32)
        prefix=True
    for spec in extra:read(spec)
    guard()
    result.update(population_scope=plan['population_scope'],frozen_native_model_binding=binding,
        actual_loaded_trainable_SHA256=weights[0]['actual_loaded_trainable_SHA256'],
        actual_original_full_base_and_LoRA_validation=weights[0],
        original_multiview_SS_step=4210,deformation_finetuning_updates=0,
        population_binding_applied_only_to_dataset_selection=True)
    result['native_SS_artifacts']=dict(status='PASS_ACTUAL_ORIGINAL_ZERO_SS_BOUND_POPULATION_ARCHIVE',
        archived_asset_count=len(panel),artifacts=captured,predicted_SS_is_available_for_full_cascade=True,
        GT_SS_is_for_scoring_only=True,full_cascade_generated=False,optimizer_updates=0)
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    audit=dict(status='PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION',population_scope=plan['population_scope'],
        assets=len(panel),model_binding=binding,actual_generation_batch=8,tail_generation_batch=len(panel)%8,
        sampling_steps=12,guidance=3.,optimizer_updates=0,actual_reference_prefix32_records_exact=prefix,
        actual_saved_SS_occupancy_counts_verified=True,original_zero_four_RGB=True,
        numerical_original_evaluator_unchanged=True,verified_inputs=check.inputs,full_goal_complete=False)
    (output/'population-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (output/'completed-bound.json').write_text(json.dumps(dict(status=audit['status'],assets=len(panel),
        population_scope=plan['population_scope'],optimizer_updates=0,full_goal_complete=False),indent=2)+'\n')
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(execute(args.plan)),flush=True)


if __name__=='__main__':main()
