"""Run the chosen physical SS family unchanged for an independent TRAIN seed.

Existing numerical runtime functions are reused. The additional trainer
observer checks actual paired startup/inputs and frozen tensors; it introduces
no neural layer, loss, optimizer update, data transformation or sampling change.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
import math
import os
from pathlib import Path

from .cascade_view_roles import ClosedInputs
from .collect_module_seed_confirmation import read_file,require
from .pipeline_confirmation_recipe_v2 import validate_arm,validate_confirmation
from .pipeline_population_contract_v2 import ALIASES,SOURCE_COUNTS,family


RUNTIMES=dict(r='r_mediated_native_runtime',r_top1='r_top1_native_runtime',
    projected='projected_native_runtime',raw_correspondence='raw_correspondence_native_runtime',
    unified_design='unified_design_native_runtime',consistent_evidence='consistent_evidence_native_runtime')
SOURCE_DIGESTS=('initial_source_SHA256','initial_source10_SHA256','initial_source12_SHA256','initial_source19_SHA256')
OBSERVER_FILES=('own_proof','own_capacity','own_capacity_trace','own_initial_identity',
    'reference_proof','reference_capacity','reference_capacity_trace',
    'reference_startup','reference_trace','reference_config')


def source_digest(value):
    present=[k for k in SOURCE_DIGESTS if k in value]
    require(len(present)==1,'one actually recorded initial source digest field required')
    return value[present[0]]


def validate_recipe(original,cfg,*,arm,phase):
    """Only explicitly recorded device metadata can vary operationally."""
    e=cfg['experiment'];gpus=e.get('assigned_GPUs');cpus=e.get('CPU_affinity_per_GPU')
    require(isinstance(gpus,list) and bool(gpus)
        and all(isinstance(g,str) and g.isdigit() for g in gpus) and len(set(gpus))==len(gpus)
        and isinstance(cpus,dict) and set(cpus)==set(gpus)
        and all(isinstance(ids,list) and bool(ids) and len(set(ids))==len(ids)
            and all(type(i) is int and i>=0 for i in ids) for ids in cpus.values()),
        'explicit physical GPU list and per-GPU integer CPU affinity required')
    normalized=deepcopy(cfg)
    for field in ['assigned_GPUs','CPU_affinity_per_GPU']:
        if field in original['experiment']:
            normalized['experiment'][field]=deepcopy(original['experiment'][field])
        else:normalized['experiment'].pop(field,None)
    validate_confirmation(original,normalized,arm=arm,phase=phase)
    require(type(cfg['training_seed']) is int and cfg['training_seed']==202,
        'actual independent seed202 required')
    return validate_arm(cfg,arm,phase)


def validate_own_engineering(proof,capacity,*,arm):
    variant=ALIASES.get(arm,arm)
    require(proof.get('status')=='PASS_ACTUAL_COLD_NATIVE_SS_FLOW_IDENTITY_AND_BACKWARD'
        and proof.get('variant')==variant and proof.get('source_tensors')==SOURCE_COUNTS[arm]
        and proof.get('optimizer_updates')==0 and proof.get('LoRA_tensors')==360
        and proof.get('actual_CPU_CUDA_numpy_python_RNG_preserved') is True
        and set(proof.get('fields_equal_to_native',{}))=={'global','proj','negative_global','negative_proj'}
        and all(v is True for v in proof['fields_equal_to_native'].values()),
        'actual own native field/velocity/loss/backward proof required')
    require(capacity.get('status')=='PASS_ACTUAL_COLD_NATIVE_TASK_SS96_THREE_UPDATE_CAPACITY'
        and capacity.get('variant')==variant and capacity.get('source_tensors')==SOURCE_COUNTS[arm]
        and capacity.get('temporary_optimizer_updates')==3
        and (capacity.get('global_batch'),capacity.get('microbatch'),capacity.get('accumulation'),
             capacity.get('world_size'),capacity.get('LoRA_tensors'))==(96,96,1,1,360)
        and capacity.get('actual_optimizer_states')==360+SOURCE_COUNTS[arm]
        and capacity.get('actual_optimizer_steps')==[3]
        and capacity.get('scientific_training_updates')==0
        and capacity.get('scientific_checkpoint_exported') is False
        and capacity.get('actual_initial_native32_recorded_identity') is True
        and capacity.get('frozen_parameters_buffers_exact') is True
        and (not SOURCE_COUNTS[arm] or capacity.get('all_source_gradients_positive_finite_at_update3') is True),
        'actual own three-update micro96 gradient/optimizer/frozen capacity proof required')
    require(proof['initial_LoRA_SHA256']==capacity['initial_LoRA_SHA256']
        and source_digest(proof)==source_digest(capacity),'proof/capacity must use the same original weights')


def validate_admission(startup,proof,capacity,reference,*,arm):
    require(startup.get('status')=='PASS_ACTUAL_COLD_NATIVE_TASK_SS_STARTUP'
        and startup.get('variant')==ALIASES.get(arm,arm)
        and startup.get('source_tensors')==SOURCE_COUNTS[arm]
        and startup.get('LoRA_tensors')==360
        and (startup.get('global_batch'),startup.get('microbatch'),startup.get('accumulation'),
             startup.get('actual_optimizer_states'))==(96,96,1,0),
        'actual own fresh native SS96 startup required')
    validate_own_engineering(proof,capacity,arm=arm)
    for actual in [proof,capacity]:
        require(startup['initial_LoRA_SHA256']==actual['initial_LoRA_SHA256']
            and source_digest(startup)==source_digest(actual),
            'scientific run must start from own original proof/capacity weights, not their trained weights')
    require(reference.get('status')=='PASS_ACTUAL_COLD_NATIVE_TASK_SS_STARTUP'
        and reference.get('variant')=='vanilla' and reference.get('source_tensors')==0,
        'actual same-seed vanilla startup required')
    for key in ['initial_LoRA_SHA256','post_constructor_RNG','TRAIN_order_SHA256']:
        require(startup[key]==reference[key],'actual chosen startup differs from same-seed V1: '+key)


def validate_engineering(values,*,arm):
    """Validate actual cold proof and capacity against the same-seed V1 runs."""
    proof,cap=values['own_proof'],values['own_capacity']
    ref_proof,ref_cap=values['reference_proof'],values['reference_capacity']
    validate_own_engineering(proof,cap,arm=arm)
    validate_admission(values['reference_startup'],ref_proof,ref_cap,
        values['reference_startup'],arm='V1')
    require(ref_proof.get('variant')=='vanilla' and ref_cap.get('variant')=='vanilla',
        'own actual same-seed vanilla engineering controls required')
    # The scientific startup is checked again from the actual chosen process.
    # Here each engineering phase is paired with its own reference phase.
    for current,reference in [(proof,ref_proof),(cap,ref_cap)]:
        require(current.get('variant')==ALIASES.get(arm,arm)
            and current.get('source_tensors')==SOURCE_COUNTS[arm]
            and current.get('LoRA_tensors')==reference.get('LoRA_tensors')==360,
            'actual engineering model identity required')
        for key in ['initial_LoRA_SHA256','post_constructor_RNG','TRAIN_order_SHA256']:
            require(current[key]==reference[key],'same-seed engineering differs from V1: '+key)
    for key in ['same_noise_velocity_SHA256','same_noise_native_flow_loss']:
        require(proof[key]==ref_proof[key],'own cold native output differs from V1: '+key)
    require(math.isfinite(proof['actual_native_loss']) and math.isfinite(proof['LoRA_gradient_L1'])
        and proof['LoRA_gradient_L1']>0,'finite actual native loss and positive LoRA backward required')
    trace=values['own_capacity_trace']
    require(len(trace)==3 and [r['step'] for r in trace]==[1,2,3]
        and trace==values['reference_capacity_trace'],
        'all three actual micro96 TRAIN asset/noise/timestep/global traces must equal V1')
    require(values['own_initial_identity'].get('status')=='PASS_ACTUAL_INITIAL_NATIVE32_RECORDED_METRIC_IDENTITY'
        and values['own_initial_identity'].get('assets')==32
        and values['own_initial_identity'].get('optimizer_updates')==0,
        'actual initial32 native metric identity before capacity updates required')
    if family(arm)=='consistent_evidence':
        require(proof.get('learned_projection_tensors')==0
            and proof.get('actual_initial_core10_equal_Top1') is True
            and proof.get('shared_carrier_use_retarget_evidence') is True
            and proof.get('actual_selected_evidence_equals_R_appearance') is True
            and (arm!='E_LEARNED_RAW' or
                proof.get('actual_initial_head_probability_equals_frozen_prior') is True),
            'own shared evidence, identity V and exact initial prior GPU proofs required')


def validate_observer_values(plan,values,cfg,arm):
    require(plan.get('schema')=='actual-selected-SS202-paired-prefix-admission-v1'
        and plan.get('module_arm')==arm and type(plan.get('training_seed')) is int
        and plan['training_seed']==cfg['training_seed']==202,
        'own actual seed202 admission inputs required')
    ref=values['reference_config']
    require(type(ref['training_seed']) is int and ref['training_seed']==202
        and ref['experiment']['native_variant']=='vanilla',
        'reference must be an actually configured seed202 vanilla run')
    validate_arm(ref,'V1','science')
    trace=values['reference_trace']
    require(len(trace)==5 and [r['step'] for r in trace]==[1,2,3,4,5],
        'five actual same-seed vanilla updates required before chosen science')
    validate_engineering(values,arm=arm)
    if family(arm)=='consistent_evidence':
        root=Path(cfg['experiment']['paired_reference_native_root'])
        require(plan['reference_startup']['path']==str(root/'startup.json')
            and plan['reference_trace']['path']==str(root/'training-noise-trace.jsonl')
            and plan['own_proof']['path']==cfg['experiment']['cold_proof_json']
            and plan['own_capacity']['path']==cfg['experiment']['micro96_capacity_json'],
            'existing consistent-evidence checks must read exactly the same actual prefix and own GPU proofs')


def load_observer(path,cfg,arm):
    path=Path(path);plan=json.loads(path.read_text());check=ClosedInputs(plan['canonical_local_root'])
    check.check(path,path.parent/'file4-closure.json')
    values={key:read_file(plan[key],check,lines=key.endswith('_trace')) for key in OBSERVER_FILES}
    validate_observer_values(plan,values,cfg,arm)
    return values,check


@contextmanager
def observed_trainer(runtime,*,arm,values,check):
    original=runtime.selected_trainer
    def selected(variant):
        Parent=original(variant)
        class Observed(Parent):
            def run(self):
                out=Path(self.output_dir)
                startup=json.loads((out/'startup.json').read_text())
                validate_admission(startup,values['own_proof'],values['own_capacity'],
                    values['reference_startup'],arm=arm)
                from .probe_member_feature_fullbatch import _frozen_digest
                before={name:_frozen_digest(model) for name,model in self.models.items()}
                result=super().run()
                require(self.step==480,'actual complete selected confirmation budget required')
                states=self.optimizer.state
                require(len(states)==360+SOURCE_COUNTS[arm]
                    and {int(value['step']) for value in states.values()}=={480},
                    'actual original LoRA and own source optimizer must complete all 480 updates')
                after={name:_frozen_digest(model) for name,model in self.models.items()}
                require(before==after,'selected confirmation changed frozen base or matcher/anchor tensors')
                from .cascade_view_roles import file_sha256
                for path,digest in check.inputs.items():require(file_sha256(path)==digest,'closed admission input changed')
                (out/'confirmation-paired-prefix-and-frozen.json').write_text(json.dumps(dict(
                    status='PASS_ACTUAL_SELECTED_SS202_PAIRED_PREFIX_AND_FROZEN480',module_arm=arm,
                    variant=variant,training_seed=202,actual_same_seed_V1_startup_and_first5_inputs_equal=True,
                    own_original_proof_capacity_initial_weights_equal=True,actual_frozen_parameters_buffers_exact=True,
                    frozen_SHA256=after,updates=480,actual_optimizer_states=len(states),
                    actual_optimizer_steps=[480],source_tensors=SOURCE_COUNTS[arm],LoRA_tensors=360,
                    verified_admission_inputs=check.inputs,
                    full_goal_complete=False),indent=2)+'\n')
                return result

            def run_step(self,packs):
                result=super().run_step(packs)
                if self.step<5:
                    require(len(self._paired_trace)==1,'one original fullbatch model forward per update required')
                    record=dict(step=self.step+1,asset_ids=[p.get('_sha256',[]) for p in packs],**self._paired_trace[0])
                    require(record==values['reference_trace'][self.step],
                        'actual selected TRAIN assets/noise/timestep/global differ from same-seed V1')
                return result
        return Observed
    runtime.selected_trainer=selected
    try:yield
    finally:runtime.selected_trainer=original


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--original-config',type=Path,required=True)
    p.add_argument('--module-arm',required=True);p.add_argument('--phase',choices=['proof','capacity','science'],required=True)
    p.add_argument('--observer-plan',type=Path);p.add_argument('--sample-id');p.add_argument('--expected-conditioning-sha256')
    a=p.parse_args();cfg=json.loads(a.config.read_text());base=json.loads(a.original_config.read_text())
    variant=validate_recipe(base,cfg,arm=a.module_arm,phase=a.phase)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    require(gpu in cfg['experiment']['assigned_GPUs'] and gpu in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','),
        'one explicitly assigned physical GPU required')
    require((a.phase=='science')==(a.observer_plan is not None),'own paired admission plan required only for science')
    require(a.phase!='proof' or a.sample_id and a.expected_conditioning_sha256,'predeclared exact TRAIN proof input required')
    os.sched_setaffinity(0,set(cfg['experiment']['CPU_affinity_per_GPU'][gpu]))
    import torch
    from experiments.multiview_finetune.runtime import install_hooks
    runtime=importlib.import_module('.'+RUNTIMES[family(a.module_arm)],__package__)
    torch.cuda.set_device(0);torch.set_num_threads(2);install_hooks()
    if a.phase=='proof':runtime.proof(cfg,a,variant)
    elif a.phase=='capacity':
        with runtime.cold_source_binding(variant):runtime.native_training(cfg,a.phase,variant)
    else:
        values,check=load_observer(a.observer_plan,cfg,a.module_arm)
        with observed_trainer(runtime,arm=a.module_arm,values=values,check=check),runtime.cold_source_binding(variant):
            runtime.native_training(cfg,a.phase,variant)


if __name__=='__main__':main()
