"""Build a selected-pipeline freeze from actual closed evidence, without TEST NN."""
import argparse
import json
import os
from pathlib import Path

from .cascade_view_roles import ClosedInputs,file_sha256 as sha
from .collect_module_seed_confirmation import read_file,require
from .pipeline_population_contract_v2 import model_binding
from .selected_pipeline_freeze_v2 import make_freeze


SCHEMA='selected-physical-pipeline-freeze-closed-inputs-v2'


def required_sources():
    root=Path(__file__).resolve().parent
    return [str(root/(name+'.py')) for name in ['freeze_selected_pipeline','selected_pipeline_freeze_v2',
        'selected_pipeline_model_contract','selected_pipeline_joint_summary_v2',
        'selected_pipeline_cascade_contract','pipeline_population_contract_v2',
        'pipeline_confirmation_recipe_v2','collect_pipeline_seed_confirmation_v2',
        'pipeline_training_seed_summary_v2','collect_selected_pbr_seed_confirmation',
        'selected_pbr_seed_confirmation','native_population_contract','evaluate_bound_population']]


def execute(plan_path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit CPU-only freeze preparation required')
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text())
    require(plan.get('schema')==SCHEMA,'actual closed selected-pipeline freeze plan required')
    check=ClosedInputs(plan['canonical_local_root']);check.check(plan_path,plan_path.parent/'file4-closure.json')
    require(set(required_sources())<=set(plan['source_SHA256']),'pin every actual freeze/confirmation numerical validator')
    for path,digest in plan['source_SHA256'].items():require(sha(path)==digest,'actual freeze source changed')
    specs=plan['development'];development={k:read_file(v,check) for k,v in specs.items()}
    selection=read_file(plan['selection'],check);qa=read_file(plan['cache_QA'],check)
    protocol=read_file(plan['evaluation_protocol'],check)
    for name in ['metric_registry','distribution_metric_registry','generation_template']:
        path=protocol[name];require(sha(path['path'])==path['SHA256'],'actual evaluation protocol input changed')
    for path,digest in protocol['source_SHA256'].items():require(sha(path)==digest,'actual final evaluation source changed')
    ss={}
    for arm,spec in plan['SS_bundles'].items():
        bundle=read_file(spec,check);receipt=spec['receipt'];hashes={}
        for key in ['native_config','checkpoint','conditioner_checkpoint']:
            if bundle[key] is not None:hashes[bundle[key]]=check.check(bundle[key],receipt)
        cfg=json.loads(Path(bundle['native_config']).read_text())
        ss[arm]=model_binding(bundle,cfg,hashes,arm=arm)
    pbrconf=development['PBR_two_seeds']
    binds=pbrconf['model_bindings'];primary=binds['101'] if '101' in binds else binds[101]
    require(set(primary)==set(plan['PBR_model_receipts']),'actual selected PBR and strong controls must match')
    for mode,binding in primary.items():
        for key in ['native_config','checkpoint','conditioner_checkpoint']:
            if binding[key] is not None:
                value=binding[key]
                require(check.check(value['path'],plan['PBR_model_receipts'][mode])==value['SHA256'],
                    'actual primary PBR bytes differ from independently confirmed model')
    # Read the actual independent-model bytes as well as the summary files.
    # Preserve real seed202 identities; never manufacture a seed101 alias.
    independent=[development['SS_two_seeds']['model_evidence'],pbrconf['model_bindings']]
    for group_index,group in enumerate(independent):
        for models in group.values():
            for item in models.values():
                binding=item['model_binding'] if group_index==0 else item
                receipt=Path(binding['native_config']['path']).parent/'file4-closure.json'
                for key in ['native_config','checkpoint','conditioner_checkpoint']:
                    if binding[key] is not None:
                        value=binding[key]
                        require(check.check(value['path'],receipt)==value['SHA256'],
                            'actual independently confirmed model bytes changed')
    stages=plan['original_stages']
    for stage in stages.values():
        for key in ['config','checkpoint']:
            value=stage[key];require(sha(value['path'])==value['SHA256'],'actual original multiview stage bytes differ')
    rows=read_file(plan['test_manifest'],check,lines=True)
    train=read_file(plan['train_manifest'],check,lines=True)
    validation=read_file(plan['validation_manifest'],check,lines=True)
    seeds=read_file(plan['test_seeds'],check)
    parities={arm:read_file(spec,check) for arm,spec in plan['parities'].items()}
    evaluations={arm:[read_file(spec,check) for spec in items] for arm,items in plan['SS_evaluations'].items()}
    def pinned(spec):return {k:spec[k] for k in ['path','SHA256']}
    evidence={k:pinned(v) for k,v in specs.items()}
    evidence.update(selection=pinned(plan['selection']),cache_QA=pinned(plan['cache_QA']))
    frozen=make_freeze(selected_pair=tuple(plan['selected_pair']),selection=selection,
        ss_models=ss,pbr_models=primary,original_stages=stages,rows=rows,train_rows=train,
        validation_rows=validation,seeds=seeds,parities=parities,development=development,
        development_evidence=evidence,ss_evaluations=evaluations,cache_QA=qa,
        test_manifest=pinned(plan['test_manifest']),evaluation_protocol=protocol,
        heldout_quality_read_before_freeze=plan['heldout_quality_read_before_freeze'])
    frozen.update(verified_closed_inputs=check.inputs,source_SHA256=plan['source_SHA256'],
        CPU_neural_forward_or_training_updates=0)
    for path,digest in check.inputs.items():require(sha(path)==digest,'actual closed evidence changed during freeze')
    for path,digest in plan['source_SHA256'].items():require(sha(path)==digest,'freeze source changed during preparation')
    for name in ['metric_registry','distribution_metric_registry','generation_template']:
        value=protocol[name];require(sha(value['path'])==value['SHA256'],'frozen evaluation input changed during preparation')
    for path,digest in protocol['source_SHA256'].items():require(sha(path)==digest,'final evaluation source changed during preparation')
    for stage in stages.values():
        for key in ['config','checkpoint']:
            value=stage[key];require(sha(value['path'])==value['SHA256'],'original stage changed during preparation')
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    (output/'freeze.json').write_text(json.dumps(frozen,indent=2,allow_nan=False)+'\n')
    return frozen


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();value=execute(a.plan)
    print(json.dumps(dict(status=value['status'],selected_pair=value['selected_pair'],
        final508_SS_generated=False,full_goal_complete=False)),flush=True)


if __name__=='__main__':main()
