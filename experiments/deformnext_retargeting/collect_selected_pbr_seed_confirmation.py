"""Read actual closed two-seed PBR stages; no neural forward or model selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from .cascade_view_roles import ClosedInputs,file_sha256 as sha
from .collect_module_seed_confirmation import (
    frozen_base_comparison,read_file,require,validate_file,verify_terminal,
)
from .native_pbr_stage_summary import is_digest
from .selected_pbr_seed_confirmation import modes_for_selected,summarize
from .pbr_confirmation_initial_identity import read_initial_identity


SCHEMA='selected-PBR-two-seed-closed-stage-inputs-v1'


def validate_trace(rows):
    require(len(rows)==1900 and [r['step'] for r in rows]==list(range(1,1901)),
        'actual complete1900 PBR updates required')
    for row in rows:
        require(type(row['step']) is int and isinstance(row['asset_ids'],list) and len(row['asset_ids'])==6
            and all(isinstance(s,str) and s for s in row['asset_ids'])
            and all(is_digest(row[k]) for k in ['x_t_SHA256','t_SHA256','global_SHA256']),
            'actual complete per-rank micro6 input/noise/timestep/global trace required')


def training_trace_digest(rows):
    """Same parsed-row digest as native stage snapshots; raw file SHA is separate."""
    validate_trace(rows)
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),
        allow_nan=False).encode()).hexdigest()


def collect(plan,check):
    require(plan.get('schema')==SCHEMA and set(plan['models'])=={'101','202'},
        'actual two-seed PBR closed-input plan required')
    modes=modes_for_selected(plan['selected_mode'])
    original_spec,mv_spec=plan['original_checkpoint'],plan['original_multiview_config']
    for spec in [original_spec,mv_spec]:
        validate_file(spec,closed=False);require(sha(spec['path'])==spec['SHA256'],'original multiview bytes changed')
    original_multiview=json.loads(Path(mv_spec['path']).read_text())
    import torch
    torch.set_num_threads(2)
    original=torch.load(original_spec['path'],map_location='cpu',weights_only=True,mmap=True)
    records,configs,hashes,bases={},{},{},{}
    proofs={mode:read_file(plan['engineering_proofs'][mode],check) for mode in modes}
    traces={};terminal_evidence={}
    for seed in (101,202):
        own=plan['models'][str(seed)]
        require(set(own)==set(modes),'actual selected PBR and deduplicated controls required')
        records[seed],configs[seed],hashes[seed],bases[seed],traces[seed],terminal_evidence[seed]=({} for _ in range(6))
        for mode in modes:
            item=own[mode];record=read_file(item['snapshot'],check);cfg=read_file(item['config'],check)
            require(cfg['training_seed']==seed,'actual PBR config belongs to another training seed')
            native=Path(cfg['output_dir'])
            validation=native/'validation.jsonl';receipt=native/'file4-closure.json'
            validation_spec=dict(path=str(validation),SHA256=check.check(validation,receipt),receipt=str(receipt))
            require(read_file(validation_spec,check,lines=True)==record['validation_rows'],
                'snapshot rows differ from the actual closed six-epoch native validation log')
            initial,evidence=read_initial_identity(native,record['science_plan'],record['validation_rows'][0],check)
            if 'initial_render_identity' in record:
                require(record['initial_render_identity']==initial,'snapshot initial producer proof differs')
            record=dict(record,initial_render_identity=initial,initial_identity_evidence=evidence)
            actual={}
            for key,expected in [('config',native/'config.json'),('checkpoint',native/'ckpts/denoiser_step0001900.pt'),
                                 ('conditioner_checkpoint',None if mode=='vanilla' else native/'ckpts/deform_conditioner_step0001900.pt')]:
                spec=item[key]
                require((spec is None)==(expected is None),'actual PBR source presence differs')
                if expected is None:continue
                validate_file(spec);require(spec['path']==str(expected),'actual PBR model path differs')
                require(check.check(spec['path'],spec['receipt'])==spec['SHA256'],'actual PBR model bytes differ')
                actual[spec['path']]=spec['SHA256']
            for name in ['active','terminal','wrapper','progress']:
                require(read_file(item['completion_files'][name],check)==record['completion_proof'][name],
                    'snapshot differs from actual closed training completion: '+name)
            terminal_evidence[seed][mode]=verify_terminal(record['completion_proof']['terminal'])
            ranks={str(r['rank']):r for r in record['completion_proof']['ranks']}
            require(set(item['rank_completions'])==set(ranks)=={'0','1','2','3'},'actual complete rank files required')
            traces[seed][mode]={}
            for rank in ['0','1','2','3']:
                require(read_file(item['rank_completions'][rank],check)==ranks[rank],'actual completion rank differs')
                spec=item['training_traces'][rank]
                rows=read_file(spec,check,lines=True);validate_trace(rows)
                require(record['training_trace_inputs'][rank]==spec,
                    'snapshot trace must identify the same actual file, raw SHA and receipt')
                require(training_trace_digest(rows)==record['training_trace_digests'][rank],
                    'snapshot parsed trace digest differs')
                traces[seed][mode][rank]=rows
                if mode!='vanilla':
                    require(rows==traces[seed]['vanilla'][rank],'actual per-rank1900 traces differ within TRAIN seed')
            current=torch.load(item['checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
            bases[seed][mode]=frozen_base_comparison(original,current);del current
            records[seed][mode],configs[seed][mode],hashes[seed][mode]=record,cfg,actual
    del original
    result=summarize(records,configs,hashes,selected=plan['selected_mode'],proofs=proofs,
        original_multiview=original_multiview,original_checkpoint=original_spec,frozen_base_proofs=bases)
    require(any(traces[101]['vanilla'][rank]!=traces[202]['vanilla'][rank] for rank in ['0','1','2','3']),
        'actual independent PBR training streams must differ')
    for spec in [original_spec,mv_spec]:require(sha(spec['path'])==spec['SHA256'],'original bytes changed during audit')
    for path,digest in check.inputs.items():require(sha(path)==digest,'closed input changed during reduction')
    result.update(actual_checkpoint_bytes_verified=True,actual_frozen_base_tensor_comparisons=bases,
        actual_terminal_evidence=terminal_evidence,verified_inputs=dict(check.inputs),
        CPU_neural_forward_or_training_updates=0)
    return result


def required_sources():
    root=Path(__file__).resolve().parent
    return [str(root/(name+'.py')) for name in ['collect_selected_pbr_seed_confirmation',
        'selected_pbr_seed_confirmation','selected_pipeline_model_contract','pipeline_population_contract_v2',
        'collect_module_seed_confirmation','native_pbr_stage_summary','pbr_confirmation_recipe',
        'r_mediated_pbr_science_recipe','cascade_view_roles','pbr_confirmation_initial_identity',
        'pbr_initial_render_identity','evaluate_pbr_streaming','evaluate_r_mediated_pbr_initial']]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    args=p.parse_args();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit CPU-only closed reader required')
    plan=json.loads(args.plan.read_text());check=ClosedInputs(plan['canonical_local_root'])
    check.check(args.plan,args.plan.parent/'file4-closure.json')
    require(set(required_sources())<=set(plan['source_SHA256']),'pin the actual PBR confirmation reader and numerical validators')
    for path,digest in plan['source_SHA256'].items():require(sha(path)==digest,'actual confirmation source changed')
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    result=collect(plan,check)
    for path,digest in plan['source_SHA256'].items():require(sha(path)==digest,'confirmation source changed during audit')
    (out/'confirmation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
