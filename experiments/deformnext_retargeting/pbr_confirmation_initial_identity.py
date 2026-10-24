"""Read the original cold latent identity and its predeclared renderer bounds."""
import math
from copy import deepcopy
from pathlib import Path

from .collect_module_seed_confirmation import read_file,require
from .native_pbr_stage_summary import IDENTITY
from .pbr_initial_render_identity import POLICY,RENDER_ABSOLUTE_LIMITS,validate_initial_latent_and_render


def initial_identity(first,reference,*,producer=None,declared_policy=None):
    """Recompute the actual producer proof without altering any metric value."""
    proof=deepcopy(validate_initial_latent_and_render(first,reference))
    if producer is None:
        require(declared_policy is None and proof['original_render_metrics_exact'],
            'legacy stages require exact rendering; bounded producers must retain their actual proof')
    else:
        require(declared_policy==POLICY and producer==proof,
            'actual native initial proof differs from its original cold32 reference')
    return proof


def read_initial_identity(native,stage_plan,first,check):
    """Bind the actual native plan, cold32 and producer file through file4 receipts."""
    native,stage_plan=Path(native),Path(stage_plan)
    def read(path):
        path=Path(path);receipt=path.parent/'file4-closure.json'
        spec=dict(path=str(path),SHA256=check.check(path,receipt),receipt=str(receipt))
        return read_file(spec,check),spec
    plan,plan_spec=read(stage_plan)
    require(plan['output']==str(native),'initial proof belongs to another native stage')
    reference,reference_spec=read(plan['initial_reference'])
    proof_path=native/'initial-latent-render-identity.json'
    producer,producer_spec=(read(proof_path) if proof_path.is_file() else (None,None))
    proof=initial_identity(first,reference,producer=producer,declared_policy=plan.get('initial_identity_policy'))
    return proof,dict(science_plan=plan_spec,cold32_reference=reference_spec,producer_proof=producer_spec)


def compare_initial(record,reference):
    """Exact RGB/noise/latent pairing; only existing render suffix bounds may vary."""
    first=record['validation_rows'][0]
    require(first['step']==reference['step']==0 and len(first['quick_per_asset'])==len(reference['quick_per_asset'])==4
        and first['quick_panel_ids']==reference['quick_panel_ids'], 'same ordered actual step0 fixed4 required')
    proof=record.get('initial_render_identity')
    comparisons=[]
    for row,ref in zip(first['quick_per_asset'],reference['quick_per_asset']):
        require(all(row[k]==ref[k] for k in IDENTITY)
            and row['actual_prediction_PBR_latent_SHA256']==ref['actual_prediction_PBR_latent_SHA256'],
            'initial RGB, noise, GT, camera and prediction latent must remain exactly equal')
        require(row['metrics'].keys()==ref['metrics'].keys(),'same actual initial metric names required')
        deltas={}
        for key,value in row['metrics'].items():
            before=ref['metrics'][key]
            require(math.isfinite(value) and math.isfinite(before),'finite actual initial render metrics required')
            suffix=next((name for name in RENDER_ABSOLUTE_LIMITS if key.endswith(name)),None)
            limit=0. if suffix is None else RENDER_ABSOLUTE_LIMITS[suffix]
            difference=abs(value-before)
            require(difference<=limit,'initial render exceeds its existing absolute bound: '+key)
            if difference:
                require(proof is not None and proof.get('policy')==POLICY
                    and proof.get('actual_inputs_noise_GT_and_PBR_latent_exact') is True
                    and proof.get('render_metrics_within_declared_absolute_bounds') is True
                    and proof.get('render_absolute_limits')==RENDER_ABSOLUTE_LIMITS
                    and proof.get('unlisted_metrics_require_exact_equality') is True,
                    'nonidentical renderer values require the actual validated native producer proof')
            deltas[key]=dict(absolute_delta=difference,absolute_limit=limit)
        comparisons.append(dict(sample_id=row['sample_id'],metrics=deltas))
    return dict(policy=POLICY,actual_inputs_noise_and_latents_exact=True,comparisons=comparisons,
        metric_values_unchanged=True,renderer_bit_identity_not_required=True)
