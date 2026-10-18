"""Check same-weight value-only intervention and exact fixed-final normal replay."""
import argparse
import json
from pathlib import Path

from .compare_ss_noise_panels import compare
from .semantic_topk_search import write_new


def summarize(normal,altered,reference):
    for e in [normal,altered]:
        if (e.get('status')!='PASS' or e.get('optimizer_updates_during_evaluation')!=0
            or e.get('step')!=480 or e.get('weights')!='raw' or e.get('sampling_steps')!=12 or e.get('guidance')!=3.):
            raise ValueError('diagnostic must use fixed raw480 and zero optimizer updates')
        if e.get('target_modality')!='rgb' or e.get('input_camera_ids')!=[8,9,10,11] or e.get('input_states')!=['M1','M0','M0','M0']:
            raise ValueError('task diagnostic requires original four-view edited RGB')
    if len(normal['per_asset'])!=32 or len(altered['per_asset'])!=32 or len(reference['per_asset'])!=128:
        raise ValueError('task diagnostic requires common32 and fixed-final128')
    for k in ['checkpoint','conditioner_checkpoint','arm_config','input_states','input_camera_ids']:
        if normal.get(k)!=altered.get(k):raise ValueError('same-weight diagnostic input differs: '+k)
    if normal.get('task_value_control')!='normal' or altered.get('task_value_control')!='wrong-address':
        raise ValueError('task diagnostic control differs')
    events=altered.get('task_value_control_events',[])
    if not events or not all(e['matching_logits_and_weights_unchanged'] for e in events) or sum(e['reassigned_valid_slots'] for e in events)<=0:
        raise ValueError('value intervention did not preserve keys/weights or change addresses')
    byid={x['sample_id']:x for x in reference['per_asset']}
    for r in normal['per_asset']:
        if any(r[k]!=byid[r['sample_id']][k] for k in ['source_cluster_id','seed','tp','fp','fn','target_voxels_sha256']):
            raise ValueError('normal diagnostic did not exactly replay its fixed-final cases')
    return {'status':'PASS_COMPLETED_TASK_VALUE_ADDRESS_DIAGNOSTIC',
            'normal_exact_to_same_weight_final128_subset':True,
            'normal_minus_wrong_address':compare([('0',altered,normal)]),
            'reassigned_valid_slots_across_actual_calls':sum(e['reassigned_valid_slots'] for e in events),
            'keys_scores_weights_candidates_support_and_source_value_multiset_preserved':True,
            'scope':'Saved-weight common32/one-noise value-address sensitivity,not a fresh-training ablation or exact Top1 accuracy;native-spatial comparison remains separate.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['normal','altered','reference','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();result=summarize(*[json.loads(getattr(a,n).read_text()) for n in ['normal','altered','reference']])
    write_new(a.output,result);print(result['status'],flush=True)


if __name__=='__main__':main()
