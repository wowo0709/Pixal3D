"""Actual-seed preservation in snapshot metadata; no neural runtime is used."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.common_geometry_pbr_contract import MODES
from experiments.deformnext_retargeting.snapshot_selected_pbr_stage import validate_completion,validate_runtime
from test_native_pbr_stage_summary import completion
from test_r_mediated_pbr_science import recipe


def completed(mode='top1-only',seed=202):
    cfg,_,_=recipe(mode,world=4);cfg['training_seed']=seed
    proof=completion(mode);proof['terminal']['actual_session_drained']=True
    for row in proof['ranks']:row['training_seed']=seed
    return cfg,proof


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('seed',[101,202])
def test_actual_seed_and_mode_are_validated_without_normalizing_or_mutating(mode,seed):
    cfg,proof=completed(mode,seed);before=deepcopy((cfg,proof))
    validate_completion(cfg,mode,proof,training_seed=seed)
    assert (cfg,proof)==before


@pytest.mark.parametrize('damage',['wrong-config-seed','one-other-rank-seed','string-seed',
    'boolean-seed','incomplete-updates','missing-drain','other-PID','NaN-birth',
    'bad-exit','other-GPU-group','not-completed','wrong-wrapper-batch'])
def test_unfinished_or_other_seed_training_cannot_be_relabelled_as_confirmation(damage):
    cfg,proof=completed();seed=202
    if damage=='wrong-config-seed':cfg['training_seed']=101
    elif damage=='one-other-rank-seed':proof['ranks'][2]['training_seed']=101
    elif damage=='string-seed':seed='202'
    elif damage=='boolean-seed':seed=True
    elif damage=='incomplete-updates':proof['ranks'][0]['actual_optimizer_updates']=1140
    elif damage=='missing-drain':proof['terminal'].pop('actual_session_drained')
    elif damage=='other-PID':proof['terminal']['PID']+=1
    elif damage=='NaN-birth':proof['active']['birth']=float('nan')
    elif damage=='bad-exit':proof['terminal']['returncode']=1
    elif damage=='other-GPU-group':proof['active']['physical_GPUs']=['4','5','6','7']
    elif damage=='not-completed':proof['progress']['stage']='running'
    else:proof['wrapper']['microbatch']=12
    with pytest.raises(ValueError):validate_completion(cfg,'top1-only',proof,training_seed=seed)


def runtime():
    return [dict(step=step,rank=2,world_size=4,effective_batch_size=24,
        accumulation_packs=1,microbatch_assets=[6],K=4,loss=dict(loss=.1),compute_seconds=1.)
        for step in range(1,1901)]


def test_actual_micro6_full1900_runtime_preserved():
    rows=runtime();before=deepcopy(rows);validate_runtime(rows,2);assert rows==before


@pytest.mark.parametrize('damage',['incomplete','other-rank','micro','accumulation','NaN-loss','step'])
def test_runtime_or_capacity_fragments_cannot_replace_complete_science(damage):
    rows=runtime()
    if damage=='incomplete':rows.pop()
    elif damage=='other-rank':rows[0]['rank']=1
    elif damage=='micro':rows[0]['microbatch_assets']=[3,3]
    elif damage=='accumulation':rows[0]['accumulation_packs']=2
    elif damage=='NaN-loss':rows[0]['loss']['loss']=float('nan')
    else:rows[1]['step']=1
    with pytest.raises(ValueError):validate_runtime(rows,2)
