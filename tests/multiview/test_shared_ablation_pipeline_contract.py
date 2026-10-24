"""Reject wrong physical checkpoint families and unpaired downstream inputs."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting import shared_ablation_pipeline_contract as contract
from experiments.deformnext_retargeting.evaluate_shared_evidence_ablation_checkpoint import control_metadata
from experiments.deformnext_retargeting.shared_evidence_ablation_recipe import VARIANTS, SOURCE_COUNTS


def fixture(arm='A_FIXED_UNIFORM'):
    root = '/runs/'+arm
    cfg = dict(training_seed=101, experiment=dict(ablation_source_SHA256={'/source/control.py':'a'*64}))
    binding = dict(schema=contract.SCHEMA,module_arm=arm,variant=arm,
        runtime_family='shared_evidence_ablation',checkpoint_bundle_schema=contract.BUNDLE_SCHEMA,
        step=480,source_tensors=SOURCE_COUNTS[arm],evidence_mode=VARIANTS[arm][1],
        shared_carrier_use_evidence=True,learned_projection_tensors=0,
        conditioning_mode='edited-rgb',source_teacher_updates=0,native_task_objective='SS_flow_mse',
        **control_metadata(cfg,arm))
    for key,path in [('native_config','config.json'),('checkpoint','ckpts/denoiser_step0000480.pt'),
                     ('conditioner_checkpoint','ckpts/deform_conditioner_step0000480.pt')]:
        binding[key] = dict(path=root+'/'+path,SHA256='b'*64)
    record = dict(status='PASS',stage='ss64',step=480,weights='raw',native_variant=arm,
        actual_source_tensors=SOURCE_COUNTS[arm],actual_LoRA_tensors=360,
        optimizer_updates_during_evaluation=0,target_modality='rgb',
        input_states=['M1','M0','M0','M0'],input_camera_ids=[8,9,10,11],
        generation_batch_size=8,sampling_steps=12,guidance=3.,deterministic_algorithms=True,
        evaluation_mode='final128',shared_carrier_use_evidence=True,learned_projection_tensors=0,
        **control_metadata(cfg,arm))
    for key,field in [('native_config','arm_config'),('checkpoint','checkpoint'),
                      ('conditioner_checkpoint','conditioner_checkpoint')]:
        record[field] = binding[key]['path']
    record['verified_inputs'] = {binding[k]['path']:binding[k]['SHA256']
        for k in ('native_config','checkpoint','conditioner_checkpoint')}
    record['per_asset'] = [dict(sample_id=str(i),source_cluster_id=str(i % 93),seed=i,
        target_voxels_sha256='c'*64,tp=1,fp=2,fn=3) for i in range(128)]
    record['native_SS_artifacts'] = dict(status='PASS_ACTUAL_ORIGINAL_SHARED_ABLATION_SS128_CAPTURE',
        archived_asset_count=128,same_forward_prediction_and_target_capture=True,
        predicted_SS_is_available_for_full_cascade=True,GT_SS_is_for_scoring_only=True,
        artifacts=[dict(sample_id=r['sample_id'],seed=r['seed'],target_voxels_SHA256=r['target_voxels_sha256'],
                        occupancy_counts={k:r[k] for k in ('tp','fp','fn')}) for r in record['per_asset']])
    return binding, record, deepcopy(record), cfg


@pytest.mark.parametrize('arm', VARIANTS)
def test_own_physical_family_is_admitted_without_parent_alias(arm):
    binding, record, ref, _ = fixture(arm)
    assert contract.validate_archive128(record,arm,ref,binding=binding) == list(map(str,range(128)))


@pytest.mark.parametrize('arm', VARIANTS)
def test_binding_factory_uses_validated_bundle_and_keeps_own_name(monkeypatch,arm):
    binding, _, _, cfg = fixture(arm)
    bundle = dict(step=480,**{k:binding[k]['path'] for k in
        ('native_config','checkpoint','conditioner_checkpoint')})
    seen = []
    def validate(actual, actualcfg):
        assert actual is bundle and actualcfg is cfg
        seen.append(True)
        return arm
    monkeypatch.setattr(contract,'validate_bundle',validate)
    hashes = {binding[k]['path']:binding[k]['SHA256'] for k in
        ('native_config','checkpoint','conditioner_checkpoint')}
    assert contract.model_binding(bundle,cfg,hashes) == binding
    assert seen == [True]


@pytest.mark.parametrize('arm', VARIANTS)
@pytest.mark.parametrize('damage',['parent-alias','source-count','physical-R','head','foreign-checkpoint','hash','missing-source','float-step'])
def test_wrong_model_cannot_be_relabelled_as_a_control(arm,damage):
    b, _, _, _ = fixture(arm)
    if damage == 'parent-alias': b['variant'] = VARIANTS[arm][0]
    elif damage == 'source-count': b['source_tensors'] += 1
    elif damage == 'physical-R': b['retargeter_present'] = not b['retargeter_present']
    elif damage == 'head': b['learned_attention_head_tensors'] = 19
    elif damage == 'foreign-checkpoint': b['checkpoint']['path'] = '/other/ckpts/denoiser_step0000480.pt'
    elif damage == 'hash': b['checkpoint']['SHA256'] = 'unknown'
    elif damage == 'missing-source': b['conditioner_checkpoint'] = None
    else: b['step'] = 480.
    with pytest.raises(ValueError): contract.validate_ss_binding(b,arm)


@pytest.mark.parametrize('damage',['source-hash','target-depth','sampler','camera','prior-control','training-seed','foreign-weight'])
def test_evaluation_must_bind_real_own_weights_and_native_protocol(damage):
    b,r,_,_ = fixture()
    if damage == 'source-hash': r['ablation_source_SHA256'] = {'/source/control.py':'d'*64}
    elif damage == 'target-depth': r['target_modality'] = 'depth'
    elif damage == 'sampler': r['sampling_steps'] = 24
    elif damage == 'camera': r['input_camera_ids'] = [0,1,2,3]
    elif damage == 'prior-control': r['ablation_control'] = 'without-logit-prior'
    elif damage == 'training-seed': r['training_seed'] = 202
    else: r['verified_inputs'][r['checkpoint']] = 'd'*64
    with pytest.raises(ValueError): contract.validate_ss_evaluation(r,b,'A_FIXED_UNIFORM')


@pytest.mark.parametrize('damage',['suffix-target','suffix-noise','suffix-count','duplicate','missing','GT-generation','parent-archive'])
def test_all128_archived_predictions_are_checked(damage):
    b,r,ref,_ = fixture()
    if damage == 'suffix-target': r['per_asset'][127]['target_voxels_sha256'] = 'd'*64
    elif damage == 'suffix-noise': r['per_asset'][127]['seed'] += 1
    elif damage == 'suffix-count': r['native_SS_artifacts']['artifacts'][127]['occupancy_counts']['tp'] += 1
    elif damage == 'duplicate': r['per_asset'][127]['sample_id'] = '0'
    elif damage == 'missing': r['per_asset'].pop()
    elif damage == 'GT-generation': r['native_SS_artifacts']['GT_SS_is_for_scoring_only'] = False
    else: r['native_SS_artifacts']['status'] = 'PASS_ACTUAL_ORIGINAL_CONSISTENT_EVIDENCE_SS128_CAPTURE'
    with pytest.raises(ValueError): contract.validate_archive128(r,'A_FIXED_UNIFORM',ref,binding=b)
