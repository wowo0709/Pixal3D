"""Closed artifact and checkpoint-data integrity, without neural forwards."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import torch

from experiments.deformnext_retargeting.cascade_view_roles import ClosedInputs, file_sha256 as sha
from experiments.deformnext_retargeting.collect_module_seed_confirmation import (
    SCHEMA, collect, frozen_base_comparison, read_file, required_sources, verify_terminal,
)
from experiments.deformnext_retargeting.unified_population_contract import bundle_validator
from test_module_training_seed_summary import inputs as statistic_inputs


def files(tmp_path):
    data = statistic_inputs(tmp_path, 'C_FIXED_TOPK')
    results, configs, traces, starts = data
    base = tmp_path/'canonical'
    base.mkdir()
    receipt = base/'file4-closure.json'
    original_path = base/'original-denoiser_step0004210.pt'
    original = dict(base=torch.tensor([1., 2.]), **{'lora_'+str(i): torch.zeros(1) for i in range(360)})
    torch.save(original, original_path)
    models = {}

    def save(path, value, *, lines=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps(r) for r in value)+'\n' if lines else json.dumps(value))

    def spec(path):
        return dict(path=str(path), SHA256=sha(path), receipt=str(receipt))

    for seed in (101, 202):
        models[str(seed)] = {}
        for arm, cfg in configs[seed].items():
            root = base/str(seed)/arm/'native'
            cfg['output_dir'] = str(root)
            cfg['trainer']['args']['continuation_ckpt']['denoiser'] = str(original_path)
            if arm != 'V1':
                ref_path = Path(cfg['experiment']['reference_science_config'])
                ref = json.loads(ref_path.read_text())
                ref['trainer']['args']['continuation_ckpt']['denoiser'] = str(original_path)
                ref_path.write_text(json.dumps(ref))
                cfg['experiment']['reference_science_config_SHA256'] = sha(ref_path)
            save(root/'config.json', cfg)
            nn_path = root/'ckpts/denoiser_step0000480.pt'
            nn_path.parent.mkdir()
            state = deepcopy(original)
            state['lora_0'].add_(.2)
            torch.save(state, nn_path)
            source = None if arm == 'V1' else root/'ckpts/deform_conditioner_step0000480.pt'
            if source is not None:
                torch.save(dict(source=torch.ones(1)), source)
            schema = ('cold-native-task-Top1-SS-checkpoint-pair-v1' if arm == 'C_DECODER'
                      else bundle_validator(arm).BUNDLE_SCHEMA)
            bundle = dict(schema=schema, variant='vanilla' if arm=='V1' else 'R_top1_only' if arm=='C_DECODER' else arm,
                step=480, native_config=str(root/'config.json'), checkpoint=str(nn_path),
                conditioner_checkpoint=None if source is None else str(source), source_teacher_updates=0,
                native_task_objective='SS_flow_mse', native_validation=dict(step=480))
            save(root/'bundle.json', bundle)
            save(root/'startup.json', starts[seed][arm])
            save(root/'training-noise-trace.jsonl', traces[seed][arm], lines=True)
            terminal = dict(PID=2147483000, birth=1., returncode=0)
            save(root/'training-terminal.json', terminal)
            evaluation_specs = []
            for noise, value in enumerate(results[seed][arm]):
                value.update(checkpoint=str(nn_path), arm_config=str(root/'config.json'),
                    conditioner_checkpoint=None if source is None else str(source),
                    native_task_checkpoint_bundle=str(root/'bundle.json'),
                    verified_inputs={str(p):sha(p) for p in [nn_path,root/'config.json']+([] if source is None else [source])})
                folder = root/f'noise{noise}'
                save(folder/'evaluation.json', value)
                save(folder/'terminal.json', terminal)
                evaluation_specs.append(dict(evaluation=spec(folder/'evaluation.json'), terminal=spec(folder/'terminal.json')))
            models[str(seed)][arm] = dict(config=spec(root/'config.json'), bundle=spec(root/'bundle.json'),
                checkpoint=spec(nn_path), conditioner_checkpoint=None if source is None else spec(source),
                startup=spec(root/'startup.json'), training_trace=spec(root/'training-noise-trace.jsonl'),
                training_terminal=spec(root/'training-terminal.json'), evaluations=evaluation_specs)
    entries = [dict(relative_path=str(p.relative_to(base)), sha256=sha(p), bytes=p.stat().st_size)
               for p in base.rglob('*') if p.is_file()]
    receipt.write_text(json.dumps(dict(status='PASS',local_equals_actual_file4_SHA256=True,files=entries)))
    plan = dict(schema=SCHEMA, selected_arm='C_FIXED_TOPK', models=models,
                original_checkpoint=dict(path=str(original_path),SHA256=sha(original_path)))
    return plan, ClosedInputs(base)


def test_actual_file_reads_tensor_data_and_paired_statistics_are_joined(tmp_path):
    plan, check = files(tmp_path)
    result = collect(plan, check)
    assert result['status'] == 'PASS_ACTUAL_CLOSED_SELECTED_MODULE_TWO_SEED_DEV128X3'
    assert result['statistics']['training_seeds'] == [101, 202]
    assert result['statistics']['measured_models'] == ['V1', 'C_DECODER', 'C_FIXED_TOPK']
    assert result['actual_frozen_base_and_model_bytes_verified']
    assert result['CPU_neural_forward_or_training_updates'] == 0
    assert result['candidate_selection_not_proven_here'] and not result['full_goal_complete']


@pytest.mark.parametrize('damage', ['missing-seed', 'checkpoint-bytes', 'config-bytes',
    'claimed-SHA', 'other-source', 'missing-receipt', 'original-bytes', 'failed-evaluation'])
def test_corrupt_or_incomplete_artifacts_cannot_become_replication_evidence(tmp_path, damage):
    plan, check = files(tmp_path)
    row = plan['models']['202']['C_FIXED_TOPK']
    if damage == 'missing-seed':plan['models'].pop('202')
    elif damage in ('checkpoint-bytes', 'config-bytes'):
        Path(row['checkpoint' if damage=='checkpoint-bytes' else 'config']['path']).write_bytes(b'changed')
    elif damage == 'claimed-SHA':row['checkpoint']['SHA256'] = 'a'*64
    elif damage == 'other-source':row['conditioner_checkpoint'] = plan['models']['202']['C_DECODER']['conditioner_checkpoint']
    elif damage == 'missing-receipt':row['startup']['receipt'] = str(tmp_path/'missing.json')
    elif damage == 'original-bytes':Path(plan['original_checkpoint']['path']).write_bytes(b'changed')
    else:row['evaluations'][0]['terminal'] = row['evaluations'][1]['evaluation']
    with pytest.raises((ValueError, FileNotFoundError)):
        collect(plan, check)


def test_frozen_base_may_not_change_even_when_shapes_and_dtype_match():
    original = dict(base=torch.ones(2), lora_A=torch.zeros(2))
    final = deepcopy(original)
    final['lora_A'].fill_(.5)
    assert frozen_base_comparison(original, final)['actual_original_frozen_NN_base_tensors_exact']
    final['base'][0] += .001
    with pytest.raises(ValueError, match='frozen original'):
        frozen_base_comparison(original, final)


@pytest.mark.parametrize('damage', ['dtype', 'keys', 'shape', 'not-tensor', 'no-LoRA'])
def test_checkpoint_structure_cannot_be_silently_reinterpreted(damage):
    original = dict(base=torch.ones(2), lora_A=torch.zeros(2))
    final = deepcopy(original)
    if damage == 'dtype':final['base'] = final['base'].half()
    elif damage == 'keys':final['new'] = torch.ones(1)
    elif damage == 'shape':final['base'] = final['base'][:1]
    elif damage == 'not-tensor':final['base'] = [1.,1.]
    else:original.pop('lora_A');final.pop('lora_A')
    with pytest.raises(ValueError):frozen_base_comparison(original, final)


def test_legacy_terminal_reports_only_the_process_absence_it_can_actually_prove():
    value = verify_terminal(dict(PID=2147483000,birth=1.,returncode=0))
    assert value['actual_child_absent'] and value['legacy_session_absence_not_inferred_from_child_PID']
    assert value['actual_recorded_session_absent'] is None
    import psutil
    with pytest.raises(ValueError):
        verify_terminal(dict(PID=os.getpid(),birth=psutil.Process().create_time(),returncode=0))
    with pytest.raises(ValueError):verify_terminal(dict(PID=2147483000,birth=1.,returncode=1))
    with pytest.raises(ValueError):verify_terminal(dict(PID=2147483000,birth=1.,returncode=0,actual_session_drained=False))


def test_an_explicit_recorded_session_must_actually_be_empty():
    with pytest.raises(ValueError, match='recorded owned session'):
        verify_terminal(dict(PID=2147483000,birth=1.,returncode=0,session_ID=os.getsid(os.getpid())))


def test_collector_pins_the_original_numerical_family_as_well_as_its_reducer():
    paths = required_sources('C_FIXED_TOPK')
    names = {Path(p).stem for p in paths}
    assert {'collect_module_seed_confirmation', 'module_training_seed_summary',
            'r_top1_native_recipe', 'projected_native_runtime'} <= names
    assert all(Path(p).is_file() for p in paths)
