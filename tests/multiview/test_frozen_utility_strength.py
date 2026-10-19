"""Keep the stronger utility study isolated from the reference model."""
import copy
import hashlib
import json

import pytest

from experiments.deformnext_retargeting.frozen_matcher_utility import validate_profile as validate_reference
from experiments.deformnext_retargeting.frozen_utility_strength import NAME, jobs, recipe, validate_profile


def source(tmp_path):
    return {'ckpt': 'none', 'load_dir': '', 'training_seed': 101, 'num_gpus': 2,
            'default_output_dir': str(tmp_path / 'runs_ss64/frozen-utility-c1'),
            'experiment': {'arm': 'frozen-utility-c1', 'required_microbatch_per_gpu': 48},
            'trainer': {'args': {'max_steps': 480, 'batch_size_per_gpu': 48, 'batch_split': 1,
                'deform_conditioner': {'implementation': 'r9-native-v5', 'arm': 'c1', 'score_temperature': .005,
                    'task_attention': {'decoder': True, 'score_prior': True, 'train_decoder': False,
                        'train_score': False, 'content_mode': 'residual-zero', 'chunk_size': 64}}}}}


def test_recipe_preserves_all_optimization_and_matching_settings_except_initial_scale(tmp_path):
    original = source(tmp_path)
    before = copy.deepcopy(original)
    cfg = recipe(original, tmp_path)
    assert original == before
    validate_profile(cfg, tmp_path)
    assert cfg['experiment']['arm'] == NAME
    assert cfg['output_dir'] == cfg['default_output_dir'] == str(tmp_path / 'runs_ss64' / NAME)
    assert cfg['config'] == cfg['experiment']['recipe_path'] == str(tmp_path / 'configs' / (NAME + '.json'))
    reference_trainer = copy.deepcopy(cfg['trainer'])
    assert reference_trainer['args']['deform_conditioner']['task_attention'].pop('initial_content_scale') == 16.
    assert reference_trainer == original['trainer']
    with pytest.raises(ValueError):
        validate_reference(cfg, tmp_path)


def test_jobs_bind_fresh_training_raw480_gpuqa_and_all_actual_epoch_closures(tmp_path):
    tasks = jobs(tmp_path, tmp_path.parent, tmp_path / 'previous', ['2', '3'], allowed_gpus=['0', '1', '2', '3'])
    assert len(tasks) == 6
    assert tasks[0]['gpus'] == ['2', '3']
    assert tasks[0]['name'] == 'train-' + NAME
    assert 'experiments.deformnext_retargeting.frozen_utility_strength' in tasks[0]['command']
    assert [task['gpu'] for task in tasks[1:4]] == ['2', '3', '2']
    assert tasks[4]['name'] == 'functional-QA-s101-' + NAME
    assert len(tasks[5]['requires']) == 6
    assert 'epoch05/file4-closure.json' in tasks[5]['requires'][4]
    assert 'final-weight-audits-v1' in tasks[5]['requires'][5]
    assert all('denoiser_step0000480.pt' in ' '.join(task['command']) for task in tasks[1:5])
    with pytest.raises(ValueError):
        jobs(tmp_path, tmp_path.parent, tmp_path / 'previous', ['6', '7'], allowed_gpus=['0', '1', '2', '3'])


@pytest.mark.parametrize('mutation', ['scale', 'temperature', 'decoder', 'eps', 'seed', 'microbatch'])
def test_strength_profile_rejects_additional_uncontrolled_changes(tmp_path, mutation):
    cfg = recipe(source(tmp_path), tmp_path)
    args = cfg['trainer']['args']
    if mutation == 'scale':
        args['deform_conditioner']['task_attention']['initial_content_scale'] = 4.
    elif mutation == 'temperature':
        args['deform_conditioner']['score_temperature'] = .05
    elif mutation == 'decoder':
        args['deform_conditioner']['task_attention']['train_decoder'] = True
    elif mutation == 'eps':
        args['task_attention_optimizer_eps'] = 1e-10
    elif mutation == 'seed':
        cfg['training_seed'] = 202
    else:
        args['batch_split'] = 2
    with pytest.raises(ValueError):
        validate_profile(cfg, tmp_path)


@pytest.mark.parametrize('custom_profile', [False, True])
def test_summary_keeps_candidate_identity_when_iterating_comparison_models(tmp_path, monkeypatch, custom_profile):
    """Synthetic routing check; no model-quality claims or real run files."""
    from experiments.deformnext_retargeting import summarize_frozen_utility_followup as module
    campaign = tmp_path / 'campaign'
    name = NAME if custom_profile else module.NAME
    namespace = 'utility-strength-followup-v1' if custom_profile else 'frozen-utility-followup-v1'
    root = campaign / namespace
    data = {}
    data[root / 'configs' / (name + '.json')] = {'training_seed': 101, 'ckpt': 'none'}
    data[campaign / 'seven-arm-final-report.json'] = {'training_audits': {'static-c1': {'initial_iou': .4}}}
    data[campaign / 'candidate-selection-v1.json'] = {'status': 'NO_PROMISING_COMBINATION_YET', 'selected_candidate': None}
    sources = [('v1', campaign.parent / 'ss-seed-confirmation-v1', 'v1'),
               ('previous-c1', campaign.parent / 'ss-seed-confirmation-v1', 'c1')]
    sources += [(arm, campaign, arm) for arm in ['static-top1-c1', 'top1-reference', 'task-full', 'task-native-spatial']]
    extra = {'frozen-utility-c1': (campaign / 'frozen-utility-followup-v1', 'frozen-utility-c1')} if custom_profile else {}
    sources += [(label, base, arm) for label, (base, arm) in extra.items()]
    for label, base, arm in [(name, root, name), *sources]:
        for noise in range(3):
            data[base / 'evaluations' / arm / f'noise-{noise}/evaluation.json'] = {'label': label, 'noise': noise}
    folders = [root / 'final-weight-audits-v1' / ('train-' + name)]
    folders += [root / 'epoch-audits-v1' / ('train-' + name) / f'epoch{e:02d}' for e in range(1, 6)]
    for folder in folders:
        audit = folder / 'audit.json'
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text('{}')
        data[audit] = {}
        data[folder / 'file4-closure.json'] = {'status': 'PASS', 'local_equals_actual_file4_SHA256': True,
            'files': [{'relative_path': str(audit.relative_to(campaign.parent)),
                       'sha256': hashlib.sha256(audit.read_bytes()).hexdigest(), 'bytes': 2}]}
    data[root / 'functional-QA-v1' / ('s101-' + name) / 'summary.json'] = {}
    for path, value in data.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(value))
    monkeypatch.setattr(module, 'read', lambda p: data[p])
    monkeypatch.setattr(module, 'audit_training', lambda *a: {})
    monkeypatch.setattr(module, 'audit_final_replay', lambda training, evaluation: evaluation['label'])
    monkeypatch.setattr(module, 'task_gradient_audit', lambda *a: {})
    monkeypatch.setattr(module, 'validate_saved_audits', lambda *a: None)
    monkeypatch.setattr(module, 'validate_evaluation', lambda *a: None)
    monkeypatch.setattr(module, 'zero_records', lambda *a: ({}, []))
    monkeypatch.setattr(module, 'MODES', {})
    seen = []

    def compare(pairs):
        assert len(pairs) == 3
        assert all(candidate['label'] == name for _, _, candidate in pairs)
        seen.append(pairs[0][1]['label'])
        return {'gain': 0., 'paired_cluster_bootstrap_ci95': [-.1, .1]}

    monkeypatch.setattr(module, 'compare', compare)
    monkeypatch.setattr(module, 'supported', lambda *a: False)
    monkeypatch.setattr(module, 'aggregate', lambda *a: {'micro_voxel_iou': .4, 'micro_voxel_f1': .5})
    module.summarize(root, campaign, name=name, namespace=namespace,
                     profile_validator=lambda *a: None, extra_sources=extra)
    result = json.loads((root / 'fixed-final-decision-v1.json').read_text())
    assert result['candidate'] == name
    assert result['training_audit']['final128_replays_epoch32'] == name
    assert set(seen) == {label for label, _, _ in sources}
    assert not result['model_adopted'] and not result['eligible_for_training_seed_confirmation']
