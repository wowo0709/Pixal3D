"""Publication scope and shared SDK routing; no network or CUDA calls."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from experiments.deformnext_retargeting import frozen_utility_wandb as follow
from experiments.deformnext_retargeting import semantic_topk_campaign_wandb as common
from experiments.deformnext_retargeting.firstwave_wandb import sha


def test_development_reporting_does_not_wait_for_or_claim_seed_confirmation(tmp_path):
    jobs = follow.publication_jobs(tmp_path)
    assert len(jobs) == 3 and all(job['resource'] == 'cpu' for job in jobs)
    assert jobs[0]['depends_on'] == ['close-frozen-utility-fixed-final-report',
                                    'summarize-frozen-utility-value-address']
    assert all('gpu' not in job and 'gpus' not in job for job in jobs)
    assert jobs[1]['depends_on'] == [jobs[0]['name']]
    assert jobs[2]['depends_on'] == [jobs[1]['name']]
    assert all(Path(output).is_relative_to(tmp_path) for job in jobs for output in job['outputs'])


def test_missing_actual_final_decision_cannot_become_a_publication(tmp_path):
    with pytest.raises(FileNotFoundError):
        follow.payload(tmp_path)
    assert not (tmp_path / 'development-wandb-v1').exists()


@pytest.mark.parametrize('model_binary', [False, True])
def test_payload_separates_final_metrics_from_common32_diagnostics(tmp_path, monkeypatch, model_binary):
    root = tmp_path / 'campaign/followup'
    folder = root / 'development-wandb-v1/presentation'
    folder.mkdir(parents=True)
    source = tmp_path / ('source.pt' if model_binary else 'source.json')
    source.write_text('synthetic fixture')
    contrast = {'gain': .001, 'paired_cluster_bootstrap_ci95': [-.001, .003]}
    scope = {'full_goal_complete': False, 'model_adopted': False, 'seed_confirmation_completed': False}
    data = {'report': {'models': {name: {'micro_voxel_iou': .4, 'micro_voxel_f1': .57}
                                  for name in follow.ORDER},
                       'contrasts': {'static-top1-c1': contrast}},
            'diagnostic': {'normal_minus_wrong_address': contrast},
            'curves': [[follow.NAME, epoch, .4, .57] for epoch in range(6)],
            'scope': scope, 'files': [source]}
    (folder / 'summary.json').write_text(json.dumps({
        'status': 'PASS_COMPLETED_CAMPAIGN_PRESENTATION', 'scope': scope,
        'input_SHA256': {str(source): sha(source)}}))
    monkeypatch.setattr(follow, 'collect', lambda root: data)
    if model_binary:
        with pytest.raises(ValueError, match='model binaries'):
            follow.payload(root)
        return
    spec = follow.payload(root)
    assert spec['metrics']['development128/frozen-utility-c1/micro_voxel_iou'] == .4
    assert spec['metrics']['development128/utility-vs-static-top1-c1/ci95_lower'] == -.001
    assert spec['metrics']['diagnostic_common32/value-address/ci95_lower'] == -.001
    assert len(spec['tables']['development128/models']['data']) == 9
    assert len(spec['tables']['common32/epoch_curves']['data']) == 6
    assert spec['scope'] == scope
    assert not any(Path(path).suffix == '.pt' for path in spec['artifact_files'].values())


@pytest.mark.parametrize('custom', [False, True])
def test_shared_publish_and_fresh_readback_keep_default_and_explicit_run_identity(tmp_path, monkeypatch, custom):
    scope = {'full_goal_complete': False, 'full_campaign_complete': False, 'model_adopted': False}
    spec = {'source_commit': 'synthetic-test', 'scope': scope, 'metrics': {'actual_test_IoU': .4},
            'tables': {}, 'images': {}, 'artifact_files': {}}
    expected_id = follow.RUN_ID if custom else common.RUN_ID
    folder = tmp_path / ('development-wandb-v1' if custom else 'campaign-wandb-v1')
    calls = {}
    actual = SimpleNamespace(state='finished', url='mock://report', summary={
        'status': 'PASS_COMPLETED_SS_CAMPAIGN_PUBLICATION', 'model_weights_uploaded': False,
        **scope, **spec['metrics']})

    class Run:
        url = actual.url

        def __init__(self):
            self.summary = {}

        def log(self, value):
            calls.setdefault('logged', []).append(value)

        def log_artifact(self, artifact):
            calls['logged_artifact'] = artifact.name
            return SimpleNamespace(wait=lambda: None, version='v0')

        def finish(self):
            calls['finished'] = True

    def init(**kwargs):
        calls['init'] = kwargs
        return Run()

    class Artifact:
        def __init__(self, name, **kwargs):
            self.name = name

    class API:
        def __init__(self, **kwargs):
            pass

        def flush(self):
            pass

        def run(self, path):
            calls['readback_run'] = path
            return actual

        def artifact(self, path):
            calls['readback_artifact'] = path
            def download(root):
                Path(root).mkdir(parents=True)
                return root
            return SimpleNamespace(download=download)

    sdk = SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs, Artifact=Artifact, Api=API)
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.setenv('WANDB_CACHE_DIR', str(tmp_path / 'initial-cache'))
    monkeypatch.setattr(common, 'payload', lambda root: spec)
    if custom:
        common.publish(tmp_path, payload_factory=lambda root: spec, output_folder=folder,
                       run_id=expected_id, run_name='Synthetic frozen utility fixture')
        common.readback(tmp_path, output_folder=folder, run_id=expected_id)
    else:
        common.publish(tmp_path)
        common.readback(tmp_path)
    assert calls['init']['id'] == calls['logged_artifact'] == expected_id
    assert calls['init']['dir'] == str(folder) and calls['init']['resume'] == 'never'
    assert calls['readback_run'] == common.PROJECT + '/' + expected_id
    assert calls['readback_artifact'] == common.PROJECT + '/' + expected_id + ':v0'
    audit = json.loads((folder / 'remote-readback/audit.json').read_text())
    assert audit['scope'] == scope and audit['fresh_download_cache'] is True
    assert calls['finished']


def test_shared_readback_rejects_another_runs_receipt_before_api_calls(tmp_path):
    folder = tmp_path / 'report'
    folder.mkdir()
    (folder / 'publication.json').write_text(json.dumps({'run_id': common.RUN_ID}))
    (folder / 'publication-spec.json').write_text('{}')
    with pytest.raises(ValueError, match='run identity'):
        common.readback(tmp_path, output_folder=folder, run_id=follow.RUN_ID)
