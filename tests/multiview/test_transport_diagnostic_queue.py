from experiments.deformnext_retargeting.queue_transport_diagnostics import build_jobs


def test_saved_weight_diagnostics_preserve_main_training_and_pair_protocol(tmp_path):
    root, work = tmp_path / 'results', tmp_path / 'runtime'
    after = 'train-native-module-c1-retarget-r'
    jobs, specifications = build_jobs(root, work, ['2', '3'], after)
    generations = [j for j in jobs if j['resource'] == 'evaluation']
    comparisons = [j for j in jobs if j['resource'] == 'cpu']
    assert len(generations) == 11 and len(comparisons) == 8
    assert len({j['name'] for j in jobs}) == len(jobs)
    assert {j['gpu'] for j in generations} == {'2', '3'}
    assert all(after in j['depends_on'] for j in generations)
    assert all('/evaluations/diagnostics/' in p for j in generations for p in j['outputs'])
    for j in generations:
        command = j['command']
        assert command[command.index('--step') + 1] == '480'
        assert command[command.index('--generation-batch') + 1] == '8'
        assert command[command.index('--panel-file') + 1] == str(work / 'panels/panel-32.json')
        assert command[command.index('--seed-map-file') + 1] == str(work / 'panels/seed-map.json')
        if 'b3-attention-r' in j['name']:
            assert 'adapter-off' not in command
    for arm in ['p2-matcher-grid', 'c1-retarget-r', 'b3-attention-r']:
        replicas = [j for j in generations if j['name'].startswith('diagnose32-' + arm + '-')]
        assert len({j['gpu'] for j in replicas}) == 1
        assert len({j['command'][j['command'].index('--checkpoint') + 1] for j in replicas}) == 1
    assert len(specifications) == 8
    for path, pairs in specifications.items():
        assert path.parent == work / 'comparison-inputs'
        assert len(pairs) == 1 and pairs[0]['noise'] == '0'
        assert pairs[0]['baseline'].endswith('/normal/evaluation.json')
