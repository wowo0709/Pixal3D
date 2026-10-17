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


def test_query_context_diagnostic_waits_for_c2_without_adding_training(tmp_path):
    jobs, specifications = build_jobs(tmp_path / 'results', tmp_path / 'runtime', ['2', '3'],
                                      'train-native-module-c2-query-r', query_only=True)
    assert len(jobs) == 3 and len(specifications) == 1
    generations = [j for j in jobs if j['resource'] == 'evaluation']
    assert len(generations) == 2
    assert {j['gpu'] for j in generations} == {'2'}
    assert {j['command'][-1] for j in generations} == {'normal', 'shuffled-deform'}
    assert all('train-native-module-c2-query-r' in j['depends_on'] for j in generations)
    assert jobs[-1]['resource'] == 'cpu'
    assert len({j['command'][j['command'].index('--checkpoint') + 1] for j in generations}) == 1


def test_cross_object_diagnostics_reuse_existing_same_device_normal_replicas(tmp_path):
    jobs, specifications = build_jobs(tmp_path / 'results', tmp_path / 'runtime', ['2', '3'],
                                      'train-native-module-c1-retarget-r', wrong_object_only=True)
    assert len(jobs) == 6 and len(specifications) == 3
    generations = [j for j in jobs if j['resource'] == 'evaluation']
    assert len(generations) == 3
    for j in generations:
        assert j['command'][-1] == 'wrong-object'
        arm = j['name'].removeprefix('diagnose32-').removesuffix('-wrong-object')
        assert f'diagnose32-{arm}-normal' in j['depends_on']
        assert j['gpu'] == ('3' if arm == 'c1-retarget-r' else '2')
    assert all('/normal/evaluation.json' in records[0]['baseline'] for records in specifications.values())
