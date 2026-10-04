import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


@pytest.fixture
def catalog(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    for i in range(4):
        (inputs / f'{i:03}.png').write_bytes(b'input-' + bytes([i]))
    transforms = inputs / 'transforms.json'
    transforms.write_text(json.dumps({'frames': [{'file_path': f'{i:03}.png'} for i in range(4)]}))
    results = []
    for name, count, steps in [('sv', 1, 0), ('mv', 4, 0), ('ft', 4, 10)]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'model.glb').write_bytes(b'glTF-' + name.encode())
        (folder / 'base_color_views.png').write_bytes(b'color-' + name.encode())
        (folder / 'normal_views.png').write_bytes(b'normal-' + name.encode())
        (folder / 'sample_manifest.json').write_text(json.dumps({
            'complete': True, 'asset': 'same-asset', 'input_manifest': str(transforms),
            'view_indices': list(range(count)), 'seed': 42, 'actual_resolution': 1024,
            'pipeline_type': '1024_cascade', 'sampler_steps': 12,
            'mesh_scale': 0.88, 'checkpoints': {'ss64': '/weights/structure.pt'},
        }))
        results.append({'id': name, 'label': name.upper(), 'directory': str(folder),
                        'glb': 'model.glb', 'training_steps': steps})
    config = tmp_path / 'cases.json'
    config.write_text(json.dumps({'cases': [{'id': 'bear', 'title': 'Bear', 'results': results}]}))
    return config


@pytest.fixture
def server_module():
    try:
        from result_viewer import server
    except ImportError:
        pytest.fail('The result viewer HTTP server has not been implemented.')
    return server


@pytest.fixture
def http_server(server_module, catalog):
    server = server_module.create_server(catalog, host='127.0.0.1', port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_address[1]}'
    server.shutdown()
    server.server_close()
    thread.join()


def test_catalog_preserves_each_results_actual_input_views(http_server):
    with urlopen(http_server + '/api/cases') as response:
        payload = json.load(response)
    case = payload['cases'][0]
    assert case['asset'] == 'same-asset'
    assert [len(result['inputs']) for result in case['results']] == [1, 4, 4]
    assert [result['training_steps'] for result in case['results']] == [0, 0, 10]
    assert [image['view_index'] for image in case['results'][1]['inputs']] == [0, 1, 2, 3]
    assert case['results'][0]['seed'] == 42
    assert case['results'][2]['resolution'] == 1024


def test_glb_and_input_urls_serve_the_corresponding_files(http_server):
    for route, expected, mime in [
        ('/files/bear/sv/model.glb', b'glTF-sv', 'model/gltf-binary'),
        ('/files/bear/mv/base_color.png', b'color-mv', 'image/png'),
        ('/files/bear/ft/normal.png', b'normal-ft', 'image/png'),
        ('/files/bear/mv/input/3.png', b'input-\x03', 'image/png'),
    ]:
        with urlopen(http_server + route) as response:
            assert response.read() == expected
            assert response.headers['Content-Type'] == mime
        with urlopen(Request(http_server + route, method='HEAD')) as response:
            assert int(response.headers['Content-Length']) == len(expected)
            assert response.read() == b''


def test_unregistered_files_are_not_served(http_server):
    for route in ['/files/bear/sv/generated_latents.pt', '/files/bear/mv/input/7.png',
                  '/../../etc/passwd', '/server.py']:
        with pytest.raises(HTTPError) as error:
            urlopen(http_server + route)
        assert error.value.code == 404


def test_mismatched_asset_cannot_be_presented_as_same_asset_comparison(server_module, catalog):
    manifest = catalog.parent / 'mv/sample_manifest.json'
    payload = json.loads(manifest.read_text())
    payload['asset'] = 'a-different-asset'
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='asset'):
        server_module.create_server(catalog, port=0)


def test_unfinished_result_cannot_be_presented_as_complete(server_module, catalog):
    manifest = catalog.parent / 'ft/sample_manifest.json'
    payload = json.loads(manifest.read_text())
    payload['complete'] = False
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='complete'):
        server_module.create_server(catalog, port=0)


def test_nonzero_anchor_keeps_original_view_numbers_and_image_order(server_module, catalog):
    # A front-first result must not be presented as anchored on original view 0.
    for name, indices in [('sv', [3]), ('mv', [3, 0, 2, 1]), ('ft', [3, 0, 2, 1])]:
        path = catalog.parent / name / 'sample_manifest.json'
        manifest = json.loads(path.read_text())
        manifest['view_indices'] = indices
        manifest['anchor_view'] = 3
        path.write_text(json.dumps(manifest))
    payload, files = server_module.load_catalog(catalog)
    results = payload['cases'][0]['results']
    assert [result['anchor_view'] for result in results] == [3, 3, 3]
    assert [item['view_index'] for item in results[1]['inputs']] == [3, 0, 2, 1]
    assert files[results[0]['inputs'][0]['url']].read_bytes() == b'input-\x03'


def test_relative_catalog_and_input_manifest_survive_moved_cwd(server_module, catalog, tmp_path, monkeypatch):
    payload = json.loads(catalog.read_text())
    for result in payload['cases'][0]['results']:
        result['directory'] = result['id']
        path = catalog.parent / result['directory'] / 'sample_manifest.json'
        manifest = json.loads(path.read_text())
        manifest['input_manifest'] = '../inputs/transforms.json'
        path.write_text(json.dumps(manifest))
    catalog.write_text(json.dumps(payload))
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    server = server_module.create_server(catalog, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = f'http://127.0.0.1:{server.server_address[1]}'
        with urlopen(origin + '/api/cases') as response:
            metadata = json.load(response)
        assert metadata['cases'][0]['results'][2]['training_steps'] == 10
        for route, expected in [('/files/bear/sv/model.glb', b'glTF-sv'),
                                ('/files/bear/mv/input/3.png', b'input-\x03')]:
            with urlopen(origin + route) as response:
                assert response.read() == expected
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_same_asset_results_serve_their_own_input_revision(server_module, catalog):
    # Same view number and asset ID can refer to different prepared revisions.
    revised = catalog.parent / 'revised-inputs'
    revised.mkdir()
    (revised / '000.png').write_bytes(b'revised-view-zero')
    transforms = revised / 'transforms.json'
    transforms.write_text(json.dumps({'frames': [{'file_path': '000.png'}]}))
    manifest_path = catalog.parent / 'ft/sample_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['input_manifest'] = str(transforms)
    manifest['view_indices'] = [0]
    manifest_path.write_text(json.dumps(manifest))
    server = server_module.create_server(catalog, host='127.0.0.1', port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = f'http://127.0.0.1:{server.server_address[1]}'
        with urlopen(origin + '/api/cases') as response:
            case = json.load(response)['cases'][0]
        assert case['asset'] == 'same-asset'
        for result, expected in zip(case['results'], [b'input-\x00', b'input-\x00', b'revised-view-zero']):
            assert result['inputs'][0]['view_index'] == 0
            with urlopen(origin + result['inputs'][0]['url']) as response:
                assert response.read() == expected
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
