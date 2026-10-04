"""CPU-only HTTP viewer for explicitly registered Pixal3D result artifacts."""
import argparse
import json
import mimetypes
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent


def load_catalog(path):
    """Return browser metadata and the exact URL → local artifact mapping."""
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    files = {}
    cases = []
    for case in config['cases']:
        results = []
        asset = None
        for result in case['results']:
            folder = path.parent / result['directory']
            manifest_path = folder / 'sample_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            if not manifest['complete']:
                raise ValueError(f"Result {result['id']} is not complete")
            if asset is not None and manifest['asset'] != asset:
                raise ValueError(f"Case {case['id']} contains different asset IDs")
            asset = manifest['asset']
            input_path = folder / manifest['input_manifest']
            frames = json.loads(input_path.read_text())['frames']
            inputs = []
            for index in manifest['view_indices']:
                route = f"/files/{case['id']}/{result['id']}/input/{index}.png"
                files[route] = input_path.parent / frames[index]['file_path']
                inputs.append({'view_index': index, 'url': route})
            prefix = f"/files/{case['id']}/{result['id']}"
            outputs = {'model': (prefix + '/model.glb', folder / result['glb']),
                       'base_color': (prefix + '/base_color.png', folder / 'base_color_views.png'),
                       'normal': (prefix + '/normal.png', folder / 'normal_views.png'),
                       'manifest': (prefix + '/manifest.json', manifest_path)}
            files.update({route: file for route, file in outputs.values()})
            steps = (manifest['sampler_steps'] if 'sampler_steps' in manifest else
                     manifest['sampler_params']['sparse_structure_sampler_params']['steps'])
            results.append({'id': result['id'], 'label': result['label'],
                'training_steps': result['training_steps'], 'inputs': inputs,
                'seed': manifest['seed'], 'resolution': manifest['actual_resolution'],
                'pipeline': manifest['pipeline_type'], 'sampler_steps': steps,
                'mesh_scale': manifest['mesh_scale'],
                'anchor_view': manifest['view_indices'][0],
                'mesh_scale_source': manifest.get('mesh_scale_source', 'Preprocessed view cache'),
                'checkpoints': manifest['checkpoints'],
                'files': {kind: route for kind, (route, _) in outputs.items()},
                'download_name': result['glb']})
        cases.append({'id': case['id'], 'title': case['title'], 'asset': asset, 'results': results})
    for file in files.values():
        if not file.is_file():
            raise FileNotFoundError(file)
    return {'cases': cases}, files


def create_server(catalog_path, host='127.0.0.1', port=8765):
    payload, artifacts = load_catalog(catalog_path)
    metadata = json.dumps(payload).encode()
    static = {route: ROOT / file for route, file in {
        '/': 'index.html', '/index.html': 'index.html', '/app.js': 'app.js',
        '/styles.css': 'styles.css', '/vendor/model-viewer.min.js': 'vendor/model-viewer.min.js',
    }.items()}
    files = {**static, **artifacts}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond(send_body=True)

        def do_HEAD(self):
            self.respond(send_body=False)

        def respond(self, send_body):
            route = urlsplit(self.path).path
            if route == '/favicon.ico':
                self.send_response(204)
                self.end_headers()
                return
            if route == '/api/cases':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(metadata)))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                if send_body:
                    self.wfile.write(metadata)
                return
            file = files.get(route)
            if file is None or not file.is_file():
                self.send_error(404, 'No registered file at this URL')
                return
            mime = 'model/gltf-binary' if file.suffix == '.glb' else mimetypes.guess_type(file)[0]
            self.send_response(200)
            self.send_header('Content-Type', mime or 'application/octet-stream')
            self.send_header('Content-Length', str(file.stat().st_size))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            if send_body:
                with file.open('rb') as stream:
                    shutil.copyfileobj(stream, self.wfile)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--catalog', type=Path, default=ROOT.parent / 'results/result_viewer/cases.json')
    args = parser.parse_args()
    server = create_server(args.catalog, args.host, args.port)
    print(f'Pixal3D result viewer: http://{args.host}:{server.server_address[1]}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
