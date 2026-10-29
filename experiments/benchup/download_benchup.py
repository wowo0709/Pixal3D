"""Download the immutable public BenchUp revision into a user-specified dataset directory."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, time, urllib.parse, urllib.request
from pathlib import Path

REPO = 'Inbar2344/BenchUp'
REVISION = 'a8b87439281c7f64788231f0935b6d6aedc3ba85'


def fetch(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    meta = json.loads(fetch('https://huggingface.co/api/datasets/' + REPO + '/revision/' + REVISION + '?blobs=true'))
    if meta['sha'] != REVISION:
        raise ValueError('unexpected repository revision')
    files = {x['rfilename']: x for x in meta['siblings'] if x['rfilename'] != '.gitattributes'}
    paths = sorted(files)
    if len(paths) != 753:
        raise ValueError('unexpected release file count')
    args.output.mkdir(parents=True, exist_ok=True)

    def download(path):
        dest = args.output / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file():
            url = 'https://huggingface.co/datasets/' + REPO + '/resolve/' + REVISION + '/' + urllib.parse.quote(path)
            for attempt in range(4):
                try:
                    data = fetch(url)
                    tmp = dest.with_name(dest.name + '.download-' + str(os.getpid()))
                    tmp.write_bytes(data)
                    tmp.replace(dest)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        data = dest.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        expected = files[path].get('lfs', {})
        if actual != expected.get('sha256') or len(data) != expected.get('size'):
            raise ValueError('release checksum mismatch: ' + path)
        return {'path': path, 'size': len(data), 'sha256': actual}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = []
        for i, rec in enumerate(pool.map(download, paths), 1):
            records.append(rec)
            if i % 25 == 0 or i == len(paths):
                print(json.dumps({'downloaded': i, 'total': len(paths)}), flush=True)
    out = {'repo': REPO, 'revision': REVISION, 'files': records}
    (args.output / 'download-manifest.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'files': len(records), 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
