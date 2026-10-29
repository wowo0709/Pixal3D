"""Convert source WEBP RGBA views losslessly to PNG for existing Pixal3D loaders."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    inputs = sorted(a.input.glob('*/source/mv/*.webp'))
    if len(inputs) != 598:
        raise ValueError('expected exactly 598 source WEBP files')
    records = []
    for path in inputs:
        relative = path.relative_to(a.input).with_suffix('.png')
        out = a.output / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as source:
            image = source.convert('RGBA')
            image.save(out, format='PNG')
        with Image.open(out) as restored:
            if image.tobytes() != restored.convert('RGBA').tobytes():
                raise ValueError('PNG differs from source RGBA: ' + str(path))
        records.append({'source': str(path.relative_to(a.input)), 'source_sha256': sha(path),
                        'png': str(relative), 'png_sha256': sha(out)})
    (a.output / 'conversion-manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'converted': len(records)}))


if __name__ == '__main__':
    main()
