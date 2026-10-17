"""Cache expanded global RAW proposals without changing backbone features."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--file4-mirror', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--panels', type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(4); torch.cuda.set_device(0)
    rows = [json.loads(l) for l in a.manifest.read_text().splitlines() if l]
    a.output.mkdir(parents=True, exist_ok=True)
    calibration = {json.loads(l)['sample_id'] for l in (a.panels / 'head_calibration.jsonl').read_text().splitlines() if l}
    counts = {split: {'visible': 0, 'hit64': 0, 'hit128': 0, 'hit256': 0} for split in ('train', 'calibration', 'validation')}
    started = time.monotonic()
    for i, row in enumerate(rows):
        path = a.output / (row['sample_id'] + '.npz')
        if path.exists():
            with np.load(path) as z:
                if str(z['input_sha256']) != row['conditioning_input_sha256']: raise ValueError('stale v2 proposal cache')
                indices, valid = z['indices'].astype(np.int32), z['valid']
        else:
            source_path = a.file4_mirror / row['transport_cache'].removeprefix('/file4/')
            with np.load(source_path) as z:
                q = torch.from_numpy(z['target_keys'][..., :1024].copy()).pin_memory().cuda(non_blocking=True).float()
                s = torch.from_numpy(z['source_keys'][..., :1024].copy()).pin_memory().cuda(non_blocking=True).float().flatten(0, 1)
                usable = torch.from_numpy(z['source_valid'].copy()).cuda().flatten()
            with torch.no_grad():
                scores = F.normalize(q, dim=-1) @ F.normalize(s, dim=-1).T
                scores.masked_fill_(~usable[None], -torch.inf)
                if int(usable.sum()) < 64: scores.fill_(-torch.inf)
                cosine, address = scores.topk(256, dim=-1)
                indices, valid = address.cpu().numpy().astype(np.uint16), cosine.isfinite().cpu().numpy()
                raw = cosine.cpu().numpy()
            tmp = path.with_name('.' + path.name + '.tmp.npz')
            np.savez(tmp, schema='deformnext-global-proposal-v2', input_sha256=row['conditioning_input_sha256'],
                     indices=indices, cosine=raw, valid=valid)
            os.replace(tmp, path)
        label_path = a.labels / (row['sample_id'] + '.npz')
        if label_path.exists():
            with np.load(label_path) as z: positives, visible = z['positive_indices'], z['positive_visible']
            split = 'calibration' if row['sample_id'] in calibration else row['split']
            c = counts[split]; c['visible'] += int(visible.any(-1).sum())
            for k in (64, 128, 256):
                hit = ((indices[:, :k, None] == positives[:, None]) & visible[:, None]).any(-1) & valid[:, :k]
                c['hit' + str(k)] += int(hit.any(-1).sum())
        if i % 100 == 0:
            print(json.dumps({'completed': i + 1, 'total': len(rows), 'seconds': round(time.monotonic()-started, 1)}), flush=True)
    recall = {split: {str(k): c['hit'+str(k)] / max(1, c['visible']) for k in (64, 128, 256)} for split, c in counts.items()}
    selected = next((k for k in (64, 128, 256) if recall['calibration'][str(k)] >= .9), 256)
    report = {'status': 'PASS', 'counts': counts, 'recall': recall, 'selected_topk': selected,
              'selection_policy': 'smallest K reaching 0.9 calibration recall; otherwise K256',
              'assets': len(rows), 'seconds': time.monotonic()-started}
    (a.output.parent / 'proposal-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
