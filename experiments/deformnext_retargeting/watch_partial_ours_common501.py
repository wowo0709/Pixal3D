"""Publish completed Ours conditions as the common evaluator writes them."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


METHODS = ('ours_selected_raw', 'ours_v1_raw', 'ours_selected_source_m0', 'ours_v1_source_m0')
METRICS = ('LPIPS', 'DreamSim', 'CLIP', 'FID', 'KID', 'PSNR', 'SSIM', 'M_IoU', 'CD', 'FScore')
ORDER = {'LPIPS': -1, 'DreamSim': -1, 'CLIP': 1, 'FID': -1, 'KID': -1,
         'PSNR': 1, 'SSIM': 1, 'M_IoU': 1, 'CD': -1, 'FScore': 1}
PRECISION = {'LPIPS': 5, 'DreamSim': 5, 'CLIP': 5, 'FID': 2, 'KID': 3,
             'PSNR': 3, 'SSIM': 4, 'M_IoU': 4, 'CD': 5, 'FScore': 4}
REMOTE = '/home/youngwoo/retargeting-unified-test508/ours-four-final-evaluation-v1/evaluation-ours-four'


def filename(method):
    return hashlib.sha256(json.dumps(method, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16] + '-results.json'


def validate(path, method, ids):
    result = json.loads(path.read_text())
    summary = result['summary']
    rows, views = result['per_deformation_seed'], result['per_view']
    expected = set(ids)
    if (result['method'] != method or result['failures']
            or summary['successful_predictions'] != len(ids)
            or not summary['full_metric_computation']
            or summary['distribution']['image_count'] != 2 * len(ids)
            or len(rows) != len(ids) or {r['sample_id'] for r in rows} != expected
            or len(views) != 2 * len(ids)
            or {(v['sample_id'], v['camera_id']) for v in views}
               != {(sid, camera) for sid in ids for camera in (9, 10)}):
        raise ValueError('incomplete or mismatched condition: ' + method)
    row = {k: float(summary['deformation_weighted'][k])
           for k in ('LPIPS', 'DreamSim', 'CLIP', 'CD', 'FScore')}
    row.update({k: float(summary['distribution'][k]) for k in ('FID', 'KID')})
    if not all(math.isfinite(v) for v in row.values()):
        raise ValueError('nonfinite condition: ' + method)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-external', type=Path, required=True)
    p.add_argument('--target8', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--evaluation-pid', type=int, required=True)
    p.add_argument('--interval-seconds', type=int, default=120)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    if a.interval_seconds < 10:
        raise ValueError('bounded polling interval required')
    raw, target = json.loads(a.raw_external.read_text()), json.loads(a.target8.read_text())
    ids = raw['sample_ids']
    if (raw['status'] != 'PASS_RAW_EXTERNAL_COMMON501_FEATURE_REPLAY' or len(ids) != 501
            or len(set(ids)) != 501 or target['status'] != 'PASS_OURS_V1_SELECTED_RAW_AND_M0_TARGET8_PAIRED'
            or target['sample_ids'] != ids):
        raise ValueError('matched common501 target and source population required')
    a.cache.mkdir(parents=True, exist_ok=True)
    while True:
        found = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-p', '55555',
            'youngwoo@n16.unist.info', f'find {REMOTE} -maxdepth 1 -name "*-results.json" -printf "%f\\n" 2>/dev/null'],
            capture_output=True, text=True, timeout=30)
        available = set(found.stdout.splitlines()) if found.returncode == 0 else set()
        values = {}
        for method in METHODS:
            name = filename(method)
            local = a.cache / name
            if not local.exists() and name in available:
                temp = a.cache / (name + '.tmp')
                try:
                    subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes -p 55555',
                                    f'youngwoo@n16.unist.info:{REMOTE}/{name}', str(temp)],
                                   check=True, timeout=60, capture_output=True)
                    validate(temp, method, ids)
                    temp.replace(local)
                except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.SubprocessError):
                    temp.unlink(missing_ok=True)
            if local.exists():
                values[method] = validate(local, method, ids)
            condition = 'source_m0' if method.endswith('source_m0') else 'raw'
            for metric in ('PSNR', 'SSIM', 'M_IoU'):
                pair = target['metrics'][metric]['selected_minus_v1_' + condition]
                values.setdefault(method, {})[metric] = float(pair['right_mean'] if 'selected' in method else pair['left_mean'])
        lines = ['# Ours 공통 TEST501 조건별 평가 진행', '',
                 '갱신: ' + datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') + '.', '',
                 '같은 501개 사례·133 source cluster. Source와 geometry는 조건별 공통 evaluator 완료 후 표시한다.',
                 'Target view 지표는 별도 paired 계산을 사용한다. KID는 원값×1,000이다.', '',
                 '| 정합 | 모델 | LPIPS ↓ | DreamSim ↓ | CLIP ↑ | FID ↓ | KID ×1,000 ↓ | PSNR ↑ | SSIM ↑ | Mask IoU ↑ | CD ↓ | F@0.02 ↑ |',
                 '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for method in METHODS:
            aligned = method.endswith('source_m0')
            partner = ('ours_v1_' if 'selected' in method else 'ours_selected_') + ('source_m0' if aligned else 'raw')
            row = ['Source M0 ICP' if aligned else 'Raw', 'Selected' if 'selected' in method else 'V1']
            for metric in METRICS:
                value = values[method].get(metric)
                if value is None:
                    row.append('—'); continue
                cell = f'{value * (1000 if metric == "KID" else 1):.{PRECISION[metric]}f}'
                other = values[partner].get(metric)
                if other is not None and value != other:
                    cell = '**' + cell + '**' if ORDER[metric] * value > ORDER[metric] * other else '<u>' + cell + '</u>'
                row.append(cell)
            lines.append('| ' + ' | '.join(row) + ' |')
        lines += ['', '굵게와 밑줄은 같은 정합 조건에서 V1·Selected의 1위와 2위다. '
                  'Source 7방법 및 전체 10방법은 [raw 표](2026-10-28-raw-common501-interim-ko.md)와 '
                  '[ICP 표](2026-10-28-ours-common501-target8-interim-ko.md)를 참고한다.', '',
                  f'완료된 공통 evaluator 조건: {sum((a.cache / filename(m)).exists() for m in METHODS)}/4. '
                  '이 문서는 중간 결과이며 원고의 최종 통합표는 전체 검증 후 생성한다.']
        a.output.parent.mkdir(parents=True, exist_ok=True)
        temp = a.output.with_suffix(a.output.suffix + '.tmp')
        temp.write_text('\n'.join(lines) + '\n'); temp.replace(a.output)
        complete = sum((a.cache / filename(m)).exists() for m in METHODS)
        print(json.dumps({'complete_conditions': complete, 'remote_query_ok': found.returncode == 0}), flush=True)
        if complete == 4 or a.once:
            break
        try: os.kill(a.evaluation_pid, 0)
        except ProcessLookupError: raise RuntimeError('evaluation ended before four validated conditions')
        time.sleep(a.interval_seconds)


if __name__ == '__main__':
    main()
