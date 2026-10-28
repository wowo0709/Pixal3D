"""Refresh the two interim comparison tables from validated Ours evaluator rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

from .watch_partial_ours_common501 import (METHODS, METRICS, ORDER, PRECISION,
                                           filename, validate)


LABELS = {'ours_selected_raw': 'Ours Selected', 'ours_v1_raw': 'Ours V1',
          'ours_selected_source_m0': 'Ours Selected',
          'ours_v1_source_m0': 'Ours V1'}


def update_table(path: Path, condition: str, cache: Path, ids: list[str]) -> int:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('| 방법 | LPIPS'))
    end = next(i for i in range(start + 2, len(lines)) if not lines[i].startswith('|'))
    indices = range(start + 2, end)
    completed = 0
    for method in METHODS:
        if (method.endswith('source_m0')) != (condition == 'source_m0'):
            continue
        candidate = cache / filename(method)
        if not candidate.is_file():
            continue
        values = validate(candidate, method, ids)
        completed += 1
        row_index = next(i for i in indices if lines[i].startswith('| ' + LABELS[method] + ','))
        cells = [cell.strip() for cell in lines[row_index].strip('|').split('|')]
        for col, metric in enumerate(METRICS, start=1):
            if metric in values:
                cells[col] = f'{values[metric] * (1000 if metric == "KID" else 1):.{PRECISION[metric]}f}'
        lines[row_index] = '| ' + ' | '.join(cells) + ' |'
    for col, metric in enumerate(METRICS, start=1):
        ranked = []
        for i in indices:
            cells = [cell.strip() for cell in lines[i].strip('|').split('|')]
            cells[col] = re.sub(r'</?u>|\*\*', '', cells[col])
            lines[i] = '| ' + ' | '.join(cells) + ' |'
            if cells[col] != '—':
                ranked.append((ORDER[metric] * float(cells[col]), i))
        ranked.sort(reverse=True)
        for rank, (_, i) in enumerate(ranked[:2]):
            cells = [cell.strip() for cell in lines[i].strip('|').split('|')]
            cells[col] = ('**' + cells[col] + '**' if rank == 0
                          else '<u>' + cells[col] + '</u>')
            lines[i] = '| ' + ' | '.join(cells) + ' |'
    if completed == 2 and condition == 'raw':
        lines = [line.replace('Selected raw의 source·geometry 지표는 공통 evaluator 완료값이고 target view 지표는 별도 paired 평가 완료값이다. V1 source 지표는 계산 중이며 `—`로 표시하고 V1 CD/F-Score는 기존 surface cache의 중간 집계다.',
                              'Ours 두 raw 행의 source·geometry 지표는 공통 evaluator 완료값이고 target view 지표는 별도 paired 평가 완료값이다.')
                 for line in lines]
    if completed == 2 and condition == 'source_m0':
        lines = [line.replace('Ours 두 행의 CD/F-Score는 위 surface cache에서 집계한 임시 값이고', 'Ours 두 행의 CD/F-Score는 공통 evaluator 완료값이고')
                     .replace('Ours source 지표의 `—`는 계산 중이다.', 'Ours source 지표는 공통 evaluator 완료값이다.')
                 for line in lines]
    if completed:
        updated = '\n'.join(lines) + '\n'
        if updated != path.read_text():
            temp = path.with_suffix(path.suffix + '.tmp')
            temp.write_text(updated)
            temp.replace(path)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-table', type=Path, required=True)
    parser.add_argument('--m0-table', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--raw-external', type=Path, required=True)
    parser.add_argument('--evaluation-pid', type=int, required=True)
    parser.add_argument('--interval-seconds', type=int, default=120)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval_seconds < 10:
        raise ValueError('bounded interval required')
    external = json.loads(args.raw_external.read_text())
    if external['status'] != 'PASS_RAW_EXTERNAL_COMMON501_FEATURE_REPLAY':
        raise ValueError('common population receipt required')
    ids = external['sample_ids']
    if len(ids) != 501 or len(set(ids)) != 501:
        raise ValueError('exact common501 population required')
    while True:
        raw = update_table(args.raw_table, 'raw', args.cache, ids)
        m0 = update_table(args.m0_table, 'source_m0', args.cache, ids)
        print(json.dumps({'raw': raw, 'source_m0': m0}), flush=True)
        if args.once or raw + m0 == 4:
            return
        import os
        try:
            os.kill(args.evaluation_pid, 0)
        except ProcessLookupError:
            raise RuntimeError('evaluation watcher ended before four conditions')
        time.sleep(args.interval_seconds)


if __name__ == '__main__':
    main()
