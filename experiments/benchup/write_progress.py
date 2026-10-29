"""Report provisional BenchUp generation, processed meshes and paired metrics."""
from __future__ import annotations
import argparse, json, os, time
from datetime import datetime, timezone
from pathlib import Path


def ids(paths, population):
    result = set()
    for path in paths:
        sid = path.parent.name
        if sid not in population:
            raise ValueError(f'unexpected case: {path}')
        result.add(sid)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--watch-pids', default='')
    a = p.parse_args()
    rows = [json.loads(line) for line in a.manifest.read_text().splitlines() if line]
    population = {r['sample_id'] for r in rows}
    if len(rows) != 109 or len(population) != 109:
        raise ValueError('expected 109 unique cases')
    roots = {'V1': ('v1-cascade-shard-*', 'v1'),
             'Selected': ('selected-pbr-shard-*', 'selected'),
             'MV-zero': ('mv-zero-cascade-shard-*', 'mv-zero'),
             'Official-RGB1': ('official-rgb1-full-v1', 'official-rgb1'),
             'TRELLIS2': ('trellis2-full-v1', 'trellis2'),
             'ReconViaGen': ('recon-full-v1', 'recon'),
             'ShapeUP': ('shapeup-texture-full-v1', 'shapeup'),
             'SpaceControl': ('spacecontrol-full-v1', 'spacecontrol'),
             'MVAdapter-Recon': ('mvadapter-recon-full-v1', 'mvadapter-recon'),
             'CorrAdapter-Recon': ('corradapter-recon-full-v1', 'corradapter-recon')}
    pids = [int(x) for x in a.watch_pids.split(',') if x]
    while True:
        lines = ['# BenchUp 단계별 진행 현황 (잠정)', '',
                 '생성 완료, file4에 검증 보관된 렌더용 mesh, 이미지 지표 완료 사례의 개수다. 대상은 공개 편집 109건이다.', '',
                 '| 모델 | 생성 | 후처리 | 이미지 지표 |', '|---|---:|---:|---:|']
        for label, (root, method) in roots.items():
            generated = ids(a.runtime.glob(root + '/cases/*/receipt.json'), population)
            processed = ids(a.runtime.glob(method + '-postprocess-shard-*/cases/*/archive.json'), population)
            basic = ids(a.runtime.glob(method + '-render-stream-v1/cases/*/diagnostic-scores.json'), population)
            learned = ids(a.runtime.glob(method + '-learned-metrics-v1/*/learned.json'), population)
            lines.append(f'| {label} | {len(generated)}/109 | {len(processed)}/109 | {len(basic & learned)}/109 |')
        lines += ['', '지표는 임시 카메라와 입력 이미지 참조를 사용한다. 방법별 완료 집합은 다르므로 공통 성공 집합의 별도 표를 이용한다.',
                  f'갱신 시각: {datetime.now(timezone.utc).isoformat(timespec="seconds")} UTC', '']
        a.output.parent.mkdir(parents=True, exist_ok=True)
        temp = a.output.with_suffix('.tmp')
        temp.write_text('\n'.join(lines))
        temp.replace(a.output)
        alive = False
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                pass
        if not alive:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
