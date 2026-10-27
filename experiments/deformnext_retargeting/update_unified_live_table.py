"""Poll node16 and atomically publish the common-502 live result table."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time

from compose_unified_test508_table import EXTERNAL, INTERNAL, INPUT, LABEL, METRIC_DIRECTIONS


REMOTE_ROOT = '/home/youngwoo/retargeting-unified-test508'
REMOTE_QUERY = '''import json
from pathlib import Path
r=Path("/home/youngwoo/retargeting-unified-test508")
def read(name):
 p=r/name
 return json.loads(p.read_text()) if p.is_file() else None
def compact_source(doc):
 if not doc:return None
 return {k:{"success":v["full_successes"],"source":v["common"]["deformation_weighted"],"distribution":v["distribution"]} for k,v in doc["methods"].items()}
def compact_target(doc):
 if not doc:return None
 return {k:v["common"]["deformation_weighted"] for k,v in doc["methods"].items()}
print(json.dumps({"progress":read("completion-progress.json"),"completion":read("unified-completion.json"),"source_external":compact_source(read("common-source-external-502.json")),"source_all":compact_source(read("common-source-all13-502.json")),"target_external":compact_target(read("target8-external-502.json")),"target_all":compact_target(read("target8-scores.json"))}))
'''


def remote(command: str) -> str:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-p', '55555',
                             'youngwoo@n16.unist.info', command],
                            check=True, text=True, capture_output=True)
    return result.stdout


def fetch() -> dict:
    return json.loads(remote('python3 -c ' + shlex.quote(REMOTE_QUERY)))


def fmt(value, digits: int) -> str:
    return '—' if value is None else f'{value:.{digits}f}'


def live_ranks(source: dict, target: dict) -> dict[str, dict[str, int]]:
    """Mark only complete available values; recalculate as rows arrive."""
    values = {method: {} for method in (*EXTERNAL, *INTERNAL)}
    for method in values:
        s, t = source.get(method), target.get(method)
        if s:
            for metric in ('LPIPS', 'DreamSim', 'CLIP', 'CD', 'FScore'):
                values[method][metric] = s['source'][metric]
            for metric in ('FID', 'KID'):
                values[method][metric] = s['distribution'][metric]
        if t:
            for metric in ('PSNR', 'SSIM', 'M_IoU'):
                values[method][metric] = t[metric]
    ranks = {method: {} for method in values}
    for metric, direction in METRIC_DIRECTIONS.items():
        unique = sorted({v[metric] for v in values.values() if metric in v},
                        reverse=direction == 'max')
        if len(unique) < 2:
            continue
        for method, per_method in values.items():
            if per_method.get(metric) == unique[0]:
                ranks[method][metric] = 1
            elif per_method.get(metric) == unique[1]:
                ranks[method][metric] = 2
    return ranks


def fmt_marked(value, digits: int, rank: int | None) -> str:
    formatted = fmt(value, digits)
    if rank == 1:
        return '**' + formatted + '**'
    if rank == 2:
        return '<u>' + formatted + '</u>'
    return formatted


def make_table(snapshot: dict) -> str:
    source = snapshot['source_all'] or snapshot['source_external'] or {}
    target = snapshot['target_all'] or snapshot['target_external'] or {}
    progress = snapshot['progress'] or {}
    actual, expected = progress.get('actual', {}), progress.get('expected', {})
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    complete = snapshot['completion'] is not None
    if complete:
        actual = expected = snapshot['completion']['counts']
    ranking = live_ranks(source, target)
    lines = ['# 공통 TEST508 평가 중간 표', '',
             f'자동 갱신: {now}. 상태: **{"완료" if complete else "진행 중"}**.', '',
             '**이 표는 입력 조건이 다른 방법을 나열한 진단표이며, 카메라 조건을 맞춘 논문 주 비교표가 아니다.** '
             '성공 수의 분모는 508이다. 아래 모든 수치 지표는 전 방법이 성공한 동일한 502개 편집(133 source cluster)에서 계산한다. '
             'FID/KID는 camera9/10의 동일한 1,004장 Inception feature를 사용한다. `—`는 아직 계산이 끝나지 않았다는 뜻이다. '
             '**굵게**는 현재 채워진 행 중 최상위, <u>밑줄</u>은 두 번째 값이며, 최종 13개 행이 채워지면 다시 순위를 매긴다.', '',
             f'진행률: ours GLB {actual.get("exports", 0)}/{expected.get("exports", 3048)}, '
             f'ours Cycles 렌더 {actual.get("ours_renders", 0)}/{expected.get("ours_renders", 9144)}, '
             f'GT·외부 target8 렌더 {actual.get("baseline_target8", 0)}/{expected.get("baseline_target8", 4054)}.', '',
             '| 방법 | 입력 | 성공 | LPIPS↓ | DreamSim↓ | CLIP↑ | FID↓ | KID↓ | PSNR↑ | SSIM↑ | M-IoU↑ | CD↓ | F@0.02↑ |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method in (*EXTERNAL, *INTERNAL):
        s = source.get(method)
        t = target.get(method)
        metrics = s['source'] if s else {}
        dist = s['distribution'] if s else {}
        rank = ranking[method]
        cell = [LABEL[method], INPUT[method],
                str(s['success']) + '/508' if s else '—',
                fmt_marked(metrics.get('LPIPS'), 4, rank.get('LPIPS')),
                fmt_marked(metrics.get('DreamSim'), 4, rank.get('DreamSim')),
                fmt_marked(metrics.get('CLIP'), 4, rank.get('CLIP')),
                fmt_marked(dist.get('FID'), 2, rank.get('FID')),
                fmt_marked(dist.get('KID'), 6, rank.get('KID')),
                fmt_marked(t.get('PSNR') if t else None, 2, rank.get('PSNR')),
                fmt_marked(t.get('SSIM') if t else None, 4, rank.get('SSIM')),
                fmt_marked(t.get('M_IoU') if t else None, 4, rank.get('M_IoU')),
                fmt_marked(metrics.get('CD'), 5, rank.get('CD')),
                fmt_marked(metrics.get('FScore'), 4, rank.get('FScore'))]
        lines.append('| ' + ' | '.join(cell) + ' |')
        if method == EXTERNAL[-1]:
            lines.append('| **Ours ↓** | | | | | | | | | | | | |')
    lines.extend(['', '입력: T1=edited M1 RGB@8, S3=original M0 RGB@9/10/11, '
                  'S4=original M0 RGB@8/9/10/11, A1=original anchor@8, '
                  'C=제공된 camera-to-world, FOV, 거리와 target M1 mesh에서 계산된 사례별 mesh scale. '
                  '이 기존 진단 결과는 GT-free camera-given 주 비교 결과가 아니다. '
                  '공식 Pixal3D는 RGB에서 MoGe로 카메라를 추정하고 native 1536 cascade를 사용한다. '
                  '보정 RGB1은 원본 single-view 가중치를 projective 1024 cascade와 C에 넣은 별도 실행이다. '
                  '9/10은 source-conditioned 방법의 입력 뷰이며 target-only 방법에는 미관측 뷰다.', '',
                  '초기 11개 동일 사례 점검은 [QA11 기록](2026-10-27-early-common-qa11-ko.md)에 분리했다. '
                  '3 source cluster만 포함하므로 위 공통 502개 표의 미완료 ours 칸에 대입하지 않는다.', '',
                  '외부법 중 MVAdapter는 504/508, CorrAdapter는 502/508 성공했다. '
                  '이 성공 수는 공통 502개 지표와 별도로 해석해야 한다. '
                  '최종 논문 주장과 유의성 판단은 모든 ours 결과 및 paired source-cluster 비교가 끝난 뒤 확정한다.', ''])
    return '\n'.join(lines)


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(content)
    temp.replace(path)


def make_final_report(snapshot: dict) -> str:
    if not snapshot['completion'] or not snapshot['source_all'] or not snapshot['target_all']:
        raise ValueError('all 13 common scores are required for the final result report')
    source, target = snapshot['source_all'], snapshot['target_all']
    def value(arm: str, metric: str) -> float:
        if metric in ('PSNR', 'SSIM', 'M_IoU'):
            return target[arm][metric]
        return source[arm]['source'][metric]
    lines = ['# 공통 TEST508 통합 평가 결과', '',
             '자동 생성된 수치 정리다. 13행 표는 감사용 전체 목록이다. 입력 조건이 다른 외부 비교 진단표'
             '(`heterogeneous-input-diagnostic-test508-table.tex`)와 내부 ablation 표(`ablation-test508-table.tex`)는'
             ' 같은 점수 파일에서 생성했고 Selected 예측/수치는 두 표에서 완전히 같다.'
             ' 논문 주장과 통계적 유의성은 paired source-cluster 대비를 확인한 뒤 확정한다.', '',
             make_table(snapshot), '', '## 같은 입력을 쓰는 내부 비교의 점추정', '',
             '| 지표 | V1 | Top-1 R | Selected | R−V1 | Selected−R |',
             '|---|---:|---:|---:|---:|---:|']
    for metric, digits in (('CD', 5), ('FScore', 4), ('LPIPS', 4),
                           ('DreamSim', 4), ('M_IoU', 4), ('PSNR', 2)):
        v, r, s = (value(a, metric) for a in ('V1', 'RETARGET', 'SELECTED'))
        lines.append(f'| {metric} | {v:.{digits}f} | {r:.{digits}f} | '
                     f'{s:.{digits}f} | {r-v:+.{digits}f} | {s-r:+.{digits}f} |')
    lines.extend(['', 'CD, LPIPS, DreamSim은 낮을수록 좋고 FScore, M-IoU, PSNR은 높을수록 좋다. '
                  '이 차이는 동일 502개 사례의 단순 평균 차이이며 paired confidence interval 또는 training-seed 변동을 대신하지 않는다.', '',
                  '외부법은 입력 조건과 생성 스택이 서로 달라 task-level 비교다. '
                  '공식 Pixal3D와 프로젝트 보정 RGB1은 별도 행이다. '
                  'camera9/10은 source-conditioned 방법의 입력 뷰이므로 보편적인 novel-view 성능으로 해석하지 않는다.', ''])
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--final-tex', type=Path, required=True)
    parser.add_argument('--final-report', type=Path, required=True)
    parser.add_argument('--interval-seconds', type=int, default=120)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval_seconds < 30:
        raise ValueError('polling interval too short')
    while True:
        try:
            snapshot = fetch()
            write_atomic(args.output, make_table(snapshot))
            print(json.dumps({'updated': str(args.output),
                              'complete': snapshot['completion'] is not None,
                              'actual': (snapshot['progress'] or {}).get('actual')}), flush=True)
            if snapshot['completion']:
                table = remote('cat ' + shlex.quote(REMOTE_ROOT + '/unified-test508-table.tex'))
                write_atomic(args.final_tex, table)
                paper_command = ('/usr/bin/python3 ' +
                    shlex.quote(REMOTE_ROOT + '/scripts/compose_paper_test508_tables.py') +
                    ' --common-source-score ' + shlex.quote(REMOTE_ROOT + '/common-source-all13-502.json') +
                    ' --target8-scores ' + shlex.quote(REMOTE_ROOT + '/target8-scores.json') +
                    ' --output-dir ' + shlex.quote(REMOTE_ROOT))
                remote(paper_command)
                for basename in ('heterogeneous-input-diagnostic-test508-table.tex',
                                 'ablation-test508-table.tex'):
                    content = remote('cat ' + shlex.quote(REMOTE_ROOT + '/' + basename))
                    write_atomic(args.final_tex.with_name(basename), content)
                write_atomic(args.final_report, make_final_report(snapshot))
                break
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            print(json.dumps({'poll_error': str(exc)}), flush=True)
            if args.once:
                raise
        if args.once:
            break
        time.sleep(args.interval_seconds)


if __name__ == '__main__':
    main()
