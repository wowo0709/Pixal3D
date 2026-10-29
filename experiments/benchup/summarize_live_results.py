"""Publish explicitly provisional paired BenchUp image diagnostics."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import numpy as np

def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def gather(render,learned):
    def load(paths):
        out={}
        for p in paths:
            try:out[p.parent.name]=json.loads(p.read_text())
            except (OSError,json.JSONDecodeError):continue
        return out
    r=load(render.glob('cases/*/diagnostic-scores.json'))
    l=load(learned.glob('*/learned.json'))
    return {sid:(r[sid],l[sid]) for sid in r.keys()&l.keys()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('v1-render','v1-learned','selected-render','selected-learned','output'):p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--zero-render',type=Path);p.add_argument('--zero-learned',type=Path)
    p.add_argument('--producer-pids',required=True);a=p.parse_args();pids=[int(x) for x in a.producer_pids.split(',')]
    while True:
        data={'V1':gather(a.v1_render,a.v1_learned),'Selected':gather(a.selected_render,a.selected_learned)}
        if (a.zero_render is None)!=(a.zero_learned is None):raise ValueError('both zero paths required')
        if a.zero_render is not None:data['MV zero']=gather(a.zero_render,a.zero_learned)
        active={name:items for name,items in data.items() if items}
        common=set.intersection(*(set(x) for x in active.values())) if active else set()
        lines=['# BenchUp 실시간 이미지 진단 (잠정)','',
            '공개된 109개 편집 이미지 중 아래 각 행의 완료 사례만 집계한다. 카메라는 원본 이미지와 source mesh에서 추정했고 target이 같은 인덱스의 카메라라는 가정은 검증되지 않았다.',
            'Target은 편집 입력 RGB, source는 원본 입력 두 뷰다. 렌더 조명이 원본 이미지와 다르고 source 보존 점수는 의도한 편집도 벌점화할 수 있으므로 아래는 논문 최종 수치가 아니다.',
            'GT target mesh가 없어 target CD/F-score는 계산하지 않는다. FID/KID는 전체 모집단 및 공통 성공 집합이 확정될 때까지 계산하지 않는다.','',
            '| 모델 | 집계 | 정합 | N / 109 | Target PSNR ↑ | Target SSIM ↑ | Target M-IoU ↑ | Source LPIPS ↓ | Source DreamSim ↓ | Source CLIP ↑ |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for subset in ('available','paired'):
            for name,items in data.items():
                ids=sorted(items if subset=='available' else common if items else [])
                if not ids:continue
                for mode in ('raw','source_m0'):
                    vals=[]
                    for sid in ids:
                        basic,learn=items[sid];target=basic['metrics'][mode][0]['metrics'];source=[x['metrics'] for x in learn['metrics'][mode][1:]]
                        vals.append([target['PSNR'],target['SSIM'],target['M_IoU'],*[np.mean([s[k] for s in source]) for k in ('LPIPS','DreamSim','CLIP')]])
                    means=np.asarray(vals,dtype=np.float64).mean(0)
                    lines.append(f'| {name} | {subset} | {mode} | {len(ids)} / 109 | '+ ' | '.join(f'{x:.4f}' for x in means)+' |')
        lines+=['',f"동일 성공 교집합: {len(common)}개 ({', '.join(active)}). 각 사례의 상세 점수와 입력 해시는 file4의 `benchup/*-image-*-provisional-v1`에 보관한다.",
                '목표는 전체 Table 1 baseline과 ours 비교이며, 이 문서는 진행 중인 일부 행만 보여 준다.','']
        a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text('\n'.join(lines));os.replace(tmp,a.output)
        if not any(live(pid) for pid in pids):break
        time.sleep(30)

if __name__=='__main__':main()
