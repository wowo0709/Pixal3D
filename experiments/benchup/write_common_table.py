"""Render provisional common-success BenchUp metrics to a human-readable table."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads(a.input.read_text())
    if r['status']!='PROVISIONAL_COMMON_SUCCESS_INPUT_IMAGE_METRICS':raise ValueError('unexpected aggregate')
    lines=['# BenchUp 공통 성공 사례 비교 (잠정)','',f"현재 전 모델 공통 성공: **{r['common_count']} / 109**. 개별 완료 건수: "+', '.join(f'{k} {v}/109' for k,v in r['available_counts'].items())+'.','',
    '모든 행은 같은 공통 사례를 쓴다. 편집 target GT mesh와 공식 카메라는 없으며, target RGB는 입력 이미지, source RGB는 원본 보존의 대리 지표다. 추정 카메라와 조명 차이 때문에 논문 최종 수치가 아니다.',
    'FID/KID는 edited target 입력 이미지와 렌더 이미지를 비교한다. 사례 수가 100 미만이거나 동일 물체의 여러 편집이 포함되면 값이 특히 불안정하다. CD/F-Score는 target GT mesh 부재로 계산할 수 없다.','',
    '| 모델 | 정합 | N | PSNR ↑ | SSIM ↑ | M-IoU ↑ | LPIPS ↓ | DreamSim ↓ | CLIP ↑ | FID ↓ | KID ↓ |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in ('raw','source_m0'):
        rows=[]
        for name,modes in r['methods'].items():
            v=modes[mode]
            if v is not None:rows.append((name,[v['PSNR'],v['SSIM'],v['M_IoU'],v['source_LPIPS'],v['source_DreamSim'],v['source_CLIP'],v['FID'],v['KID']['KID'] if v['KID'] else None]))
        # Rank each metric independently within the same alignment regime.
        ranks=[]
        for i,higher in enumerate((True,True,True,False,False,True,False,False)):
            values=sorted({vals[i] for _,vals in rows if vals[i] is not None},reverse=higher)
            ranks.append(values[:2] if len(rows)>1 else [])
        for name,vals in rows:
            cells=[]
            for i,v in enumerate(vals):
                if v is None:cells.append('—');continue
                value=f'{v:.5f}'
                if ranks[i] and v==ranks[i][0]:value=f'**{value}**'
                elif len(ranks[i])>1 and v==ranks[i][1]:value=f'<u>{value}</u>'
                cells.append(value)
            lines.append(f"| {name} | {mode} | {r['common_count']} | "+' | '.join(cells)+' |')
    if r['common_count']==0:lines+=['','아직 모든 모델에서 성공한 공통 사례가 없어 수치를 계산하지 않았다.']
    lines.append('');a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text('\n'.join(lines));tmp.replace(a.output)

if __name__=='__main__':main()
