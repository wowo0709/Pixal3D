"""Write a provisional, common-population BenchUp table for a paper draft."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=json.loads(a.input.read_text())
    if r['status']!='PROVISIONAL_COMMON_SUCCESS_INPUT_IMAGE_METRICS':raise ValueError('unexpected aggregate')
    names={'V1':'MV-Pixal3D (FT)','Selected':'GEAR3D (selected)','MV-zero':'MV-Pixal3D (zero-shot)','Official-RGB1':'Pixal3D (RGB1)','TRELLIS2':'TRELLIS.2','ReconViaGen':'ReconViaGen','ShapeUP':'ShapeUP','SpaceControl':'SpaceControl','MVAdapter-Recon':'MVAdapter + Recon','CorrAdapter-Recon':'CorrAdapter + Recon'}
    columns=('source_LPIPS','source_DreamSim','source_CLIP','FID','KID','PSNR','SSIM','M_IoU')
    directions=(False,False,True,False,False,True,True,True)
    out=['% PROVISIONAL: validate target camera assumption and complete evaluation before using in a paper.',r'\begin{table*}[t]',r'\centering',
         r'\caption{Provisional BenchUp comparison on '+str(r['common_count'])+r' shared successful edits. Raw and source-M0 results use the same cases; source-M0 aligns only to the input source mesh. Source image metrics are preservation proxies, and target image metrics use the edited input. Cameras are estimated and the target-view association is unverified. Target CD and F-score are unavailable without target ground-truth meshes. FID/KID compare target renders with edited inputs and can be unstable for small, correlated samples.}',
         r'\label{tab:benchup-provisional}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{llrrrrrrrrrr}',r'\toprule',
         r'Method & Alignment & LPIPS$\downarrow$ & DreamSim$\downarrow$ & CLIP$\uparrow$ & FID$\downarrow$ & KID$\downarrow$ & PSNR$\uparrow$ & SSIM$\uparrow$ & M-IoU$\uparrow$ & CD$\downarrow$ & F-score$\uparrow$ \\',r'\midrule']
    for mi,mode in enumerate(('raw','source_m0')):
        rows=[]
        for name,modes in r['methods'].items():
            v=modes[mode];values=[]
            for key in columns:
                value=v.get(key) if v is not None else None
                if key=='KID' and value is not None:value=value['KID']
                values.append(value)
            rows.append((name,values))
        ranks=[]
        for i,higher in enumerate(directions):
            ranks.append(sorted({v[i] for _,v in rows if v[i] is not None},reverse=higher)[:2])
        if mi:out.append(r'\midrule')
        for name,values in rows:
            cells=[]
            for i,value in enumerate(values):
                if value is None:cells.append('--');continue
                text=f'{value:.4f}'
                if ranks[i] and value==ranks[i][0]:text=r'\textbf{'+text+'}'
                elif len(ranks[i])>1 and value==ranks[i][1]:text=r'\underline{'+text+'}'
                cells.append(text)
            out.append(names.get(name,name.replace('_',r'\_'))+' & '+('Raw' if mode=='raw' else 'Source-M0')+' & '+' & '.join(cells+['--','--'])+r' \\')
    out.extend([r'\bottomrule',r'\end{tabular}%',r'}',r'\end{table*}',''])
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text('\n'.join(out));tmp.replace(a.output)
    print(json.dumps({'status':'PROVISIONAL_LATEX_TABLE','cases':r['common_count'],'output':str(a.output)}))

if __name__=='__main__':main()
