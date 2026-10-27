"""Draw publication-size method figures and a scored-image TEST teaser.

Only the teaser reads generated images. All predictions and the GT reference
come from the completed TEST scoring archives; no mesh is rendered here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
from PIL import Image


NAVY = '#202B3B'
MUTED = '#596878'
BLUE = '#337DB6'
LIGHT_BLUE = '#DDEDF7'
VIOLET = '#7654A6'
LIGHT_VIOLET = '#EEE7F7'
TEAL = '#188B83'
LIGHT_TEAL = '#DEF2ED'
AMBER = '#C68032'
LIGHT_AMBER = '#FAEEDD'
LINE = '#B7C1CB'


def setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.linewidth': .7, 'savefig.transparent': False,
    })


def canvas(width: float, height: float, xmax: float = 12, ymax: float = 5):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, xmax), ylim=(0, ymax))
    ax.set_aspect('equal', adjustable='box'); ax.axis('off')
    return fig, ax


def label(ax, x, y, content, *, size=8.5, color=NAVY, weight='normal', ha='left', va='center'):
    return ax.text(x, y, content, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, linespacing=1.15, zorder=10)


def arrow(ax, p, q, *, color=NAVY, width=1.4, head=10, style='solid', curve=0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=head,
        linewidth=width, color=color, linestyle=style,
        connectionstyle=f'arc3,rad={curve}', shrinkA=0, shrinkB=0, zorder=5))


def card(ax, x, y, w, h, *, face='white', edge=LINE, radius=.08, lw=.8):
    patch=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.02,rounding_size={radius}',
        facecolor=face,edgecolor=edge,lw=lw,zorder=1)
    ax.add_patch(patch)
    return patch


def image_tile(ax, x, y, w, h, rgb, *, edge='#D5DDE3'):
    ax.imshow(rgb,extent=[x,x+w,y,y+h],origin='upper',zorder=2,interpolation='bilinear')
    ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=edge,lw=.75,zorder=4))


def patch_grid(ax, x, y, n=5, *, cell=.13, fill='#E5EAF0', accent=None, chosen=()):
    for row in range(n):
        for col in range(n):
            color=accent if (row,col) in chosen else fill
            ax.add_patch(Rectangle((x+col*cell,y+(n-1-row)*cell),cell*.88,cell*.88,
                facecolor=color,edgecolor='white',lw=.4,zorder=3))


def voxel(ax, x, y, *, scale=1, color=TEAL, alpha=.9):
    a=.34*scale;b=.18*scale
    front=np.array([[x,y],[x+a,y],[x+a,y+a],[x,y+a]])
    top=np.array([[x,y+a],[x+b,y+a+b],[x+a+b,y+a+b],[x+a,y+a]])
    side=np.array([[x+a,y],[x+a+b,y+b],[x+a+b,y+a+b],[x+a,y+a]])
    for points,opacity in [(front,alpha),(top,alpha*.58),(side,alpha*.7)]:
        ax.add_patch(Polygon(points,facecolor=color,alpha=opacity,edgecolor='white',lw=.4,zorder=4))


def render_teaser(output: Path, campaign_audit: Path, sample_id: str):
    campaign=json.loads(campaign_audit.read_text())
    if campaign['status']!='PASS_ACTUAL_SELECTED_FINAL_CASE_CAMPAIGN' or sample_id not in campaign['actual_cases']:
        raise ValueError('actual completed TEST case required')
    entries=campaign['actual_cases'][sample_id]
    selected_root=Path(entries['SELECTED']['audit']['path']).parent
    case=json.loads((selected_root/'case-plan.json').read_text())
    plan=case['phases'][0]['plan']
    target_path=Path(plan['RGB_paths'][0]); source_paths=[Path(x) for x in plan['RGB_paths'][1:4]]
    if len(source_paths)!=3:raise ValueError('the actual four-view input must have three M0 sources')
    def load_input(path):
        with Image.open(path) as img:
            rgba=np.asarray(img.convert('RGBA').resize((512,512),Image.Resampling.LANCZOS)).astype(np.float32)/255
        return rgba[...,:3]*rgba[...,3:4]
    input_images=[load_input(p) for p in [target_path,*source_paths]]
    names=['M1 target','M0 source 1','M0 source 2','M0 source 3']
    aliases=['zero-edited4','V1','SELECTED']
    refs={}; predictions={}; hashes={}
    for alias in aliases:
        root=Path(entries[alias]['audit']['path']).parent/'shared'
        for cid in (8,9):
            path=root/f'camera{cid:02d}.npz'
            with np.load(path) as bank:
                rgb=np.transpose(bank['prediction'],(1,2,0)).clip(0,1)
                gt=np.transpose(bank['GT_shared_renderer'],(1,2,0)).clip(0,1)
            predictions[alias,cid]=rgb
            ref=refs.setdefault(cid,gt)
            if not np.array_equal(ref,gt):raise ValueError('scored GT pixels differ across model outputs')
            hashes[f'{alias}/camera{cid:02d}']=hashlib.sha256(path.read_bytes()).hexdigest()
    fig,ax=canvas(7.05,4.10,xmax=14.1,ymax=8.2)
    ax.set_facecolor('white')
    label(ax,.25,7.86,'Cross-geometry editing',size=10.2,weight='bold')
    label(ax,4.7,7.86,'Target geometry + undeformed source appearance',size=8,color=MUTED)
    top_x=[.35,1.90,3.45,5.00]
    for x,name,rgb in zip(top_x,names,input_images):
        label(ax,x+.71,7.38,name,size=7.1,weight='bold',ha='center')
        image_tile(ax,x,5.52,1.42,1.42,rgb)
    arrow(ax,(6.54,6.25),(7.1,6.25),color=TEAL,width=1.65)
    label(ax,8.47,6.47,'Semantic transfer',size=8.3,weight='bold',ha='center',color=TEAL)
    label(ax,8.47,6.03,'ray-wise evidence',size=7.4,ha='center',color=MUTED)
    arrow(ax,(9.68,6.25),(10.4,6.25),color=TEAL,width=1.65)
    label(ax,12.02,6.47,'Pixal3D cascade',size=8.3,weight='bold',ha='center')
    label(ax,12.02,6.03,'SS  /  Shape  /  PBR',size=7.4,ha='center',color=MUTED)
    ax.plot([.32,13.76],[5.27,5.27],color=LINE,lw=.8)
    column_names=['GT','4-view zero-shot','Vanilla V1','Ours']
    xvals=[.88,4.01,7.14,10.27]
    for x,name in zip(xvals,column_names):
        label(ax,x+1.28,4.9,name,size=8.5,ha='center',weight='bold' if name=='Ours' else 'normal',
              color=TEAL if name=='Ours' else NAVY)
    for row,cid in enumerate((8,9)):
        y=2.84-row*2.48
        label(ax,.06,y+1.1,'Target' if cid==8 else 'Source',
              size=7.3,color=MUTED,ha='left')
        panels=[refs[cid],predictions['zero-edited4',cid],predictions['V1',cid],predictions['SELECTED',cid]]
        for x,rgb in zip(xvals,panels):image_tile(ax,x,y,2.55,1.96,rgb)
    label(ax,7.12,.12,'Illustrative TEST case; conditional joint-improvement example.',size=7.1,color=MUTED,ha='center')
    fig.savefig(output/'paper-teaser.pdf',dpi=220,facecolor='white')
    fig.savefig(output/'paper-teaser.png',dpi=220,facecolor='white')
    plt.close(fig)
    (output/'paper-teaser-provenance.json').write_text(json.dumps(dict(sample_id=sample_id,
        selection='predeclared conditional joint-improvement, lower median',
        fullframe_scored_arrays=True,source_prediction_SHA256=hashes,
        input_RGB_SHA256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [target_path,*source_paths]}),indent=2)+'\n')


def render_overall(output: Path):
    fig,ax=canvas(7.05,2.52,xmax=14.1,ymax=5.04)
    label(ax,.25,4.69,'Cross-geometry evidence transfer',size=10.1,weight='bold')
    ax.plot([.25,13.86],[4.43,4.43],color=LINE,lw=.75)
    label(ax,1.01,3.90,'M1 target',size=7.9,weight='bold',ha='center',color=BLUE)
    patch_grid(ax,.51,2.72,n=5,cell=.2,fill='#C7DCEA',accent=BLUE,chosen={(1,2),(2,2),(3,2)})
    label(ax,2.80,3.90,'M0 sources ×3',size=7.9,weight='bold',ha='center',color=VIOLET)
    for i,x in enumerate((2.15,2.4,2.65)):
        patch_grid(ax,x,2.77-i*.09,n=4,cell=.20,fill='#DCCFEA',accent=VIOLET,chosen={(1,1),(2,2)})
    arrow(ax,(1.59,3.2),(4.00,3.55),color=BLUE,width=1.7)
    label(ax,5.33,3.80,'Native target state',size=7.75,weight='bold',ha='center',color=BLUE)
    ax.plot([4.03,6.72],[3.50,3.50],color=BLUE,lw=2.3,zorder=4)
    arrow(ax,(6.72,3.50),(7.60,3.12),color=BLUE,width=1.7)
    arrow(ax,(3.64,2.85),(4.42,2.22),color=VIOLET,width=1.5)
    for i,x in enumerate((4.66,5.10,5.54)):
        ax.add_patch(Circle((x,2.21),.14,facecolor=VIOLET,edgecolor='white',alpha=.90-.18*i,zorder=4))
    ax.plot([4.52,5.68],[2.21,2.21],color=VIOLET,lw=1.15,zorder=3)
    label(ax,5.21,1.81,'Top-K ray evidence',size=7.55,ha='center',weight='bold',color=VIOLET)
    arrow(ax,(5.86,2.21),(7.60,2.73),color=VIOLET,width=1.7)
    ax.plot([7.82,13.71],[2.47,2.47],color=LINE,lw=.9,zorder=1)
    for x,color,name,sub in [(8.08,TEAL,'SS64','geometry'),(9.80,'#718393','Shape','frozen'),
                             (11.52,AMBER,'PBR','appearance')]:
        voxel(ax,x,2.66,scale=1.25,color=color)
        label(ax,x+.31,2.15,name,size=8.05,weight='bold',ha='center',color=color)
        label(ax,x+.31,1.72,sub,size=7.0,ha='center',color=MUTED)
    arrow(ax,(8.72,3.03),(9.63,3.03),width=1.5)
    arrow(ax,(10.44,3.03),(11.35,3.03),width=1.5)
    arrow(ax,(12.15,3.03),(13.00,3.03),width=1.5)
    voxel(ax,13.20,2.66,scale=1.25,color=TEAL)
    label(ax,13.51,2.15,'3D',size=8.05,weight='bold',ha='center')
    label(ax,13.51,1.72,'asset',size=7,ha='center',color=MUTED)
    label(ax,8.40,.86,'source correction',size=7.2,ha='center',color=TEAL)
    label(ax,11.86,.86,'separate Top-1 retargeter',size=7.2,ha='center',color=AMBER)
    ax.plot([7.57,9.21],[1.15,1.15],color=TEAL,lw=1)
    ax.plot([10.63,13.11],[1.15,1.15],color=AMBER,lw=1)
    label(ax,.36,.42,'Target path stays native; matched source evidence updates SS. Shape is frozen; PBR has its own appearance adapter.',
          size=7.2,color=MUTED)
    fig.savefig(output/'paper-overall.pdf',facecolor='white')
    fig.savefig(output/'paper-overall.png',dpi=250,facecolor='white')
    plt.close(fig)


def render_matching(output: Path):
    fig,ax=canvas(3.42,3.15,xmax=6.84,ymax=6.3)
    label(ax,.22,6.02,'(a) Semantic correspondence',size=9.2,weight='bold')
    ax.plot([.22,6.62],[5.76,5.76],color=LINE,lw=.7)
    label(ax,.90,5.43,'Target patch',size=7.2,ha='center',color=BLUE)
    patch_grid(ax,.35,4.27,n=5,cell=.22,fill='#D7E7F1',accent=BLUE,chosen={(2,2)})
    label(ax,5.96,5.43,'Source patches',size=7.2,ha='center',color=VIOLET)
    patch_grid(ax,5.39,4.27,n=5,cell=.22,fill='#E6DDF0',accent=VIOLET,chosen={(1,1),(2,3),(3,1)})
    arrow(ax,(1.51,4.82),(2.01,4.82),color=BLUE,width=1.2)
    card(ax,2.10,4.44,2.50,.76,face=LIGHT_VIOLET,edge=VIOLET)
    label(ax,3.35,4.94,'Omega + D1 matcher',size=7.45,ha='center',weight='bold')
    label(ax,3.35,4.66,'descriptor scores',size=7.0,ha='center')
    arrow(ax,(4.67,4.82),(5.30,4.82),color=VIOLET,width=1.2)
    label(ax,.24,3.75,'Top-K proposals',size=7.7,weight='bold')
    for i,(w,p) in enumerate([(1.8,.56),(1.2,.32),(.68,.18)]):
        y=3.28-i*.38
        ax.add_patch(Circle((.43,y),.09,facecolor=VIOLET,alpha=p+.2,edgecolor='none'))
        ax.add_patch(Rectangle((.69,y-.065),w,.13,facecolor=VIOLET,alpha=p+.28,edgecolor='none'))
        label(ax,2.63,y,f'{i+1}',size=7.1,color=MUTED)
    ax.plot([3.28,3.28],[2.35,3.65],color=LINE,lw=.8)
    label(ax,3.57,3.53,'Members of one ray',size=7.55,weight='bold')
    for i,x in enumerate((3.75,4.42,5.09)):
        ax.add_patch(Circle((x,2.84),.27,facecolor=LIGHT_BLUE,edgecolor=BLUE,lw=.8))
        label(ax,x,2.84,f'$m_{i+1}$',size=7.0,color=BLUE,ha='center')
    arrow(ax,(4.73,2.45),(4.73,2.03),color=VIOLET,width=1.3)
    card(ax,.43,.72,5.9,1.17,face=LIGHT_TEAL,edge=TEAL)
    label(ax,3.38,1.56,'One distribution per source-camera ray',size=7.8,ha='center',weight='bold',color=TEAL)
    label(ax,3.38,1.19,'Member posteriors vote  →  retain Top-K',size=7.45,ha='center')
    label(ax,3.38,.43,'Absent / unsupported matches keep the native source path.',size=7.15,color=MUTED,ha='center')
    fig.savefig(output/'paper-matching.pdf',facecolor='white')
    fig.savefig(output/'paper-matching.png',dpi=260,facecolor='white')
    plt.close(fig)


def render_retargeting(output: Path):
    fig,ax=canvas(3.42,3.15,xmax=6.84,ymax=6.3)
    label(ax,.22,6.02,'(b) Ray-wise evidence retargeting',size=9.2,weight='bold')
    ax.plot([.22,6.62],[5.76,5.76],color=LINE,lw=.7)
    label(ax,.65,5.36,'Matching scores',size=7.45,color=VIOLET)
    for i,(x,h) in enumerate([(0.45,.92),(0.95,.57),(1.45,.34)]):
        ax.add_patch(Rectangle((x,4.23),.3,h,facecolor=VIOLET,alpha=.85-.18*i))
        label(ax,x+.15,4.05,f'$k_{i+1}$',size=7.2,ha='center',color=MUTED)
    arrow(ax,(2.09,4.72),(2.67,4.72),color=VIOLET,width=1.4)
    card(ax,2.70,4.26,1.83,.91,face=LIGHT_TEAL,edge=TEAL)
    label(ax,3.61,4.88,'Native values',size=7.3,weight='bold',ha='center')
    label(ax,3.61,4.55,'weighted  $E_r$',size=7.1,ha='center')
    arrow(ax,(4.57,4.72),(5.07,4.72),color=TEAL,width=1.4)
    card(ax,5.17,4.26,1.28,.91,face=LIGHT_AMBER,edge=AMBER)
    label(ax,5.81,4.85,'Retarget',size=7.4,ha='center',weight='bold')
    label(ax,5.81,4.53,'$E_r-S_r$',size=7.1,ha='center')
    label(ax,.25,3.33,'Shared ray decision',size=7.65,weight='bold')
    for i,x in enumerate((.55,1.38,2.21,3.04)):
        voxel(ax,x,2.4,scale=1.05,color=TEAL,alpha=.83)
        ax.add_patch(Circle((x+.23,3.12),.065,facecolor=VIOLET,edgecolor='none',zorder=6))
    ax.plot([.3,3.9],[3.12,3.12],color=VIOLET,lw=1.15,ls='--',zorder=5)
    label(ax,3.94,3.12,'same $p_r$',size=7.25,color=VIOLET)
    arrow(ax,(5.76,4.2),(5.76,2.57),color=AMBER,width=1.5)
    card(ax,4.44,1.67,2.04,.72,face=LIGHT_AMBER,edge=AMBER)
    label(ax,5.46,2.03,'$u_r[g(E_r-S_r)+R_r]$',size=7.1,ha='center')
    arrow(ax,(5.11,1.55),(3.88,1.17),color=AMBER,width=1.4)
    card(ax,.32,.52,3.55,1.05,face=LIGHT_BLUE,edge=BLUE)
    label(ax,2.09,1.25,'Native MV4 + correction',size=7.15,ha='center',weight='bold')
    label(ax,2.09,.87,'Target / global paths unchanged',size=6.85,ha='center')
    label(ax,5.44,.87,'SS grid',size=7.9,ha='center',color=TEAL,weight='bold')
    fig.savefig(output/'paper-retargeting.pdf',facecolor='white')
    fig.savefig(output/'paper-retargeting.png',dpi=260,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--campaign-audit',type=Path,required=True)
    parser.add_argument('--teaser-sample',required=True)
    args=parser.parse_args()
    setup();args.output_dir.mkdir(parents=True,exist_ok=True)
    render_teaser(args.output_dir,args.campaign_audit,args.teaser_sample)
    render_overall(args.output_dir)
    render_matching(args.output_dir)
    render_retargeting(args.output_dir)


if __name__=='__main__':
    main()
