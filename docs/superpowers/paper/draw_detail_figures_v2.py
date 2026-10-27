"""Draw schematic, image-grounded one-column method details.

Highlighted patches and posterior values are illustrative. The input RGB pixels
are SHA-verified archived images; the diagrams do not represent measured
matching accuracy or generation quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np
from draw_pipeline_overview_v2 import load_input

INK='#243244';MUTED='#637387';BLUE='#3679ae';PURPLE='#7959a5';TEAL='#12877d';AMBER='#c78337';LIGHT='#dbe2e8'


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'pdf.fonttype':42,'ps.fonttype':42})


def canvas():
    fig,ax=plt.subplots(figsize=(3.42,2.9));fig.subplots_adjust(0,0,1,1)
    ax.set_xlim(0,6.84);ax.set_ylim(0,5.8);ax.set_aspect('equal');ax.axis('off')
    return fig,ax


def label(ax,x,y,s,size=7.2,color=INK,weight='normal',ha='left'):
    ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va='center',zorder=10)


def arr(ax,a,b,color=INK,lw=1.1):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,color=color,
                                 lw=lw,shrinkA=0,shrinkB=0,zorder=8))


def tile(ax,x,y,w,img,edge):
    ax.imshow(img,extent=[x,x+w,y,y+w],interpolation='lanczos',zorder=2)
    ax.add_patch(Rectangle((x,y),w,w,fill=False,edgecolor=edge,lw=.7,zorder=3))


def grid(ax,x,y,w,h,n=4,m=4,edge='#aabdc3'):
    ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=edge,lw=.7))
    for i in range(1,n):ax.plot([x+i*w/n]*2,[y,y+h],color=edge,lw=.5)
    for i in range(1,m):ax.plot([x,x+w],[y+i*h/m]*2,color=edge,lw=.5)


def matching(output,imgs):
    fig,ax=canvas()
    label(ax,.2,5.53,'A   Semantic proposal aggregation',8.1,INK,'bold')
    ax.plot([.2,6.64],[5.31,5.31],color=LIGHT,lw=.75)
    tile(ax,.25,3.58,1.48,imgs[0],BLUE)
    tile(ax,5.11,3.58,1.48,imgs[2],PURPLE)
    label(ax,.99,3.36,'edited target',6.65,BLUE,'bold','center')
    label(ax,5.85,3.36,'original source',6.65,PURPLE,'bold','center')
    # Patch markers only illustrate the support relation.
    for p in ((.74,4.50),(1.06,4.30),(1.30,4.04)):
        ax.add_patch(Circle(p,.066,facecolor=BLUE,edgecolor='white',lw=.35,zorder=5))
    for p in ((5.48,4.57),(5.83,4.33),(6.10,4.15)):
        ax.add_patch(Circle(p,.066,facecolor=PURPLE,edgecolor='white',lw=.35,zorder=5))
    arr(ax,(1.84,4.36),(2.38,4.36),BLUE)
    label(ax,3.42,4.73,'$\Omega$ descriptors + D1 scores',7.0,INK,'bold','center')
    for i,(x,w) in enumerate(((2.47,1.54),(2.47,1.25),(2.47,.81))):
        y=4.36-i*.22
        ax.plot([x,x+w],[y,y],color=PURPLE,lw=2.2,alpha=1-.16*i)
    arr(ax,(4.28,4.13),(5.01,4.13),PURPLE)
    ax.plot([.2,6.64],[3.07,3.07],color=LIGHT,lw=.7)
    label(ax,.28,2.81,'Target members on one source-camera ray',6.9,INK,'bold')
    vals=np.array([[.55,.13,.08,.15,.09],[.12,.45,.18,.10,.15],[.13,.10,.49,.14,.14]])
    cols=5;x0=1.55;y0=1.24;dx=.39;dy=.40
    for i in range(3):
        label(ax,1.28,y0+(2-i)*dy+.17,f'$m_{i+1}$',7.1,BLUE,ha='right')
        for k in range(cols):
            x=x0+k*dx;y=y0+(2-i)*dy
            alpha=.15+.75*vals[i,k]/.55
            ax.add_patch(Rectangle((x,y),.34,.33,facecolor=PURPLE,alpha=alpha,
                                   edgecolor='white',lw=.4,zorder=3))
    label(ax,3.90,2.33,'mean vote',6.9,PURPLE,'bold')
    arr(ax,(3.55,1.81),(4.18,1.81),PURPLE)
    mean=vals.mean(axis=0)
    for k in range(cols):
        x=4.37+k*.39
        ax.add_patch(Rectangle((x,1.58),.31,.31,facecolor=TEAL,
                               alpha=.2+.7*mean[k]/mean.max(),edgecolor='white',lw=.4))
    label(ax,5.30,1.29,'retain Top-$K$; normalize',6.8,TEAL,'bold','center')
    label(ax,.27,.55,'No supported member  →  native source features',6.65,MUTED)
    label(ax,.27,.20,'Patch dots / weights are schematic; image pixels are archived.',6.1,MUTED)
    fig.savefig(output/'paper-matching-v2.pdf',facecolor='white')
    fig.savefig(output/'paper-matching-v2.png',dpi=270,facecolor='white')
    plt.close(fig)


def retargeting(output):
    fig,ax=canvas()
    label(ax,.2,5.53,'B   Ray-shared source correction',8.1,INK,'bold')
    ax.plot([.2,6.64],[5.31,5.31],color=LIGHT,lw=.75)
    # Source-view ray through the conditioning lattice.
    grid(ax,.63,2.96,2.20,1.55,n=4,m=3)
    ax.plot([.30,3.02],[3.50,4.12],color=PURPLE,lw=1.55,zorder=5)
    for x in (1.0,1.48,1.96,2.44):
        y=3.50+(x-.30)*.62/2.72
        ax.add_patch(Circle((x,y),.068,facecolor=PURPLE,edgecolor='white',lw=.4,zorder=6))
    label(ax,1.73,4.81,'one $p_r$ for all sites',7.1,PURPLE,'bold','center')
    label(ax,1.73,2.70,'source-view ray $r$',6.75,MUTED,ha='center')
    # Distribution and weighted native value, expressed as a small quantitative glyph.
    label(ax,3.43,4.75,'proposals $k$',6.9,PURPLE,'bold')
    for i,(x,h) in enumerate(((3.52,.86),(3.96,.58),(4.40,.35),(4.84,.23))):
        ax.add_patch(Rectangle((x,3.55),.23,h,facecolor=PURPLE,alpha=.88-.12*i))
    label(ax,5.90,4.17,'native DINO',6.85,TEAL,'bold','center')
    label(ax,5.90,3.82,'values $v_k$',6.85,TEAL,ha='center')
    arr(ax,(5.16,3.94),(5.43,3.94),TEAL)
    ax.plot([.2,6.64],[2.39,2.39],color=LIGHT,lw=.7)
    label(ax,.35,2.06,'$E_r=\sum_k p_r(k)v_k$',7.5,TEAL,'bold')
    arr(ax,(2.45,2.04),(2.83,2.04),TEAL)
    label(ax,2.95,2.06,'$E_r-S_r$',7.55,AMBER,'bold')
    arr(ax,(4.10,2.04),(4.48,2.04),AMBER)
    label(ax,4.61,2.06,'$u_r[g(\cdot)+R_r]$',7.05,AMBER,'bold')
    ax.plot([.35,6.48],[1.58,1.58],color=LIGHT,lw=.65)
    label(ax,.35,1.30,'native MV4 carrier',7.05,BLUE,'bold')
    ax.plot([.35,2.57],[1.05,1.05],color=BLUE,lw=1.55)
    arr(ax,(2.57,1.05),(3.12,1.05),BLUE)
    ax.add_patch(Circle((3.32,1.05),.17,facecolor='white',edgecolor=INK,lw=.85))
    label(ax,3.32,1.05,'+',8.7,INK,ha='center')
    arr(ax,(4.02,1.95),(3.52,1.24),AMBER)
    arr(ax,(3.50,1.05),(4.03,1.05),INK)
    label(ax,4.18,1.05,'SS64 conditioned grid',7.05,INK,'bold')
    label(ax,.35,.36,'Target stays native; unsupported rays add no correction.',6.35,MUTED)
    fig.savefig(output/'paper-retargeting-v2.pdf',facecolor='white')
    fig.savefig(output/'paper-retargeting-v2.png',dpi=270,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-provenance',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    imgs,info=load_input(args.input_provenance)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    setup();matching(args.output_dir,imgs);retargeting(args.output_dir)
    (args.output_dir/'detail-figures-provenance.json').write_text(json.dumps(info,indent=2)+'\n')

if __name__=='__main__':main()
