"""Draw a compact, image-grounded overview of the frozen editing pipeline.

The four input images are verified against the archived teaser provenance.
They illustrate the inputs only; the diagram makes no quantitative claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np
from PIL import Image

INK = '#243244'
MUTED = '#637387'
BLUE = '#3679ae'
PURPLE = '#7959a5'
TEAL = '#12877d'
AMBER = '#c78337'
LIGHT = '#dbe2e8'


def text(ax, x, y, s, size=7.5, color=INK, weight='normal', ha='left'):
    return ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                   ha=ha, va='center', zorder=10)


def arrow(ax, a, b, color=INK, lw=1.35):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=8,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=9))


def tile(ax, x, y, w, h, rgb, border=LIGHT):
    ax.imshow(rgb, extent=[x,x+w,y,y+h], interpolation='lanczos', zorder=2)
    ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=border,lw=.65,zorder=3))


def load_input(provenance):
    rows = json.loads(provenance.read_text())['input_RGB_SHA256']
    if len(rows) != 4:
        raise ValueError('target and three source images required')
    images=[]
    for path, expected in rows.items():
        path=Path(path)
        if not path.is_absolute():
            path=provenance.parent/path
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'input SHA mismatch: {path}')
        images.append(np.array(Image.open(path).convert('RGBA')))
    union=np.logical_or.reduce([a[:,:,3]>8 for a in images])
    yy,xx=np.where(union)
    pad=25
    left=max(0,xx.min()-pad);right=min(512,xx.max()+pad+1)
    top=max(0,yy.min()-pad);bottom=min(512,yy.max()+pad+1)
    width=max(right-left,bottom-top)
    cx=(left+right)/2;cy=(top+bottom)/2
    left=int(max(0,min(512-width,cx-width/2)));top=int(max(0,min(512-width,cy-width/2)))
    out=[]
    for a in images:
        a=a[top:top+width,left:left+width]
        rgb=a[:,:,:3].astype(np.float32)/255
        alpha=a[:,:,3:4].astype(np.float32)/255
        out.append(rgb*alpha + .055*(1-alpha))
    return out, {'crop_xywh': [int(left),int(top),int(width),int(width)], 'input_sha256':rows}


def draw_ray_lattice(ax):
    # Wireframe volume, viewed obliquely.  A single source-camera ray passes
    # several voxel sites; the posterior is shared, not recomputed per site.
    front=np.array([[7.05,2.37],[8.63,2.37],[8.63,4.32],[7.05,4.32]])
    depth=np.array([.52,.37])
    back=front+depth
    for a,b in ((front,back),):
        for poly in (a,b):
            ax.add_patch(Polygon(poly,closed=True,fill=False,edgecolor='#9db3bb',lw=.72,zorder=3))
        for i in range(4):
            ax.plot([a[i,0],b[i,0]],[a[i,1],b[i,1]],color='#9db3bb',lw=.72,zorder=3)
    for f in (.25,.5,.75):
        ax.plot([7.05+1.58*f]*2,[2.37,4.32],color='#d4dfe3',lw=.55,zorder=2)
        ax.plot([7.05,8.63],[2.37+1.95*f]*2,color='#d4dfe3',lw=.55,zorder=2)
    ray_y=3.37
    ax.plot([6.22,9.20],[ray_y-.32,ray_y+.38],color=PURPLE,lw=1.65,zorder=5)
    xs=np.linspace(7.23,8.94,5)
    for i,x in enumerate(xs):
        y=ray_y-.32+(x-6.22)*.70/2.98
        ax.add_patch(Circle((x,y),.075,facecolor=PURPLE,edgecolor='white',lw=.5,zorder=6))
    text(ax,7.82,5.08,'one posterior per source ray',7.4,PURPLE,'bold',ha='center')
    text(ax,7.82,2.05,'source-view ray through SS grid',7.1,MUTED,ha='center')
    return (8.92,3.38)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-provenance',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    images,record=load_input(args.input_provenance)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.05,3.05))
    fig.subplots_adjust(0,0,1,1)
    ax.set_xlim(0,14.1);ax.set_ylim(0,6.1);ax.axis('off')
    ax.set_aspect('equal',adjustable='box')
    text(ax,.23,5.78,'Input views',8.25,INK,'bold')
    text(ax,4.9,5.78,'Evidence transfer',8.25,INK,'bold')
    text(ax,10.05,5.78,'Native 3D generation',8.25,INK,'bold')
    ax.plot([.18,13.88],[5.53,5.53],color=LIGHT,lw=.85)
    # The same crop across all images preserves relative framing.
    tile(ax,.22,3.35,1.76,1.76,images[0],BLUE)
    for j,x in enumerate((2.08,2.98,3.88)):
        tile(ax,x,3.68,.84,.84,images[j+1],PURPLE)
    text(ax,1.1,3.04,'edited target $I_t$',7.25,BLUE,'bold',ha='center')
    text(ax,3.27,3.04,'source $I_{1:3}$',7.25,PURPLE,'bold',ha='center')
    text(ax,.22,2.54,'calibrated cameras; distinct shapes',6.95,MUTED)
    # Target route is carried as a distinct native conditioning stream.
    ax.plot([.22,4.5],[1.55,1.55],color=BLUE,lw=1.25)
    ax.add_patch(Circle((.36,1.55),.095,facecolor=BLUE,edgecolor='none'))
    text(ax,.55,1.84,'native target path',7.15,BLUE,'bold')
    arrow(ax,(4.25,1.55),(5.03,1.55),BLUE)
    # Matching stays separate from the original feature carrier.
    arrow(ax,(4.78,4.14),(5.17,4.14),PURPLE)
    text(ax,5.18,4.43,'$\Omega$ + D1',7.5,PURPLE,'bold')
    for k,(x,h) in enumerate(((5.28,.62),(5.53,.43),(5.78,.28))):
        ax.add_patch(Rectangle((x,3.34),.15,h,facecolor=PURPLE,alpha=.82-.13*k,zorder=4))
    text(ax,5.62,3.10,'Top-$K$',7.0,PURPLE,ha='center')
    arrow(ax,(6.00,3.62),(6.43,3.62),PURPLE)
    draw_ray_lattice(ax)
    # Clean stage strip: no detailed modules in the overview.
    ax.plot([9.4,9.4],[.65,5.17],color=LIGHT,lw=.8)
    text(ax,9.77,4.86,'SS64',8.2,TEAL,'bold')
    text(ax,11.04,4.86,'LoRA + source correction',6.95,MUTED)
    ax.plot([9.78,13.69],[4.54,4.54],color=TEAL,lw=2.0)
    text(ax,9.77,4.04,'Shape512  /  Shape1024',7.8,INK,'bold')
    text(ax,9.77,3.70,'frozen cascade',7.0,MUTED)
    ax.plot([9.78,13.69],[3.42,3.42],color='#9babb7',lw=2.0)
    text(ax,9.77,2.90,'PBR1024',8.2,AMBER,'bold')
    text(ax,11.18,2.90,'Top-1 source adapter',7.0,MUTED)
    ax.plot([9.78,13.69],[2.57,2.57],color=AMBER,lw=2.0)
    text(ax,9.77,1.91,'Textured 3D asset',8.0,INK,'bold')
    text(ax,9.77,1.55,'target geometry + source appearance',6.9,MUTED)
    arrow(ax,(9.04,3.42),(9.63,3.42),TEAL)
    ax.plot([.18,13.88],[.72,.72],color=LIGHT,lw=.75)
    text(ax,.23,.41,'Native target conditioning retained',7.0,BLUE)
    text(ax,5.35,.41,'source evidence retargeted along rays',7.0,PURPLE)
    text(ax,11.2,.41,'same 4-view carrier',7.0,TEAL)
    fig.savefig(args.output_dir/'paper-overall-v2.pdf',facecolor='white')
    fig.savefig(args.output_dir/'paper-overall-v2.png',dpi=260,facecolor='white')
    plt.close(fig)
    (args.output_dir/'paper-overall-v2-provenance.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
