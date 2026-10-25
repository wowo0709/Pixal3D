"""Vector schematic of the implemented working SS path and stage composition.

This reads no weights, features or metrics and makes no performance claim.
Numerical outputs belong outside the source checkout.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


def draw(output, *, selected=False):
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'ps.fonttype':42})
    fig, ax = plt.subplots(figsize=(12.6, 6.6))
    fig.subplots_adjust(left=.012, right=.988, bottom=.015, top=.985)
    ax.set(xlim=(0,18), ylim=(0,9)); ax.axis('off')
    frozen='#EDF0F4'; match='#EEE8F8'; evidence='#E1F2EC'; learned='#FCE9CE'
    ink='#253141'; native='#47748D'; semantic='#7A5AA5'; adapt='#B37325'

    def box(x,y,w,h,label,color=frozen,fontsize=12.2,style='solid'):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=.08',
            facecolor=color,edgecolor=ink,linewidth=.9,linestyle=style,zorder=2)
        ax.add_patch(patch); ax.text(x+w/2,y+h/2,label,ha='center',va='center',
            fontsize=fontsize,color=ink,linespacing=1.28,zorder=3)

    def arrow(points,color=ink,style='solid'):
        for a,b in zip(points[:-2],points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.35,ls=style,zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
            mutation_scale=11,linewidth=1.35,color=color,linestyle=style,zorder=1))

    ax.text(.2,8.67,'A   SS source adaptation: '+('selected Fixed Top-K' if selected else 'working Fixed Top-K candidate'),
        fontsize=15,fontweight='bold',color=ink)
    box(.2,7.0,2.8,1.05,'Edited target RGB\n+ 3 source RGBs\nCalibrated cameras')
    box(3.6,7.0,3.0,1.05,'Frozen native encoder\nDINOv3 feature maps\nTarget T; sources S')
    box(7.3,7.0,3.7,1.05,'Native MV4 carrier\nOriginal target contribution\nGlobal tokens unchanged')
    box(14.4,7.0,3.0,1.05,'Native SS flow\nFrozen base + LoRA\nConditioning C',learned)
    arrow([(3.03,7.525),(3.55,7.525)],native)
    arrow([(6.63,7.525),(7.25,7.525)],native)
    arrow([(11.03,7.525),(13.10,7.525)],native)
    plus=Circle((13.4,7.525),.25,facecolor='white',edgecolor=ink,lw=1.1,zorder=3)
    ax.add_patch(plus);ax.text(13.4,7.525,'+',ha='center',va='center',fontsize=20,color=ink,zorder=4)
    arrow([(13.67,7.525),(14.35,7.525)],native)

    box(.2,4.9,2.8,1.05,'Frozen VGGT-Omega\nSelected epoch 7\nMatching descriptors',match)
    box(3.6,4.9,3.0,1.05,'Frozen D1 matcher\nRaw semantic Top-64\nPer-target-patch scores',match)
    box(7.3,4.9,3.7,1.05,'Source-camera ray groups\nDistinct target members\nMix posteriors; retain 64',match)
    box(11.7,4.9,3.7,1.05,'Gather native source values\n'+r'$E_r=\sum p_r(k)v_k$'+'\nIdentity values',evidence,12.5)
    arrow([(1.6,6.97),(1.6,6.0)],semantic)
    arrow([(3.03,5.425),(3.55,5.425)],semantic)
    arrow([(6.63,5.425),(7.25,5.425)],semantic)
    arrow([(11.03,5.425),(11.65,5.425)],semantic)
    ax.text(11.35,5.78,r'$p_r$',ha='center',fontsize=12,color=semantic)

    # Matching keys and untransformed native values are distinct streams.
    arrow([(5.1,6.97),(5.1,6.4),(13.55,6.4),(13.55,6.0)],native)
    ax.text(9.55,6.46,'Native DINO values',ha='center',va='bottom',fontsize=11.3,color=native)
    arrow([(6.95,6.4),(6.95,3.325),(7.25,3.325)],native)
    ax.text(6.78,3.95,r'$T_r, S_r$',rotation=90,ha='center',va='center',fontsize=12,color=native)

    box(.2,2.8,5.8,1.05,'No evidence: retain native MV4\n'
        'Cold initialization: g = 0 and R = 0\n'
        'Fixed path has no additional Q/K or V/O head',frozen,11.8,'dashed')
    box(7.3,2.8,3.7,1.05,'Source use + retargeting\n'
        +r'$R_r=R_\phi([E_r-S_r,T_r])$'+'\n'
        +r'$\Delta_r=u_r\{g(E_r-S_r)+R_r\}$',learned,12.1)
    box(11.7,2.8,3.7,1.05,'Native bilinear projection\n'
        'Sum source deltas / 4\nSource correction only',evidence,12.3)
    arrow([(13.55,4.87),(13.55,4.4),(9.15,4.4),(9.15,3.90)],semantic)
    arrow([(11.03,3.325),(11.65,3.325)],adapt)
    arrow([(15.43,3.325),(15.9,3.325),(15.9,6.7),(13.4,6.7),(13.4,7.23)],adapt)

    ax.plot([.2,17.4],[2.35,2.35],color='#C3C9D0',lw=.8)
    ax.text(.2,2.05,'B   '+('Frozen stage composition' if selected else 'Actual stage composition under study'),fontsize=14,fontweight='bold',color=ink)
    box(.2,.72,2.8,.88,'SS64\nLoRA + source R',learned,12.0)
    box(3.6,.72,3.0,.88,'Shape512\nFrozen',frozen,12.5)
    box(7.3,.72,3.0,.88,'Shape1024\nFrozen',frozen,12.5)
    box(11.0,.72,3.5,.88,'PBR1024\nLoRA + separate Top-1 R',learned,12.0)
    box(15.2,.72,2.2,.88,'Textured\n3D output',evidence,12.2)
    for start,stop in [(3.03,3.55),(6.63,7.25),(10.33,10.95),(14.53,15.15)]:
        arrow([(start,1.16),(stop,1.16)])
    ax.text(.2,.27,'Same four RGB views / cameras condition the native stages; PBR uses its own source conditioner.',
        fontsize=11.2,color=ink)
    stem='selected-pipeline-overview' if selected else 'working-pipeline-overview'
    fig.savefig(output/(stem+'.pdf'),metadata={'Title':'Implemented '+('selected' if selected else 'working')+' Fixed SS evidence path and stage composition'})
    fig.savefig(output/(stem+'.png'),dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--selected',action='store_true')
    args=parser.parse_args()
    draw(args.output_dir,selected=args.selected)


if __name__=='__main__':
    main()
