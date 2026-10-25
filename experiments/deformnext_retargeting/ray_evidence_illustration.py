"""Select an auditable illustration from saved native probability arrays.

Ground-truth labels select and annotate a diagnostic relation only. No weights
are recomputed, no feature is transported, and no neural model is executed.
"""
import numpy as np

from .ray_attention_utilization import utilization


def first_recoverable_relation(observation, labels):
    """Lexicographically first ray/target relation with Top1 miss and K hit.

    At least two observed member sites are required to illustrate grouping.
    The stored ray IDs mask unobserved target members; their count is NOT the
    size of the entire native ray used by retargeting/source-use pooling.
    """
    utilization(**observation, **labels)
    rays, patches = observation['ray_ids'], observation['patch_ids']
    candidates, valid, prior = (observation[k] for k in ('candidates','candidate_valid','prior'))
    truth = labels['positive_indices']
    visible = (labels['positive_visible'] & observation['source_valid'][truth]
               & labels['target_foreground'][:,None])
    encoded = np.unique(np.concatenate([
        (rays[v,rays[v]>=0]+v*1024)*1024 + patches[rays[v]>=0]
        for v in range(3)]))
    for code in encoded:
        ray, patch = int(code//1024), int(code%1024)
        if not visible[patch].any():
            continue
        members = np.flatnonzero(rays[ray//1024] == ray%1024)
        if len(members) < 2:
            continue
        hit = ((candidates[ray,:,None] == truth[patch,None,:])
               & visible[patch,None,:]).any(-1) & valid[ray]
        top = int(prior[ray].argmax())
        if hit[top] or not hit.any():
            continue
        order = np.argsort(-prior[ray],kind='stable')
        ranks = np.empty_like(order); ranks[order] = np.arange(1,65)
        return dict(source_ray=ray,source_view_index=ray//1024,native_ray_patch=ray%1024,
            target_patch=patch,observed_member_sites=members.tolist(),
            observed_target_member_patches=np.unique(patches[members]).tolist(),
            observed_members_not_entire_native_ray=True,
            candidates=candidates[ray].tolist(),valid=valid[ray].tolist(),
            prior=prior[ray].tolist(),actual_weights=observation['weights'][ray].tolist(),
            top1_slot=top,top1_candidate=int(candidates[ray,top]),
            GT_in_K_slots=np.flatnonzero(hit).tolist(),GT_in_K_ranks=ranks[hit].tolist(),
            GT_mass_prior=float(prior[ray,hit].sum(dtype=np.float64)),
            GT_mass_actual=float(observation['weights'][ray,hit].sum(dtype=np.float64)),
            GT_mass_uniform=float(hit.sum()/valid[ray].sum()),
            use_probability=float(observation['use_probability'][ray]),
            source_map_correction_rms=float(observation['correction_rms'][ray]),
            conditional_selection_not_population_performance=True,
            GT_annotation_did_not_change_actual_weights=True)
    return None


def patch_box(index, width, height):
    """Map the native SS32x32 image-feature patch to an existing RGB image."""
    if (type(index) is not int or not 0 <= index < 1024
            or type(width) is not int or type(height) is not int
            or min(width,height) <= 0 or width%32 or height%32):
        raise ValueError('native32patch address and exactly divisible RGB dimensions required')
    dx,dy = width/32,height/32
    return ((index%32)*dx,(index//32)*dy,dx,dy)


def render_fixed_illustration(selection, images, pooled, output):
    """Render existing RGB/arrays as a standalone scientific PDF and PNG.

    Images are the actual edited target and three source RGB inputs, with
    alpha premultiplied onto black exactly as in the native dataset. The
    diagram shows stored observed member sites, not invented 3D coordinates.
    """
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

    prior = np.asarray(selection['prior'])
    actual = np.asarray(selection['actual_weights'])
    if (len(images) != 4 or not np.array_equal(prior,actual)
            or len(prior) != 64 or not selection['observed_members_not_entire_native_ray']
            or pooled['assets'] != 32 or pooled['relations'] <= 0):
        raise ValueError('actual Fixed64 arrays, four images and explicit DEV32 diagnostic scope required')
    for image in images:
        if (image.ndim != 3 or image.shape[-1] != 3
                or not np.issubdtype(image.dtype,np.floating)
                or not np.isfinite(image).all() or (image<0).any() or (image>1).any()):
            raise ValueError('existing native unit-range premultiplied RGB arrays required')
        patch_box(0,int(image.shape[1]),int(image.shape[0]))
    blue,green,gold,red = '#277DA1','#218C65','#C38B00','#C74440'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig = plt.figure(figsize=(11,7.1),layout='constrained')
    grid = fig.add_gridspec(3,4,height_ratios=[2.5,1.2,1.9])
    axes = [fig.add_subplot(grid[0,i]) for i in range(4)]
    for i,(ax,image) in enumerate(zip(axes,images)):
        ax.imshow(image);ax.set_axis_off()
        ax.set_title('Edited target · camera8' if i==0 else f'Source · camera{i+8}',fontsize=11)

    def box(ax,image,patch,color,linewidth=2.,style='-'):
        x,y,dx,dy = patch_box(int(patch),int(image.shape[1]),int(image.shape[0]))
        ax.add_patch(Rectangle((x,y),dx,dy,fill=False,edgecolor=color,linewidth=linewidth,linestyle=style))

    for patch in selection['observed_target_member_patches']:
        box(axes[0],images[0],patch,gold,.6)
    box(axes[0],images[0],selection['target_patch'],red,2.2)
    view = selection['source_view_index']
    box(axes[view+1],images[view+1],selection['native_ray_patch'],gold,2.,'--')
    winner = selection['top1_candidate']
    box(axes[winner//1024+1],images[winner//1024+1],winner%1024,blue,2.3)
    for slot in selection['GT_in_K_slots']:
        candidate = selection['candidates'][slot]
        box(axes[candidate//1024+1],images[candidate//1024+1],candidate%1024,green,2.5)
    axes[0].text(.01,-.04,'Red: selected target member\nGold: observed member patches',
                 transform=axes[0].transAxes,fontsize=8.5,va='top',color='#444444')
    axes[1].text(.01,-.04,'Blue: actual Top-1 candidate\nGreen: exact GT candidate in Top-K',
                 transform=axes[1].transAxes,fontsize=8.5,va='top',color='#444444')
    axes[2].text(.01,-.04,'Dashed gold: native source-ray patch\nGT is used only for this annotation',
                 transform=axes[2].transAxes,fontsize=8.5,va='top',color='#444444')
    axes[3].text(.01,-.04,f"GT candidate rank(s): {', '.join(map(str,selection['GT_in_K_ranks']))}\n"
        f"Observed GT mass: {100*selection['GT_mass_actual']:.3g}%",transform=axes[3].transAxes,
        fontsize=8.5,va='top',color='#444444')

    group = fig.add_subplot(grid[1,:]);group.set_axis_off();group.set_xlim(0,1);group.set_ylim(0,1)
    members = selection['observed_member_sites'];shown = members[:8]
    group.text(.01,.91,f"One source-camera ray · {len(members)} observed member sites",weight='bold',fontsize=10)
    for i,site in enumerate(shown):
        x = .015+i*.044
        group.add_patch(Rectangle((x,.32),.037,.33,facecolor='#F5F0DC',edgecolor=gold,linewidth=.8))
        group.text(x+.0185,.485,str(site),ha='center',va='center',fontsize=6.8)
    if len(members)>len(shown):group.text(.375,.485,'…',ha='center',va='center',fontsize=14)
    for start,end in [(.40,.49),(.69,.77)]:
        group.add_patch(FancyArrowPatch((start,.49),(end,.49),arrowstyle='-|>',mutation_scale=12,color='#555555'))
    for x,width,text in [(.49,.20,'One 64-candidate\nmatching posterior'),(.77,.22,'Shared evidence E\ncarrier · use · retargeter')]:
        group.add_patch(FancyBboxPatch((x,.20),width,.57,boxstyle='round,pad=.008',
            facecolor='#EAF1F5',edgecolor=blue,linewidth=.9))
        group.text(x+width/2,.49,text,ha='center',va='center',fontsize=9)
    group.text(.01,.02,'Stored IDs show observed target members only; native R/use pooling includes the full ray context.',fontsize=8,color='#555555')

    weight = fig.add_subplot(grid[2,:3]);order = np.argsort(-prior,kind='stable')
    hit = np.zeros(64,dtype=bool);hit[selection['GT_in_K_slots']] = True
    colors = [green if hit[i] else blue if i==selection['top1_slot'] else '#AEBCC6' for i in order]
    weight.bar(np.arange(1,65),prior[order],width=.86,color=colors)
    weight.set(xlim=(0,65),xlabel='Candidate rank by frozen matching probability',ylabel='Probability')
    weight.set_xticks([1,16,32,48,64]);weight.spines[['top','right']].set_visible(False)
    weight.set_title('Actual saved ray posterior · Top-1 misses, Top-K contains a match',fontsize=10,loc='left')
    weight.text(.98,.92,f"GT mass: {100*selection['GT_mass_actual']:.3g}%\n"
        f"Source-use u: {selection['use_probability']:.4f}\n"
        f"Source-map correction RMS: {selection['source_map_correction_rms']:.4f}",
        transform=weight.transAxes,ha='right',va='top',fontsize=8.5)
    for slot in selection['GT_in_K_slots']:
        rank = int(np.flatnonzero(order==slot)[0])+1
        weight.plot(rank,prior[slot],marker='v',color=green,markersize=6)

    mass = fig.add_subplot(grid[2,3]);names=['Uniform','Fixed','Learned']
    values = [100*pooled[k] for k in ('uniform','fixed','learned')]
    mass.bar(names,values,color=['#B4B4B4',blue,'#8F7BB5'],width=.7)
    for i,value in enumerate(values):mass.text(i,value+.13,f'{value:.2f}',ha='center',va='bottom',fontsize=9)
    mass.set_ylim(0,max(values)*1.30);mass.set_ylabel('Exact-GT probability mass (%)',fontsize=9)
    mass.tick_params(axis='x',labelsize=8.5);mass.spines[['top','right']].set_visible(False)
    mass.set_title(f"DEV32 pooled diagnostic\n{pooled['relations']:,} recoverable relations",fontsize=9.5)
    fig.suptitle('Saved ray evidence: candidate coverage, probability allocation and source adaptation',fontsize=13)
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    fig.savefig(output/'dev-ray-evidence.pdf',metadata={'Title':'Actual trained Fixed ray-evidence development diagnostic'})
    fig.savefig(output/'dev-ray-evidence.png',dpi=180)
    plt.close(fig)
