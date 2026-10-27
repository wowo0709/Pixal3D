"""Build the fair-protocol teaser from the selected TEST case and scored renders.

All prediction tiles use source-M0 post-generation alignment. The GT tiles and
input photographs are fetched by their frozen manifests. Every PNG is SHA
checked against its renderer receipt before the figure is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

OFFICIAL_METHOD = 'pixal3d_official_rgb1_source_m0'
OURS_METHODS = {'V1':'ours_v1_source_m0','SELECTED':'ours_selected_source_m0'}
GT_ROOT = '/home/youngwoo/retargeting-unified-test508/gt-accessible-root'
REFS = '/home/youngwoo/retargeting-unified-test508/references.jsonl'
TARGET8 = '/home/youngwoo/retargeting-unified-test508/target8-gt-jobs.jsonl'


def sha(path: Path) -> str:
    state=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            state.update(block)
    return state.hexdigest()


def remote_row(host:str,port:int,path:str,sid:str)->dict:
    code=('import json,sys; from pathlib import Path;'
          'rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()];'
          'hit=[r for r in rows if r["sample_id"]==sys.argv[2]];'
          'assert len(hit)==1;print(json.dumps(hit[0]))')
    argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p',str(port),host,
          ' '.join(shlex.quote(x) for x in ('python3','-c',code,path,sid))]
    return json.loads(subprocess.run(argv,check=True,text=True,capture_output=True,timeout=30).stdout)


def fetch(host:str,port:int,source:str,dest:Path,socket:Path|None=None)->None:
    dest.parent.mkdir(parents=True,exist_ok=True)
    ssh='ssh '+(f'-S {socket} ' if socket and socket.is_socket() else '')+f'-o BatchMode=yes -p {port}'
    subprocess.run(['rsync','-a','-e',ssh,f'{host}:{source}',str(dest)],check=True)


def staged_render(host:str,port:int,source:str,stage:Path,protocol:str,
                  socket:Path|None=None)->dict:
    image=stage/(hashlib.sha256(source.encode()).hexdigest()[:16]+'.png')
    receipt=image.with_suffix('.receipt.json')
    fetch(host,port,source,image,socket)
    fetch(host,port,str(Path(source).with_suffix('.receipt.json')),receipt,socket)
    row=json.loads(receipt.read_text())
    if (row['schema']!='deformnext-render-receipt-v1'
            or row['protocol_sha256']!=protocol
            or row['output_sha256']!=sha(image)):
        raise ValueError(f'PNG/receipt/protocol mismatch: {source}')
    return {'path':image,'sha256':row['output_sha256'],
            'receipt_sha256':sha(receipt),'source':source,
            'camera_sha256':row['camera_sha256'],'job_sha256':row['job_sha256']}


def locate(plan:Path,sid:str,method:str,camera:int,archive_root:str)->tuple[str,str]:
    hits=[]
    for chunk in sorted(plan.glob('chunk[0-9][0-9][0-9]')):
        if not (chunk/'completed.json').is_file():
            raise ValueError(f'incomplete render chunk: {chunk}')
        for line in (chunk/'jobs.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['sample_id']==sid and row['method']==method and row['camera_id']==camera:
                hits.append((str(Path(archive_root)/chunk.name/row['output_rgba']),
                             row['protocol_sha256']))
    if len(hits)!=1:raise ValueError(f'exactly one scored render job required: {method}/{sid}/{camera}')
    return hits[0]


def input_images(campaign:Path,sid:str)->tuple[list[np.ndarray],dict]:
    d=json.loads(campaign.read_text())
    matching=[e for e in d['plans'] if e['sample_id']==sid and e['arm']=='V1']
    if len(matching)!=1:raise ValueError('one frozen V1 input plan required')
    shape=json.loads(Path(matching[0]['shape_plan']).read_text())
    paths=[Path(x) for x in shape['RGB_paths']]
    if len(paths)!=4 or any(not p.is_file() for p in paths):
        raise ValueError('edited target and three source RGB/A inputs required')
    arrays=[np.asarray(Image.open(p).convert('RGBA')) for p in paths]
    if len({a.shape for a in arrays})!=1:raise ValueError('input image shapes differ')
    return arrays,{str(p):sha(p) for p in paths}


def crop_union(arrays:list[np.ndarray],pad_fraction:float=.08)->tuple[int,int,int,int]:
    if len({a.shape[:2] for a in arrays})!=1:raise ValueError('render image dimensions differ')
    mask=np.logical_or.reduce([a[:,:,3]>8 for a in arrays])
    ys,xs=np.where(mask)
    if len(xs)==0:raise ValueError('all visual masks empty')
    h,w=mask.shape
    side=max(int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1))
    side=min(max(h,w),max(16,int(side*(1+2*pad_fraction))))
    cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2
    x=int(max(0,min(w-side,round(cx-side/2))))
    y=int(max(0,min(h-side,round(cy-side/2))))
    return x,y,side,side


def rgb_crop(a:np.ndarray,box:tuple[int,int,int,int])->np.ndarray:
    x,y,w,h=box;b=a[y:y+h,x:x+w].astype(np.float32)/255
    return b[:,:,:3]*b[:,:,3:4]


def draw(out:Path,choice:dict,inputs:list[np.ndarray],rendered:dict,
         provenance:dict)->None:
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.05,4.56));fig.subplots_adjust(0,0,1,1)
    ax.set_xlim(0,14.1);ax.set_ylim(0,9.12);ax.set_aspect('equal');ax.axis('off')
    ink='#243244';muted='#637387';blue='#3679ae';purple='#7959a5';teal='#12877d'
    def text(x,y,s,size=7.2,color=ink,weight='normal',ha='left'):
        ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va='center')
    def tile(x,y,w,img,edge='#dbe2e8'):
        ax.imshow(img,extent=[x,x+w,y,y+w],interpolation='lanczos')
        ax.add_patch(Rectangle((x,y),w,w,fill=False,edgecolor=edge,lw=.7))
    input_box=crop_union(inputs,.09)
    text(.25,8.82,'Edited target',7.5,blue,'bold')
    text(2.20,8.82,'Undeformed source views',7.5,purple,'bold')
    for i,x in enumerate((.25,2.20,3.80,5.40)):
        tile(x,7.26,1.42,rgb_crop(inputs[i],input_box),blue if i==0 else purple)
    text(7.25,8.35,'Target shape  +  source appearance',8.2,ink,'bold')
    text(7.25,7.90,'Calibrated cameras; object GT scale not supplied',7.0,muted)
    text(7.25,7.49,'Predictions aligned to input M0 after generation',7.0,muted)
    ax.plot([.23,13.86],[7.02,7.02],color='#dbe2e8',lw=.7)
    headers=['M1 GT','Official RGB1','Vanilla V1','Ours selected']
    xs=[1.20,4.27,7.34,10.41];w=2.56
    for x,h in zip(xs,headers):text(x+w/2,6.73,h,7.65,teal if h=='Ours selected' else ink,'bold','center')
    render_box=crop_union(list(rendered.values()),.06)
    for row,(camera,y) in enumerate(((8,3.94),(9,.97))):
        text(.25,y+1.55,'Target' if camera==8 else 'Source',7.2,ink,'bold')
        text(.25,y+1.23,'view 8' if camera==8 else 'view 9',6.7,muted)
        for x,name in zip(xs,('GT','official','V1','Selected')):
            tile(x,y,w,rgb_crop(rendered[name,camera],render_box),
                 teal if name=='Selected' else '#dbe2e8')
    text(7.02,.44,'All predictions: identical Cycles cameras, source-M0 alignment, and shared display crop.',6.65,muted,ha='center')
    out.mkdir(parents=True,exist_ok=False)
    fig.savefig(out/'paper-teaser.pdf',facecolor='white')
    fig.savefig(out/'paper-teaser.png',dpi=240,facecolor='white')
    plt.close(fig)
    provenance['selection']=choice
    provenance['input_display_crop_xywh']=list(map(int,input_box))
    provenance['render_display_crop_xywh']=list(map(int,render_box))
    provenance['figure_pdf_sha256']=sha(out/'paper-teaser.pdf')
    (out/'paper-teaser-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    caption=(r'\caption{Common-success TEST qualitative comparison on '
             +str(choice['common_samples'])
             +r' edits. The edited target RGB and three undeformed source RGBs '
             +r'are the inputs to our two four-view pipelines; official Pixal3D '
             +r'RGB1 receives the edited target RGB and calibrated camera only. '
             +r'The upper prediction row uses target camera 8 and the lower row '
             +r'uses source camera 9. The M1 reference retains the source '
             +r'material. Every prediction uses the same Cycles cameras and '
             +r'a post-generation source-M0-only Sim(3) fit, with no target mesh '
             +r'used for registration. The case follows the prespecified '
             +r'paired LPIPS and target mask-IoU median selection rule; it '
             +r'illustrates a case rather than a population success rate.}'
             +'\n')
    (out/'paper-teaser-caption.tex').write_text(caption)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--common-manifest',type=Path,required=True)
    p.add_argument('--ours-evaluation',type=Path,required=True)
    p.add_argument('--ours-target8',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--official-plan',type=Path,required=True)
    p.add_argument('--ours-plan',type=Path,required=True)
    p.add_argument('--official-archive',required=True)
    p.add_argument('--ours-archive',required=True)
    p.add_argument('--staging-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--node16-host',default='youngwoo@n16.unist.info')
    p.add_argument('--file4-host',default='rvi@10.0.12.170')
    p.add_argument('--ssh-port',type=int,default=55555)
    p.add_argument('--file4-control-socket',type=Path,default=Path('/tmp/file4-ctl-v2'))
    args=p.parse_args()
    choice=json.loads(args.selection.read_text())
    if choice['status']!='PASS_FAIR_COMMON_TEST_TEASER_CASE_SELECTED':
        raise ValueError('final common-success teaser selection required')
    common=json.loads(args.common_manifest.read_text())
    sid=choice['sample_id'];inputs,input_hashes=input_images(args.campaign,sid)
    if (common['status']!='PASS_THIRTEEN_METHOD_COMMON_SUCCESS_BINDING'
            or common['samples']!=choice['common_samples']
            or sid not in common['sample_ids']
            or sha(args.common_manifest)!=choice['common_manifest_sha256']):
        raise ValueError('teaser choice and final common-success population differ')
    if sha(args.ours_target8)!=choice['target8_sha256']:
        raise ValueError('teaser choice and target-camera evaluation differ')
    result_hashes={}
    for path in args.ours_evaluation.glob('*-results.json'):
        method=json.loads(path.read_text())['method']
        if method in choice['evaluation_result_sha256']:
            if method in result_hashes:
                raise ValueError('duplicate evaluator result for teaser: '+method)
            result_hashes[method]=sha(path)
    if result_hashes!=choice['evaluation_result_sha256']:
        raise ValueError('teaser choice and source-camera evaluation differ')
    reference=remote_row(args.node16_host,args.ssh_port,REFS,sid)
    target=remote_row(args.node16_host,args.ssh_port,TARGET8,sid)
    if (reference['protocol_sha256']!=target['protocol_sha256']
            or target['camera_id']!=8):
        raise ValueError('GT target and source camera contract differs')
    protocol=reference['protocol_sha256']
    source_views={v['camera_id']:v for v in reference['views']}
    if 9 not in source_views:raise ValueError('source camera9 GT missing')
    sources={
        ('GT',8):('/'+target['output_rgba'].lstrip('/'),args.node16_host,None),
        ('GT',9):(str(Path(GT_ROOT)/source_views[9]['rgba']),args.node16_host,None),
    }
    for arm,method in OURS_METHODS.items():
        for camera in (8,9):
            path,p_sha=locate(args.ours_plan,sid,method,camera,args.ours_archive)
            if p_sha!=protocol:raise ValueError('Ours render protocol differs')
            sources[(arm,camera)]=(path,args.file4_host,args.file4_control_socket)
    for camera in (8,9):
        path,p_sha=locate(args.official_plan,sid,OFFICIAL_METHOD,camera,args.official_archive)
        if p_sha!=protocol:raise ValueError('official render protocol differs')
        sources[('official',camera)]=(path,args.file4_host,args.file4_control_socket)
    rendered={};records={}
    args.staging_dir.mkdir(parents=True,exist_ok=True)
    for key,(path,host,socket) in sources.items():
        rec=staged_render(host,args.ssh_port,path,args.staging_dir,protocol,socket)
        records[f'{key[0]}/camera{key[1]}']={k:v for k,v in rec.items() if k!='path'}
        rendered[key]=np.asarray(Image.open(rec['path']).convert('RGBA'))
    if len({a.shape for a in rendered.values()})!=1:
        raise ValueError('common Cycles image shapes differ')
    provenance={'status':'PASS_FAIR_COMMON_TEST_TEASER_RENDER_SHA_VERIFIED',
                'sample_id':sid,'protocol_sha256':protocol,
                'selection_sha256':sha(args.selection),
                'input_RGB_SHA256':input_hashes,'render_records':records,
                'shared_crop':True,'raw_or_source_m0':'source_m0'}
    draw(args.output_dir,choice,inputs,rendered,provenance)
    print(json.dumps({'status':provenance['status'],'sample_id':sid,
                      'renders':len(rendered),'output':str(args.output_dir)}))

if __name__=='__main__':main()
