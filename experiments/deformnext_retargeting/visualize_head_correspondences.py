"""Fixed-panel RGB correspondence diagnostics; GT markers are display only."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image,ImageDraw

from .matching_v2_head import MatchingV2Head
from .train_matching_v2 import load,read_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','panels','labels','file4-mirror','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--wandb',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(0)
    selected=json.loads((a.root/'selected-head.json').read_text())
    state=torch.load(selected['head'],map_location='cpu',weights_only=True)
    model=MatchingV2Head().cuda().eval();model.load_state_dict(state['model'],strict=True)
    args=SimpleNamespace(proposals=a.root/'proposals',labels=a.labels,file4_mirror=a.file4_mirror,topk=selected['topk'])
    lookup={r['sample_id']:r for r in read_rows(a.panels/'head_validation.jsonl')}
    ids=json.loads((a.panels/'panel-32.json').read_text())['sample_ids'][:4]
    canvas=Image.new('RGB',(2048,len(ids)*550),(25,25,25));draw=ImageDraw.Draw(canvas);records=[]
    colors={'exact':(30,230,70),'near1':(255,170,20),'far':(255,55,55)}
    for line,sid in enumerate(ids):
        row=lookup[sid];paths=[row['target_anchor'],*row['source_images']];y0=line*550
        for view,path in enumerate(paths):
            with Image.open(a.file4_mirror/path.removeprefix('/file4/')) as im:
                rgba=np.asarray(im.convert('RGBA').resize((512,512),Image.Resampling.LANCZOS)).astype(float)/255
            rgb=(rgba[...,:3]*rgba[...,3:4]*255).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb),(view*512,y0+28))
            draw.text((view*512+8,y0+4),f"{'M1 target' if view==0 else 'M0 source'} RGB camera {8+view}",fill='white')
        q,source,_,indices,raw,valid,positive,visible,fg,pos,_=load(row,args,'cuda',False)
        candidates=torch.where(fg&visible.any(-1)&valid.any(-1))[0].cpu().tolist()
        candidates=sorted(candidates,key=lambda x:hashlib.sha256(f'{sid}:{x}'.encode()).digest())[:10]
        chunk=torch.tensor(candidates,device='cuda',dtype=torch.long)
        with torch.inference_mode():scores,_=model(q[chunk],source[indices[chunk]],raw[chunk],valid[chunk])
        addresses=indices[chunk].gather(1,scores.argmax(-1)[:,None])[:,0].cpu().tolist()
        examples=[]
        for number,(query,address) in enumerate(zip(candidates,addresses),1):
            view=address//1024;patch=address%1024;px=8+16*(patch%32);py=8+16*(patch//32)
            truth=pos[query].cpu().tolist();vis=visible[query].cpu().tolist()
            exact=any(address==t and v for t,v in zip(truth,vis))
            near=any(view==t//1024 and v and abs(patch%32-t%32)<=1 and abs(patch//32-(t%1024)//32)<=1 for t,v in zip(truth,vis))
            status='exact' if exact else ('near1' if near else 'far');color=colors[status]
            origin=(8+16*(query%32),y0+28+8+16*(query//32));end=((view+1)*512+px,y0+28+py)
            draw.line((origin,end),fill=color,width=2)
            for point in [origin,end]:
                x,y=point;draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2);draw.text((x+5,y-7),str(number),fill=color)
            for address_gt,is_visible in zip(truth,vis):
                if not is_visible:continue
                vx=(address_gt//1024+1)*512+8+16*(address_gt%32);vy=y0+28+8+16*((address_gt%1024)//32)
                draw.rectangle((vx-5,vy-5,vx+5,vy+5),outline=(180,255,200),width=2)
            examples.append({'query':query,'selected_address':address,'status':status})
        draw.text((8,y0+538),f"{row['category']} | line: Top1, squares: visible GT | green exact, orange near1, red far | diagnostic only",fill='white')
        records.append({'sample_id':sid,'queries':examples})
    a.output.mkdir(parents=True,exist_ok=True);image=a.output/'correspondences.png';canvas.save(image)
    (a.output/'manifest.json').write_text(json.dumps({'head_sha256':selected['head_sha256'],
        'selection':'first4 IDs of preregistered panel32; query hash order, no outcome selection',
        'GT_used_for_display_only':True,'records':records},indent=2)+'\n')
    if a.wandb:
        import wandb
        run=wandb.init(entity='youngwoojeon-unist',project='pixal3d-deformnext-retargeting',
            id='ss-matching-v2-correspondence-visuals',resume='allow',dir=str(a.output))
        run.log({'selected-head/Top1-and-GT':wandb.Image(str(image))});run.finish()
    print(json.dumps({'image':str(image),'samples':len(ids),'head_sha256':selected['head_sha256']}),flush=True)


if __name__=='__main__':main()
