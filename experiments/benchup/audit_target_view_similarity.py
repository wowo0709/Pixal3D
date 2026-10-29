"""Report source-view retrieval as a non-authoritative target-camera diagnostic."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',choices=['0','1','2','3'],required=True);p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu:raise ValueError('explicit physical GPU required')
    import torch
    from torchvision import transforms
    from open_clip.openai import load_openai_model
    from open_clip.constants import OPENAI_DATASET_MEAN,OPENAI_DATASET_STD
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    if len(rows)!=109 or len({r['sample_id'] for r in rows})!=109:raise ValueError('frozen 109 samples required')
    registry=json.loads(a.registry.read_text());weight=(a.registry.parent/registry['paths']['clip']).resolve()
    entries=[x for x in registry['files'] if (a.registry.parent/x['path']).resolve()==weight]
    if len(entries)!=1:raise ValueError('registered CLIP weights required')
    h=hashlib.sha256()
    with weight.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=entries[0]['sha256'] or weight.stat().st_size!=entries[0]['bytes']:raise ValueError('weight mismatch')
    model=load_openai_model(str(weight),precision='fp32',device='cuda:0').eval().requires_grad_(False)
    pre=transforms.Compose([transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(OPENAI_DATASET_MEAN,OPENAI_DATASET_STD)])
    sources={}
    for uid in sorted({r['source_uid'] for r in rows}):
        files=sorted((a.source_root/uid/'source/mv').glob('render_*.png'))
        if len(files)!=26:raise ValueError(f'{uid}: expected 26 source views')
        sources[uid]=files
    files=list(dict.fromkeys([p for group in sources.values() for p in group]+[Path(r['target_anchor']) for r in rows]))
    embeddings={}
    def convert(path):
        with Image.open(path) as im:
            rgba=im.convert('RGBA').resize((512,512),Image.Resampling.LANCZOS)
            x=np.asarray(rgba,dtype=np.float32)/255
        rgb=x[...,:3]*x[...,3:]+1-x[...,3:]
        return pre(Image.fromarray(np.rint(rgb*255).astype(np.uint8)))
    with torch.inference_mode():
        for i in range(0,len(files),a.batch_size):
            batch=files[i:i+a.batch_size];x=torch.stack([convert(p) for p in batch]).cuda()
            features=model.encode_image(x,normalize=True).float().cpu().numpy()
            for path,feature in zip(batch,features):embeddings[path]=feature
    results=[]
    for row in rows:
        uid=row['source_uid'];paths=sources[uid];target=embeddings[Path(row['target_anchor'])]
        scores=np.array([target@embeddings[path] for path in paths]);order=np.argsort(-scores,kind='stable')
        matching=[i for i,path in enumerate(paths) if path.stem==row['original_view_name']]
        if len(matching)!=1:raise ValueError('same-index source missing')
        rank=int(np.flatnonzero(order==matching[0])[0])+1
        results.append({'sample_id':row['sample_id'],'source_uid':uid,'same_index_view':row['original_view_name'],'same_index_rank':rank,'same_index_cosine':float(scores[matching[0]]),'best_view':paths[int(order[0])].stem,'best_cosine':float(scores[order[0]]),'best_minus_same_index':float(scores[order[0]]-scores[matching[0]])})
    ranks=np.array([r['same_index_rank'] for r in results]);margin=np.array([r['best_minus_same_index'] for r in results])
    output={'status':'PROVISIONAL_CAMERA_VIEW_RETRIEVAL_DIAGNOSTIC','manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'clip_weight_sha256':entries[0]['sha256'],'reference':'edited target RGB queried against all 26 original source-view RGBs of same object','warning':'CLIP similarity is sensitive to edits and appearance, not a camera ground truth; ranks neither prove nor disprove target/source camera identity. No retrieval result is used to select or modify a camera.','samples':len(results),'objects':len(sources),'top1':int((ranks==1).sum()),'top3':int((ranks<=3).sum()),'median_rank':float(np.median(ranks)),'median_score_gap':float(np.median(margin)),'results':results}
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(output,indent=2)+'\n');tmp.replace(a.output)
    print(json.dumps({k:v for k,v in output.items() if k!='results'}),flush=True)

if __name__=='__main__':main()
