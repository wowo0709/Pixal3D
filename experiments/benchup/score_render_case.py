"""Score image-space render diagnostics against explicit BenchUp input images."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest','chunk','benchmark-root','output'):p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    plan=json.loads((a.chunk/'plan.json').read_text());sid=plan['sample_id']
    rows=[x for x in map(json.loads,a.manifest.read_text().splitlines()) if x['sample_id']==sid]
    if len(rows)!=1:raise ValueError('unique case required')
    row=rows[0];sys.path.insert(0,str(a.benchmark_root));from deformnext_benchmark.metrics import image_metrics
    def rgba(path):
        with Image.open(path) as im:
            value=np.asarray(im.convert('RGBA').resize((512,512),Image.Resampling.LANCZOS),dtype=np.float64)/255
        alpha=value[...,3];return value[...,:3]*alpha[...,None]+(1-alpha[...,None]),alpha
    reference=[row['target_anchor'],row['source_images'][0],row['source_images'][1]];output={}
    for mode in ('raw','source_m0'):
        per=[]
        for i,path in enumerate(reference):
            pred,pa=rgba(a.chunk/'stage'/mode/f'view{i}.png');gt,ga=rgba(path)
            per.append({'reference':'edited_target_RGB' if i==0 else f'original_source_RGB_{i}','metrics':image_metrics(pred,gt,pa,ga)})
        output[mode]=per
    result={'status':'IMAGE_DIAGNOSTIC_PROVISIONAL_NOT_FINAL_BENCHMARK','sample_id':sid,'reference_resize':'RGBA Lanczos 1024 to 512','target_camera_assumption':row['target_camera_assumption'],'source_camera_qa_iou':row['source_camera_qa_iou'],'metrics':output,'target_3D_metrics':'unavailable_no_GT_target_mesh','note':'input source views can penalize intended edits; render lighting differs from dataset'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
