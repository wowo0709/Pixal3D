"""Generate official MV-Adapter target-only views for the BenchUp pipeline."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('repo', 'manifest', 'base', 'vae', 'adapter', 'output'): p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--gpu', required=True, choices=['0', '1', '2', '3'])
    p.add_argument('--method', choices=['mvadapter', 'corradapter'], default='mvadapter')
    p.add_argument('--rank', type=int, default=0); p.add_argument('--world-size', type=int, default=1); p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu or not 0 <= a.rank < a.world_size: raise ValueError('explicit allowed GPU and valid shard required')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, str(a.repo))
    import torch
    from scripts.inference_i2mv_sdxl import prepare_pipeline, run_pipeline
    torch.set_num_threads(2); torch.cuda.set_device(0)
    pipe = prepare_pipeline(str(a.base), str(a.vae), None, None, str(a.adapter), None, 6, 'cuda', torch.float16)
    rows = [json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    if len(rows) != 109 or len({r['sample_id'] for r in rows}) != 109: raise ValueError('frozen population required')
    rows = rows[a.rank::a.world_size]
    if a.limit: rows = rows[:a.limit]
    expected_status = 'MVADAPTER_TARGET_ONLY_SIX_GENERATED_VIEWS' if a.method == 'mvadapter' else 'CORRADAPTER_TARGET_ONLY_SIX_GENERATED_VIEWS'
    code_commit = '4277e0018232bac82bb2c103caf0893cedb711be' if a.method == 'mvadapter' else 'a269d224f89aa7ae2bed1b4c61b2613352e8a1d2'
    for row in rows:
        sid = row['sample_id']; target = Path(row['target_anchor']); out = a.output / 'cases' / sid
        if out.exists():
            receipt = out / 'receipt.json'
            if not receipt.is_file(): raise FileExistsError(f'partial case: {out}')
            old = json.loads(receipt.read_text())
            if old.get('status') != expected_status or old['target_sha256'] != sha(target) or any(sha(out / v['name']) != v['sha256'] for v in old['views']): raise ValueError(f'output mismatch: {out}')
            print(json.dumps({'status':'reused', 'sample_id':sid}), flush=True); continue
        start = time.monotonic()
        with torch.inference_mode():
            images, ref = run_pipeline(pipe, num_views=6, text='high quality', image=str(target), height=768, width=768, num_inference_steps=50, guidance_scale=3., seed=0, device='cuda', azimuth_deg=[0,45,90,180,270,315])
        if len(images) != 6: raise ValueError('six views required')
        out.mkdir(parents=True); views=[]
        for i, image in enumerate(images):
            name=f'view_{i:02d}.png'; image.save(out/name); views.append({'name':name,'sha256':sha(out/name),'azimuth_deg':[0,45,90,180,270,315][i]})
        ref.save(out/'reference.png')
        receipt={'status':expected_status,'sample_id':sid,'target_sha256':sha(target),'views':views,'reference_sha256':sha(out/'reference.png'),'source_images_used':False,'target_mesh_used':False,'seed':0,'steps':50,'guidance_scale':3.,'code_commit':code_commit,'seconds':round(time.monotonic()-start,2)}
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)

if __name__ == '__main__': main()
