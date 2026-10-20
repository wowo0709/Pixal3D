"""Offline, GPU-only official DreamSim and original OpenAI CLIP image scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .perceptual_image_inputs import paired_images


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8388608), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_models(root):
    registry = json.loads((root/'registry.json').read_text())
    if registry['status'] != 'PASS_OFFICIAL_PERCEPTUAL_WEIGHT_SOURCE_CACHE_AND_CPU_LOAD':
        raise ValueError('prepared official metric source/weight registry required')
    for row in registry['files']:
        path = root/row['path']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('prepared metric source or weight changed: '+str(path))
    sys.path.insert(0, str(root/registry['isolated_dependencies']))
    import dreamsim
    from open_clip.openai import load_openai_model
    from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
    from torchvision import transforms
    hub_load = torch.hub.load
    hub_download = torch.hub.download_url_to_file

    def pinned(repo, model, *args, **kwargs):
        if repo != 'facebookresearch/dino:main' or model != 'dino_vitb16':
            raise ValueError('unexpected metric source lookup: '+str((repo, model)))
        kwargs.pop('source', None)
        return hub_load(str(root/registry['DINO_source']), model, *args, source='local', **kwargs)

    def offline(*args, **kwargs):
        raise ValueError('network weight fallback is forbidden; use prepared registry')

    torch.hub.load = pinned
    torch.hub.download_url_to_file = offline
    try:
        metric, preprocess = dreamsim.dreamsim(pretrained=True, device='cuda',
            cache_dir=str(root/registry['DreamSim_cache']), normalize_embeds=True, dreamsim_type='ensemble')
        clip = load_openai_model(str(root/registry['CLIP_checkpoint']), precision='fp32', device='cuda')
    finally:
        torch.hub.load = hub_load
        torch.hub.download_url_to_file = hub_download
    metric.eval().requires_grad_(False)
    clip.eval().requires_grad_(False)
    adapters = [value for name,value in metric.named_parameters() if 'lora_' in name]
    trained = [value for name,value in metric.named_parameters() if 'lora_B' in name]
    if len(adapters) != 72 or not trained or not all(torch.count_nonzero(v) for v in trained):
        raise ValueError('complete trained official DreamSim ensemble LoRA required')
    if (metric.base_model.model.embed_size != 1792
            or list(metric.base_model.model.model_list) != ['dino_vitb16', 'clip_vitb16', 'open_clip_vitb16']
            or any('lora_' in name for name,_ in clip.named_parameters())
            or clip.visual.transformer.resblocks[0].mlp.gelu.__class__.__name__ != 'QuickGELU'
            or any(p.requires_grad for model in (metric,clip) for p in model.parameters())):
        raise ValueError('frozen official ensemble and separate original OpenAI CLIP required')
    clip_preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)])
    return metric, preprocess, clip, clip_preprocess, registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if gpu not in {'0','1','2','3'} or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one explicitly assigned physical GPU0-3 required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.3,0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    dream, dream_pre, clip, clip_pre, registry = load_models(Path(plan['metric_models']))
    rows, sanity = [], None
    with torch.inference_mode():
        for item in plan['inputs']:
            path = Path(item['shared_npz'])
            with np.load(path) as z:
                images, audit = paired_images(z['prediction'].copy(), z['GT_shared_renderer'].copy(), z['GT_alpha'].copy())
            d = {name:dream_pre(img).cuda() for name,img in images.items()}
            c = {name:clip_pre(img).unsqueeze(0).cuda() for name,img in images.items()}
            de = {name:dream.embed(img) for name,img in d.items()}
            ce = {name:clip.encode_image(img, normalize=True) for name,img in c.items()}
            score = {}
            for suffix,tag in [('full','fullframe'),('GTcrop','GT_foreground_crop')]:
                p,g = 'prediction_'+suffix,'reference_'+suffix
                score['DreamSim_'+tag] = float(1-F.cosine_similarity(de[p],de[g],dim=-1).item())
                score['CLIP_cosine_'+tag] = float(F.cosine_similarity(ce[p],ce[g],dim=-1).item())
            if sanity is None:
                identity_d = float(dream(d['reference_full'], d['reference_full']).item())
                identity_c = float(F.cosine_similarity(ce['reference_full'], ce['reference_full'],dim=-1).item())
                direct_d = float(dream(d['prediction_full'],d['reference_full']).item())
                sanity = dict(status='PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY',
                    DreamSim_identity=identity_d, CLIP_identity=identity_c,
                    DreamSim_official_forward=direct_d, DreamSim_embedding_cosine=score['DreamSim_fullframe'],
                    independent_original_CLIP=True, trained_DreamSim_LoRA_tensors=72,
                    CPU_model_forward_or_training_updates=0, actual_changed_render_pair=True)
                if (abs(identity_d)>1e-6 or abs(identity_c-1)>1e-6
                        or abs(direct_d-score['DreamSim_fullframe'])>1e-6
                        or direct_d<=1e-5 or score['CLIP_cosine_fullframe']>=1-1e-5):
                    raise ValueError('official metric identity/forward/changed-pair validation failed: '+json.dumps(sanity))
                (output/'GPU-sanity.json').write_text(json.dumps(sanity,indent=2)+'\n')
            if (not all(np.isfinite(v) for v in score.values())
                    or any(not -1e-6<=v<=2+1e-6 for k,v in score.items() if k.startswith('DreamSim'))
                    or any(not -1-1e-6<=v<=1+1e-6 for k,v in score.items() if k.startswith('CLIP'))):
                raise ValueError('finite official metric ranges required')
            row=dict(sample_id=item['sample_id'],source_cluster_id=item['source_cluster_id'],
                arm=item['arm'],camera_id=item['camera_id'],scores=score,image_audit=audit,
                shared_npz_SHA256=sha(path))
            rows.append(row)
            with (output/'per-view.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(dict(completed=len(rows),expected=len(plan['inputs']),arm=item['arm'],scores=score)),flush=True)
    report=dict(status='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION',per_view=rows,
        GPU_sanity=sanity,metric_registry_SHA256=sha(Path(plan['metric_models'])/'registry.json'),
        model_weights=registry['sources'],DreamSim_lower_is_better=True,CLIP_image_cosine_higher_is_better=True,
        separate_GT_only_crop_supplementary=True,fullframe_primary=True,model_precision='float32,TF32off,noAMP',
        seconds=time.monotonic()-started,peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        no_model_training=True,no_dataset_download=True,global_environment_changed=False,
        full_panel=bool(plan.get('full_panel',False)),scope=plan['scope'],model_adopted=False,full_goal_complete=False)
    (output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
