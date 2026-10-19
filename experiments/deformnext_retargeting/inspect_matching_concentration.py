"""Inspect predicted match attention, without GT correspondence or model updates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from pixal3d.modules.r9_matching import categorical_weights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    cfg=json.loads(args.config.read_text());ds=cfg['dataset']['args'];spec=cfg['trainer']['args']['deform_conditioner']
    if spec['implementation']!='r9-native-v5' or spec['topk']!=64:
        raise ValueError('declared R9 semantic64 score bank required')
    manifest=Path(cfg['validation']['roots_file'])
    rows={r['sample_id']:r for r in [json.loads(line) for line in manifest.read_text().splitlines()]}
    ids=json.loads(args.panel.read_text())['sample_ids'];assert len(ids)==len(set(ids))==32
    def resolve(raw):
        text=str(raw)
        for original,replacement in sorted(ds['path_map'].items(),key=lambda row:-len(row[0])):
            if text==original or text.startswith(original+'/'):
                return Path(replacement+text[len(original):])
        return Path(text)
    support=[];consumption=[];banks=[]
    evidence=[];temperatures=sorted(set([float(spec['score_temperature']),.005,.01,.02,.05,.1]))
    weights_by_temp={t:[] for t in temperatures}
    for sid in ids:
        row=rows[sid];source=resolve(row['transport_cache']);match=Path(ds['match_root'])/(sid+'.npz')
        with np.load(source) as z:valid_source=torch.from_numpy(z['source_valid'].copy()).bool().flatten()
        with np.load(match) as z:
            assert str(z['conditioning_input_sha256'])==row['conditioning_input_sha256']
            indices=torch.from_numpy(z['semantic_indices'].copy()).long()
            logits=torch.from_numpy(z['matching_logits'].copy()).float()
            valid=torch.from_numpy(z['candidate_valid'].copy()).bool() & valid_source[indices]
        with Image.open(resolve(row['target_anchor'])) as image:
            alpha=np.asarray(image.convert('RGBA').resize((512,512),Image.Resampling.NEAREST))[...,3]
            target=torch.from_numpy((alpha[8::16,8::16]>127).copy()).flatten()
        valid &= target[:,None] & (valid_source.sum()>=64)
        active=valid.any(-1)
        with np.load(Path(ds['projection_root'])/(sid+'.npz')) as z:
            patch=torch.from_numpy(z['patch_ids'].copy()).long()
        if ((patch<0)|(patch>=1024)).any():raise ValueError('invalid native PBR consumer patch address')
        counts=torch.bincount(patch,minlength=1024).float()[active]
        consumption.append(counts);support.append(int(active.sum()));banks.append(logits[active])
        one=dict(sample_id=sid,active_RGB_patch_rays=int(active.sum()),consumed_valid_PBR_sparse_sites=int(counts.sum()),temperatures={})
        for temperature in temperatures:
            probability=categorical_weights(logits,valid,temperature)[active]
            entropy=-(probability*probability.clamp_min(1e-30).log()).sum(-1)
            stats=torch.stack([entropy.exp(),probability.max(-1).values],-1)
            weights_by_temp[temperature].append(stats)
            one['temperatures'][str(temperature)]=dict(effective_K=float(stats[:,0].mean()),Top1_mass=float(stats[:,1].mean()))
        evidence.append(one)
    counts=torch.cat(consumption)
    aggregate={}
    for temperature,parts in weights_by_temp.items():
        stats=torch.cat(parts)
        aggregate[str(temperature)]=dict(RGB_patch_rays=len(stats),
            ray_mean_effective_K=float(stats[:,0].mean()),ray_median_effective_K=float(stats[:,0].median()),
            ray_mean_Top1_mass=float(stats[:,1].mean()),fraction_rays_Top1_mass_over90pct=float((stats[:,1]>.9).float().mean()),
            fraction_rays_Top1_mass_over99pct=float((stats[:,1]>.99).float().mean()),
            PBR_sparse_site_weighted_effective_K=float((stats[:,0]*counts).sum()/counts.sum()),
            PBR_sparse_site_weighted_Top1_mass=float((stats[:,1]*counts).sum()/counts.sum()))
    args.output.mkdir(parents=True,exist_ok=False)
    record=dict(status='PASS_PREDICTED_MATCHING_CONCENTRATION32',model_config=str(args.config),
        declared_temperature=spec['score_temperature'],source_clusters=len({rows[sid]['source_cluster_id'] for sid in ids}),
        actual_candidate_bank_and_RGB_alpha=True,GT_correspondence_not_loaded=True,actual_PBR_projection_consumer=True,
        per_temperature=aggregate,per_asset=evidence,optimizer_updates=0,
        scope='Attention concentration only;temperature changes are not evaluated quality gains.',
        source_SHA256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),args.config,args.panel,manifest]})
    (args.output/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
