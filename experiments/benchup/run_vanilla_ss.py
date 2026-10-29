"""GT-free BenchUp SS generation using pinned DeformNeXt SS weights."""
from __future__ import annotations
import argparse, copy, hashlib, json, os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('ATTN_BACKEND', 'sdpa')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
import torch

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest','scale-predictions','evaluation','output'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--gpu',required=True,choices=['0','1','2','3'])
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--arm',choices=['V1','SELECTED'],default='V1')
    a=p.parse_args()
    if a.world_size<1 or not 0<=a.rank<a.world_size: raise ValueError('invalid shard')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu: raise ValueError('explicit allowed GPU required')
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    scales={r['sample_id']:r['predicted_mesh_scale'] for r in map(json.loads,a.scale_predictions.read_text().splitlines())}
    receipt=json.loads(a.scale_predictions.with_suffix('.receipt.json').read_text())
    if receipt['output_sha256']!=sha(a.scale_predictions) or receipt['manifest_sha256']!=sha(a.manifest): raise ValueError('scale receipt mismatch')
    if len(rows)!=109 or len(scales)!=109 or set(scales)!={r['sample_id'] for r in rows}: raise ValueError('population mismatch')
    for r in rows: r['mesh_scale']=scales[r['sample_id']]
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    manifest=a.output/'inference-manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    prior=json.loads(a.evaluation.read_text()); cfg=json.loads(Path(prior['arm_config']).read_text())
    if (prior['conditioner_checkpoint'] is None) != (a.arm=='V1'): raise ValueError('arm/checkpoint mismatch')
    from pixal3d import datasets, models, trainers
    from pixal3d.utils.data_utils import recursive_to_device
    from train import setup_rng
    from experiments.deformnext_retargeting.projection_maps import anchor_patch_indices
    class InferenceDataset(datasets.DeformNeXtSS64):
        def __getitem__(self,index):
            item=self._common(self.rows[index])
            if a.arm=='SELECTED':
                grid=np.indices((16,16,16)).reshape(3,-1).T
                item['transport_patch_grid']=torch.from_numpy(anchor_patch_indices(self.rows[index],grid,16).reshape(16,16,16)).long()
            return item
    ds_args=copy.deepcopy(cfg['validation']['dataset_args'])
    ds_args.update(expected_split='test',expected_ss_schema=None,path_map={},verify_files=False)
    if a.arm=='V1': ds_args['transport_arm']='v1'
    else: ds_args.update(projection_root=None,match_root=None)
    dataset=InferenceDataset(str(manifest),**ds_args)
    setup_rng(0,cfg['training_seed']);torch.set_num_threads(2);torch.cuda.set_device(0)
    with torch.device('meta'):
        model_defs={name:getattr(models,spec['name'])(**spec['args']) for name,spec in cfg['models'].items()}
    kw=copy.deepcopy(cfg['trainer']['args'])
    kw.update(continuation_ckpt={'denoiser':prior['checkpoint']},batch_size_per_gpu=1,batch_split=1,num_workers=0,prefetch_data=False,max_steps=1)
    if a.arm=='V1':
        kw['deform_conditioner']=None;kw.get('model_learning_rates',{}).pop('deform_conditioner',None)
        from experiments.deformnext_retargeting.r_mediated_native_runtime import selected_trainer
        if cfg['trainer']['name'] != 'RMediatedNativeFlowTrainer': raise ValueError('unexpected V1 trainer')
        trainer_type=selected_trainer('vanilla')
    else:
        from experiments.deformnext_retargeting.consistent_evidence_native_runtime import selected_trainer
        if prior.get('variant')!='E_FIXED_RAW':raise ValueError('frozen selected variant required')
        trainer_type=selected_trainer(prior['variant'])
    trainer=trainer_type(model_defs,dataset,**kw,output_dir=str(a.output/'loader'),load_dir=None,step=None,wandb_run=None)
    if a.arm=='SELECTED':
        trainer.deform_conditioner.load_state_dict(torch.load(prior['conditioner_checkpoint'],map_location='cpu',weights_only=True),strict=True)
        trainer.deform_conditioner.eval()
    trainer.models['denoiser'].eval().requires_grad_(False);trainer.p_uncond=0
    steps=int(cfg['validation']['quick_metrics']['sampling_steps'])
    selected=[(i,row) for i,row in enumerate(rows) if i%a.world_size==a.rank]
    if a.limit: selected=selected[:a.limit]
    records=[]
    for i,row in selected:
        pack=recursive_to_device(dataset.collate_fn([dataset[i]]),trainer.device)
        pack.pop('_dataset_name',None);pack.pop('_sha256',None)
        seed=42+i;torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        noise=torch.randn((1,8,16,16,16),device=trainer.device)
        with torch.inference_mode(),torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
            latent=trainer.get_sampler().sample(trainer.models['denoiser'],noise=noise,**trainer.get_inference_cond(**pack),steps=steps,guidance_strength=3.,verbose=False).samples
            occ=dataset.decode_latent(latent,batch_size=1)[0,0]>0
        occupancy=occ.cpu().numpy();dest=a.output/'ss-artifacts';dest.mkdir(exist_ok=True)
        out=dest/(hashlib.sha256(row['sample_id'].encode()).hexdigest()+'.npz')
        np.savez_compressed(out,prediction_latent=latent.float().cpu().numpy()[0],prediction_occupancy_packed=np.packbits(occupancy.reshape(-1)),occupancy_shape=np.array([64,64,64],np.int32),sample_id=np.asarray(row['sample_id']),seed=np.int64(seed))
        record={'status':'SS_GENERATED_NO_TARGET_3D_GT','sample_id':row['sample_id'],'artifact':str(out),'sha256':sha(out),'occupied':int(occupancy.sum()),'seed':seed}
        records.append(record)
        with (a.output/'progress.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    (a.output/'complete.json').write_text(json.dumps({'status':'PASS_GT_FREE_SS_SHARD','samples':len(records),'rank':a.rank,'world_size':a.world_size,'manifest_sha256':sha(a.manifest),'scale_sha256':sha(a.scale_predictions),'evaluation_sha256':sha(a.evaluation)},indent=2)+'\n')

if __name__=='__main__': main()
