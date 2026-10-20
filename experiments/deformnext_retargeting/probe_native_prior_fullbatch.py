"""Disposable native fullbatch96 proof with source-prior and FFN chunking."""
import argparse
import json
from pathlib import Path


def main():
    from pixal3d.trainers.flow_matching import target_first_observed_query as original
    from pixal3d.trainers.flow_matching.native_prior_source_ray import (
        NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer as Trainer,
    )
    from . import probe_target_first_fullbatch as proof
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    chunk=cfg['trainer']['args'].get('ffn_chunk_assets')
    if cfg['trainer']['name']!=Trainer.__name__ or chunk not in (12,24):
        raise ValueError('explicit native-prior/fullmicro96/bounded-FFN recipe required')
    captured=[];batches=[];handles=[]

    class Captured(Trainer):
        def init_models_and_more(self,**kwargs):
            super().init_models_and_more(**kwargs)
            captured.append(self)
            for block in self.models['denoiser'].blocks:
                handles.append(block.mlp.mlp[0].register_forward_pre_hook(
                    lambda _,inputs:batches.append(int(inputs[0].shape[0]))))

    Captured.__name__=Trainer.__name__
    saved=original.ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer
    original.ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer=Captured
    try:
        proof.main()
    finally:
        original.ObservedQueryTargetFirstRayImageConditionedProjFlowMatchingCFGTrainer=saved
        for handle in handles:
            handle.remove()
    if len(captured)!=1 or not batches or max(batches)>chunk:
        raise ValueError('actual native forward/backward must not expand full96 inside FFN')
    path=Path(cfg['output_dir'])/'capacity.json'
    report=json.loads(path.read_text())
    report.update(native_MV4_prior_preserved=True,source_update='u*g*(E-original_source)+u*R',
        ffn_chunks=captured[0]._native_ffn_chunk_report,
        actual_maximum_FFN_expansion_batch=max(batches),actual_FFN_expansion_calls=len(batches),
        native_global96_micro96_accum1_unchanged=True)
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
