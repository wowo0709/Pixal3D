"""Original native PBR training with exact latent/bounded-render initial QA."""
import argparse
import json
import os
from pathlib import Path
import torch.multiprocessing as mp

from . import train_r_mediated_pbr_native as native
from .pbr_initial_render_identity import POLICY,validate_initial_latent_and_render


def worker(rank,cfg,plan):
    from . import r_mediated_pbr_science_metrics as metrics
    previous=metrics.validate_initial4
    previous_factory=native.science_trainer
    reports=[]
    def validate(actual,reference):
        proof=validate_initial_latent_and_render(actual,reference);reports.append(proof)
    metrics.validate_initial4=validate
    def factory(base,config,document):
        trainer,evaluate=previous_factory(base,config,document)
        def checked_eval(instance,*args,**kwargs):
            row=evaluate(instance,*args,**kwargs)
            if instance.step==0:
                if len(reports)!=1:raise ValueError('one actual strict latent/bounded-render proof required')
                folder=Path(instance.output_dir)/'validation-panels/step0000000'
                proof=reports[0]
                (folder/'initial-identity.json').write_text(json.dumps(dict(
                    status='PASS_NATIVE_PBR_EXACT_LATENT_BOUNDED_RENDER_INITIAL4',
                    actual_inputs_noise_PBR_latent_equal_original32=True,
                    actual_rendering_metrics_equal_original32=proof['original_render_metrics_exact'],
                    render_metrics_within_declared_absolute_bounds=True,initial_identity_policy=POLICY,
                    optimizer_updates=0,proof=proof,model_adopted=False),indent=2)+'\n')
                row['initial_identity_policy']=POLICY
                (Path(instance.output_dir)/'initial-latent-render-identity.json').write_text(json.dumps(proof,indent=2)+'\n')
            return row
        return trainer,checked_eval
    native.science_trainer=factory
    try:native.worker(rank,cfg,plan)
    finally:
        native.science_trainer=previous_factory;metrics.validate_initial4=previous


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());cfg,_=native.guard_plan(a.plan,plan)
    if plan.get('initial_identity_policy')!=POLICY:
        raise ValueError('explicit initial identity policy required')
    from .cascade_view_roles import ClosedInputs
    check=ClosedInputs(plan['canonical_local_root'])
    calibration=Path(plan['initial_renderer_calibration']);confirmation=Path(plan['initial_renderer_vanilla_confirmation'])
    for path in[calibration,confirmation]:check.check(path,path.parent/'file4-closure.json')
    cold=json.loads(calibration.read_text());vanilla=json.loads(confirmation.read_text())
    if cold.get('actual_optimizer_updates')!=0 or any(r['identity_mismatches']for r in cold['differences']):
        raise ValueError('actual cold feature latent identity calibration required')
    if vanilla.get('status')!='PASS_UNCHANGED_VANILLA_INITIAL4_LATENT_AND_RENDER_BOUND_CONFIRMATION':
        raise ValueError('unchanged vanilla renderer confirmation required')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if visible!=['0','1','2','3']or not set(visible)<=set(os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(',')):
        raise ValueError('original assigned DDP4 group required')
    output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=False);cfg['config']=str(plan['config'])
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    mp.spawn(worker,args=(cfg,plan),nprocs=4,join=True)
    native.guard_plan(a.plan,plan)
    for rank in range(4):
        record=json.loads((output/f'stage-completed.rank{rank}.json').read_text())
        if record['status']!='PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE':raise ValueError('all rank native1900 completion required')
    print(json.dumps(dict(status='PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE',mode=plan['mode'],initial_identity_policy=POLICY,full_goal_complete=False)),flush=True)


if __name__=='__main__':main()
