"""Run the declared frozen-matcher utility profile with native SS launch gates."""
import argparse
import json
from pathlib import Path

from .matching_v2_ss import train


def validate_profile(cfg, root):
    if cfg.get('ckpt') != 'none' or cfg.get('load_dir') != '':
        raise ValueError('fresh utility recipe requires ckpt="none" and load_dir="" strings')
    spec=cfg['trainer']['args']['deform_conditioner']
    required={'decoder':True,'score_prior':True,'train_decoder':False,'train_score':False,
              'content_mode':'residual-zero','chunk_size':64}
    if (spec.get('implementation')!='r9-native-v5' or spec.get('arm')!='c1'
        or spec.get('task_attention')!=required or spec.get('score_temperature')!=.005
        or spec.get('correspondence_mode','semantic')!='semantic'
        or spec.get('deformation_dim') is not None):
        raise ValueError('frozen D1/scalar prior,semantic64,C1,zero utility,T0.005 profile required')
    name=cfg['experiment']['arm']
    if Path(cfg['default_output_dir']).resolve()!=(root/'runs_ss64'/name).resolve():
        raise ValueError('followup output must belong to its declared root and arm')
    if (cfg['num_gpus']!=2 or cfg['trainer']['args']['max_steps']!=480
        or cfg['trainer']['args']['batch_size_per_gpu']!=48 or cfg['trainer']['args']['batch_split']!=1
        or cfg['experiment'].get('required_microbatch_per_gpu')!=48):
        raise ValueError('followup requires fresh480updates,global96,DDP2,micro48,accumulation1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['train'],required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    args=p.parse_args()
    validate_profile(json.loads(args.config.read_text()),args.root)
    # The existing engine runs its actual native/backward proof and two-update
    # maximum-batch probe before starting fresh original-SS4210 training.
    train(args)


if __name__=='__main__':main()
