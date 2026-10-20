"""Run a separately resolved member-posterior SS proof/train/evaluation phase.

Only this process selects the new constructor. Existing frozen recipes and
numerical training/sampling/proof entrypoints remain unchanged on disk.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

from .train_target_first_observed import TRAINERS,validate_recipe

NAME='MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer'
SCHEMA='member-posterior-source-ray-v1'
PROPOSAL='Deduplicated observed target-patch D1 posteriors mixed per source-camera ray; final Top64'


def validate_member_recipe(cfg):
    experiment=cfg['experiment']
    if (cfg['trainer']['name']!=NAME
            or cfg['trainer']['args'].get('ffn_chunk_assets') not in (12,24)
            or experiment.get('conditioning_schema')!=SCHEMA
            or experiment.get('matching_proposals')!=PROPOSAL
            or experiment.get('no_query_fallback')!='Native pretrained MV4 prior, zero source correction'
            or experiment.get('initial_function')!='Exact native MV4 global/projection/negative; zero source correction'
            or experiment.get('readout_arm') not in {'top1','weighted','attention'}):
        raise ValueError('explicit member-posterior semantic readout/nativeMV4/boundedFFN recipe required')
    adapted=copy.deepcopy(cfg);arm=experiment['readout_arm']
    adapted['trainer']['name']=TRAINERS[arm]
    return validate_recipe(adapted)


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--phase',choices=['native','capacity','science','evaluation'],required=True)
    local,remaining=parser.parse_known_args()
    from pixal3d.trainers.flow_matching import native_prior_source_ray as original
    from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer as Trainer
    from . import train_native_prior_source_ray as training
    from . import probe_native_prior_source_ray as proof
    from . import probe_native_prior_fullbatch as capacity
    from . import evaluate_native_prior_source_ray as evaluation
    saved=(original.NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer,
        training.NAME,training.validate_native_prior_recipe,sys.argv)
    original.NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer=Trainer
    training.NAME=NAME;training.validate_native_prior_recipe=validate_member_recipe
    try:
        sys.argv=[sys.argv[0],*remaining]
        {'native':proof.main,'capacity':capacity.main,'science':training.main,'evaluation':evaluation.main}[local.phase]()
    finally:
        (original.NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer,
            training.NAME,training.validate_native_prior_recipe,sys.argv)=saved
    if local.phase in {'native','evaluation'}:
        out=Path(remaining[remaining.index('--output')+1])
    else:
        cfg=json.loads(Path(remaining[remaining.index('--config')+1]).read_text());out=Path(cfg['output_dir'])
    if local.phase=='evaluation':
        path=out/'evaluation.json';report=json.loads(path.read_text())
        report['native_prior_source_ray'].update(member_posterior_proposals=PROPOSAL,
            frozen_matching_decoder_and_score=True,distinct_patch_votes_per_ray=True,
            candidate_budget=64,matching_support_is_original_mean_RAW64=False)
        path.write_text(json.dumps(report,indent=2)+'\n')
    (out/'member-posterior-runtime.json').write_text(json.dumps(dict(
        status='PASS_COMPLETED_MEMBER_POSTERIOR_RUNTIME_PHASE',phase=local.phase,
        trainer=NAME,matching_proposals=PROPOSAL,physical_deformation_query_channels=0,
        native_MV4_carrier_and_native_optimizer_sampler_AMP_protocol_preserved=True,
        no_GT_in_query_candidate_prior_or_context=True,initial_and_capacity_or_quality_scope_owned_by_phase=True,
        full_goal_complete=False),indent=2)+'\n')


if __name__=='__main__':
    main()
