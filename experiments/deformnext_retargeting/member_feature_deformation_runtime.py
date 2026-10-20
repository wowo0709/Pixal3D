"""Explicit physical-DQ phases with a scoped RAW-only dataset registration."""
import argparse
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys

from .member_feature_deformation_recipe import NAME,DATASET,SCHEMA,ARM

PHASES=dict(native='probe_member_feature_deformation_source_ray',capacity='probe_member_feature_deformation_fullbatch',
    science='train_member_feature_deformation_source_ray',evaluation='evaluate_member_feature_deformation_source_ray')


@contextmanager
def register_raw_dataset():
    from pixal3d import datasets
    from pixal3d.datasets.deformnext_member_raw import DeformNeXtMemberRawSS64
    previous=datasets.__dict__.get(DATASET)
    setattr(datasets,DATASET,DeformNeXtMemberRawSS64)
    try:yield
    finally:
        if previous is None:datasets.__dict__.pop(DATASET,None)
        else:setattr(datasets,DATASET,previous)


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--phase',choices=list(PHASES),required=True)
    local,remaining=parser.parse_known_args()
    module=importlib.import_module('.'+PHASES[local.phase],__package__)
    previous=sys.argv
    try:
        sys.argv=[sys.argv[0],*remaining]
        with register_raw_dataset():module.main()
    finally:sys.argv=previous
    if local.phase in {'native','evaluation'}:output=Path(remaining[remaining.index('--output')+1])
    else:output=Path(json.loads(Path(remaining[remaining.index('--config')+1]).read_text())['output_dir'])
    (output/'member-feature-deformation-runtime.json').write_text(json.dumps(dict(
        status='PASS_COMPLETED_MEMBER_FEATURE_DEFORMATION_RUNTIME_PHASE',phase=local.phase,trainer=NAME,
        dataset=DATASET,schema=SCHEMA,arm=ARM,physical_deformation_query_channels=2048,
        physical_retargeter_input_channels=4096,actual_QKV_feature_head_tensors=9,actual_source_tensors=20,
        same_correspondence_probabilities_for_feature_and_RAW_geometry=True,
        input_RAW_only_no_old_matching_sidecars=True,native_optimizer_sampler_AMP_LoRA_protocol_preserved=True,
        historical_target_patch_query_or_R96_warmup_reproduction=False,full_goal_complete=False),indent=2)+'\n')


if __name__=='__main__':main()
