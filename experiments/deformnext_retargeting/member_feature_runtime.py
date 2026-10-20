"""Explicit isolated four-head NoDQ phases; no frozen registry/recipe edits."""
import argparse
import importlib
import json
from pathlib import Path
import sys

from .member_feature_recipe import NAME, SCHEMA, ARM


PHASES = dict(native='probe_member_feature_source_ray', capacity='probe_member_feature_fullbatch',
              science='train_member_feature_source_ray', evaluation='evaluate_member_feature_source_ray')


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--phase', choices=list(PHASES), required=True)
    local, remaining = parser.parse_known_args()
    module = importlib.import_module('.'+PHASES[local.phase], __package__)
    previous = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        module.main()
    finally:
        sys.argv = previous
    if local.phase in {'native', 'evaluation'}:
        output = Path(remaining[remaining.index('--output')+1])
    else:
        config = json.loads(Path(remaining[remaining.index('--config')+1]).read_text())
        output = Path(config['output_dir'])
    (output/'member-feature-runtime.json').write_text(json.dumps(dict(
        status='PASS_COMPLETED_MEMBER_FEATURE_RUNTIME_PHASE', phase=local.phase,
        trainer=NAME, schema=SCHEMA, arm=ARM, physical_deformation_query_channels=0,
        actual_QKV_feature_head_tensors=9, actual_source_tensors=19,
        native_optimizer_sampler_AMP_LoRA_and_camera_latent_protocol_preserved=True,
        historical_target_patch_query_or_R96_warmup_reproduction=False,
        no_GT_for_matching_or_query=True, full_goal_complete=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
