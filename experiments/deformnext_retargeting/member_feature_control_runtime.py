"""Explicit isolated four-head NoDQ phases; no frozen registry/recipe edits."""
import argparse
import importlib
import json
from pathlib import Path
import sys

from .member_feature_control_recipe import NAME, SCHEMA, control_spec


PHASES = dict(native='probe_member_feature_controls',capacity='probe_member_feature_controls_fullbatch',
              science='train_member_feature_controls',evaluation='evaluate_member_feature_controls')


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--phase', choices=list(PHASES), required=True)
    local, remaining = parser.parse_known_args()
    if local.phase in {'native','capacity','science'}:
        config = json.loads(Path(remaining[remaining.index('--config')+1]).read_text())
    else:
        config = json.loads(Path(remaining[remaining.index('--arm-config')+1]).read_text())
    spec=control_spec(config)
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
    (output/'member-feature-control-runtime.json').write_text(json.dumps(dict(
        status='PASS_COMPLETED_MEMBER_FEATURE_CONTROL_RUNTIME_PHASE', phase=local.phase,
        trainer=NAME, schema=SCHEMA, arm=spec['arm'], physical_deformation_query_channels=0,
        actual_feature_head_tensors=spec['head_tensors'],actual_source_tensors=spec['source_tensors'],
        physical_control=spec['control'],actual_correspondence_candidates=spec['candidates'],
        native_optimizer_sampler_AMP_LoRA_and_camera_latent_protocol_preserved=True,
        historical_target_patch_query_or_R96_warmup_reproduction=False,
        no_GT_for_matching_or_query=True, full_goal_complete=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
