"""Explicit source-input binding for the unchanged native SS96 phases.

GPU mechanics/capacity and source-disjoint quality confirmation remain
separate admission requirements; availability of this entry grants neither.
"""
import argparse
import json
from pathlib import Path
import sys

from .source_teacher_initialization import source_input_binding


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--source-input-design', choices=['original-NoDQ', 'source-contrast-NoDQ'], required=True)
    parser.add_argument('--source-teacher-init-spec', type=Path)
    local, remaining = parser.parse_known_args()
    phase = remaining[remaining.index('--phase')+1]
    if phase not in {'native', 'capacity', 'science'}:
        raise ValueError('native mechanics/capacity/fresh joint SS phases only')
    config = json.loads(Path(remaining[remaining.index('--config')+1]).read_text())
    experiment = config['experiment']
    if (experiment.get('source_input_design') != local.source_input_design
            or experiment.get('additional_source_teacher_supervision') is not (local.source_teacher_init_spec is not None)
            or config.get('load_dir') != '' or config.get('load_ckpt') is not None):
        raise ValueError('explicit fresh source-input design and teacher supervision declaration required')
    if phase == 'native' and local.source_teacher_init_spec is not None:
        raise ValueError('cold native identity probe cannot assert identity for a warm source teacher')
    teacher = (json.loads(local.source_teacher_init_spec.read_text())
               if local.source_teacher_init_spec is not None else None)
    from pixal3d.trainers.flow_matching import member_feature_source_ray as trainer_module
    from . import member_feature_runtime
    argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        with source_input_binding(trainer_module, variant=local.source_input_design, teacher=teacher) as instances:
            member_feature_runtime.main()
    finally:
        sys.argv = argv
    if not instances or any(module._source_input_initialization is None for module in instances):
        raise ValueError('actual initialized source conditioner required')
    output = (Path(remaining[remaining.index('--output')+1]) if phase == 'native'
              else Path(config['output_dir']))
    (output/'source-input-joint-runtime.json').write_text(json.dumps(dict(
        status='PASS_COMPLETED_NATIVE_SOURCE_INPUT_PHASE', phase=phase,
        source_input_design=local.source_input_design,
        initialization=[module._source_input_initialization for module in instances],
        native_SS_LoRA_optimizer_sampler_AMP_and_epoch_validation_unchanged=True,
        teacher_labels_not_loaded_by_joint_dataset_or_loss=True,
        actual_GPU_validation_requires_separate_reports=True,
        semantic_matching_score_decoder_or_attention_effect_not_established=True,
        model_adopted=False, full_goal_complete=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
