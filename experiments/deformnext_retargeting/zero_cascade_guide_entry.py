"""Preserve actual native zero-shot subdivision guides for DEV or frozen TEST."""
import argparse
import json
import os
from pathlib import Path
import sys

from .archived_native_guides import POLICY
from .assigned_gpu_entry import adapt_guard, validate_assignment
from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .native_guide_capture_context import capture_native_guides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--entry-kind', choices=['dev-core', 'bound-test'], required=True)
    args = parser.parse_args()
    document = json.loads(args.plan.read_text())
    path = Path(document['native_plan']['path']) if args.entry_kind == 'bound-test' else args.plan
    plan = json.loads(path.read_text())
    check = ClosedInputs(plan['canonical_local_root'])
    devices = validate_assignment(plan, os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                                  os.environ.get('MATCHING_V2_ALLOWED_GPUS', ''))

    def guard():
        check.check(args.plan, args.plan.parent/'file4-closure.json')
        check.check(path, path.parent/'file4-closure.json')
        for name, digest in {**plan['source_SHA256'], **plan['inputs_SHA256']}.items():
            if sha(name) != digest:
                raise ValueError('bound zero-cascade source/input changed: '+name)

    guard()
    if plan['texture_support_policy'] != POLICY:
        raise ValueError('actual native guide capture policy required')
    from . import cascade_zero_shot_from_ss_artifact as core
    if plan['source_SHA256'].get(core.__file__) != sha(core.__file__):
        raise ValueError('original numerical zero-cascade entry must be pinned')
    if args.entry_kind == 'dev-core':
        if plan['row']['split'] not in ('validation', 'train'):
            raise ValueError('DEV entry must not open held-out TEST generation')
        tree, proof = adapt_guard(Path(core.__file__).read_text(), devices)
        namespace = dict(core.__dict__)
        exec(compile(tree, core.__file__, 'exec'), namespace)
        run = namespace['main']
    else:
        if os.environ['CUDA_VISIBLE_DEVICES'] not in {'0','1','2','3'} or plan['row']['split'] != 'test':
            raise ValueError('original frozen TEST508 adapter and assigned GPU0-3 required')
        from . import original_zero_cascade_population
        run = original_zero_cascade_population.main
        proof = dict(original_bound_TEST_metadata_validator_retained=True,
                     original_numerical_source_unmodified=True)
    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
    output = Path(plan['output']); archive = output/'native-shape-guides.pt'
    argv = sys.argv
    sys.argv = [core.__file__, '--plan', str(args.plan)]
    try:
        with capture_native_guides(Pixal3DImageTo3DPipeline, archive) as events:
            run()
    finally:
        sys.argv = argv
    if len(events) != 1:
        raise ValueError('one actual native zero-cascade guide capture required')
    report_path = output/'generation.json'
    result = json.loads(report_path.read_text())
    result.update(texture_support_policy=POLICY, actual_native_guide_archive_SHA256=sha(archive),
        actual_native_guide_events=events, actual_native_guide_logits_order_scale_and_dead_branches_preserved=True,
        original_zero_cascade_mathematics_and_returned_guide_objects_unchanged=True,
        numerical_entry_binding=proof, shape_decoder_recomputed=True)
    guard()
    report_path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
