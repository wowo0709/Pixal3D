"""Capture or reuse actual native Shape guides without another Shape decoding."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from .archived_native_guides import POLICY, save_guides, load_guides
from .assigned_gpu_entry import adapt_guard, validate_assignment
from .cascade_view_roles import ClosedInputs, file_sha256 as sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['capture', 'replay'], required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    devices = validate_assignment(plan, os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                                  os.environ.get('MATCHING_V2_ALLOWED_GPUS', ''))
    check = ClosedInputs(plan['canonical_local_root'])

    def guard():
        check.check(args.plan, args.plan.parent/'file4-closure.json')
        for path, digest in {**plan.get('source_SHA256', {}), **plan.get('input_SHA256', {}),
                             **plan.get('inputs_SHA256', {})}.items():
            if sha(path) != digest:
                raise ValueError('bound source/input changed: '+path)
        for path, receipt in plan.get('closed_inputs', {}).items():
            check.check(path, receipt)

    guard()
    if plan['texture_support_policy'] != POLICY:
        raise ValueError('explicit actual native guide capture/reuse policy required')
    target = ('cascade_from_ss_artifact' if args.kind == 'capture' else 'replay_r_pbr_on_generated_shape')
    module = importlib.import_module('experiments.deformnext_retargeting.'+target)
    source = Path(module.__file__)
    if plan['source_SHA256'].get(str(source)) != sha(source):
        raise ValueError('original numerical entry source must be explicitly pinned')
    tree, device_proof = adapt_guard(source.read_text(), devices)
    namespace = dict(module.__dict__)
    exec(compile(tree, str(source), 'exec'), namespace)
    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
    previous = Pixal3DImageTo3DPipeline.decode_shape_slat
    events = []
    output = Path(plan['output'])
    archive = output/'native-shape-guides.pt' if args.kind == 'capture' else Path(plan['native_guides'])
    if args.kind == 'replay':
        check.check(archive, archive.parent/'file4-closure.json')

    def decode(self, shape, resolution):
        if resolution != 1024 or events:
            raise ValueError('one actual native1024 Shape decode/captured guide reuse required')
        if args.kind == 'capture':
            meshes, guides = previous(self, shape, resolution)
            hashes = save_guides(archive, shape, guides)
        else:
            from pixal3d.modules.sparse import SparseTensor
            guides = load_guides(archive, shape, SparseTensor)
            hashes = sha(archive)
            meshes = []
        events.append(dict(archive=str(archive), tensor_identity=hashes, actual_guide_count=len(guides)))
        return meshes, guides

    Pixal3DImageTo3DPipeline.decode_shape_slat = decode
    argv = sys.argv
    sys.argv = [target, '--plan', str(args.plan)]
    try:
        namespace['main']()
    finally:
        sys.argv = argv
        Pixal3DImageTo3DPipeline.decode_shape_slat = previous
    if len(events) != 1:
        raise ValueError('actual native guide capture/reuse was not executed')
    report_path = output/'generation.json'
    report = json.loads(report_path.read_text())
    report.update(texture_support_policy=POLICY, actual_native_guide_events=events,
        actual_native_guide_archive_SHA256=sha(archive),
        shape_decoder_recomputed=args.kind == 'capture',
        actual_native_guide_logits_order_scale_and_dead_branches_preserved=True,
        numerical_entry_AST_device_literal_only=device_proof,
        no_native_sampling_or_decoder_math_changed=True)
    guard()
    report_path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
