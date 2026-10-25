"""Export prespecified final comparison figures from actual scored archives.

The DEV freeze, original declaration, complete ten-metric report and original
float/uint8 image identities are prerequisites. No model execution occurs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import textwrap

import numpy as np

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .collect_module_seed_confirmation import read_file
from .extract_official_inception import render_uint8
from .final_qualitative_selection import select_cases, validate_declaration
from .metric_image_identity import scored_image_observation
from .selected_final_metric_summary import EIGHT_STATUS, TEN_STATUS, validate_contract
from .selected_final_population import compose_models
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


SCHEMA = 'final-qualitative-closed-TEST508-inputs-v1'
STATUS = 'PASS_ACTUAL_SCORED_IMAGE_FINAL_QUALITATIVE_FIGURES'


def required_sources():
    root = Path(__file__).resolve().parent
    return [str(root / (name + '.py')) for name in ['export_final_qualitative', 'final_qualitative_selection',
        'cascade_view_roles', 'collect_module_seed_confirmation', 'metric_image_identity',
        'extract_official_inception', 'selected_final_metric_summary', 'selected_final_population',
        'selected_pipeline_freeze_v2', 'selected_pipeline_model_contract']]


def frozen_admission(context, check):
    """Check the actual pre-TEST freeze before opening either quality report."""
    require(context.get('schema') == SCHEMA and set(required_sources()) <= set(context['source_SHA256']),
        'explicit complete qualitative exporter source binding required')
    for path, digest in context['source_SHA256'].items():
        require(sha(path) == digest, 'qualitative source bytes changed')
    inputs = context['source_inputs']
    parent = read_file(inputs['parent_freeze'], check)
    controls = read_file(inputs['control_freeze'], check)
    models = compose_models(parent, controls)
    contract = validate_contract(parent)
    protocol = parent['evaluation_protocol']
    require(exact(protocol.get('qualitative_selection_plan'), inputs['selection_plan'])
        and all(protocol['source_SHA256'].get(path) == digest for path, digest in context['source_SHA256'].items()),
        'freeze this exact original qualitative declaration and exporter before any TEST prediction')
    declared = read_file(inputs['selection_plan'], check)
    validate_declaration(declared, parent['population'])
    require(all(inputs['test_manifest'][k] == parent['population']['manifest'][k] for k in ['path', 'SHA256']),
        'actual frozen TEST metadata manifest required')
    metadata = read_file(inputs['test_manifest'], check, lines=True)
    fields = ['sample_id', 'source_uid', 'source_cluster_id', 'split']
    require([{k: row[k] for k in fields} for row in metadata] == declared['all_metadata'],
        'predeclared metadata must match the actual original TEST manifest')
    return parent, controls, models, contract, declared


def report_admission(eight, ten, *, parent, models, contract, inputs):
    require(eight.get('status') == EIGHT_STATUS and ten.get('status') == TEN_STATUS
        and eight.get('scope') == ten.get('scope') == 'final-test508'
        and eight.get('assets') == ten.get('assets') == 508
        and eight.get('source_clusters') == ten.get('source_clusters') == 134
        and exact(eight.get('models'), models) and exact(ten.get('models'), models)
        and exact(eight.get('metric_contract'), contract) and exact(ten.get('metric_contract'), contract)
        and exact(ten.get('role_statistics'), eight['statistics'])
        and ten.get('model_selection_from_test') is False and eight.get('model_selection_from_test') is False
        and ten.get('all_ten_metrics_collected') is True,
        'actual complete same-model TEST508 ten metrics and exact role scores required')
    require(all(exact(ten['source_inputs'].get(k), inputs[k]) for k in ['parent_freeze', 'control_freeze'])
        and ten.get('same_image_join', {}).get('status') == 'PASS_SAME_ACTUAL_SCORED_RGB_ACROSS_ALL_TEN_METRICS'
        and ten['same_image_join'].get('native_float_RGB_and_official_fullframe_uint8_identities_verified') is True
        and ten['same_image_join'].get('shared_reference_RGB_and_alpha_identical_across_models') is True,
        'use the actual frozen ten-metric same-image result, not a replacement render')
    ids = parent['population']['sample_ids']; clusters = parent['population']['source_clusters_by_sample']
    expected = [(sid, cid) for sid in ids for cid in [8, 9, 10]]
    require(set(eight['scored_image_observations']) == set(models), 'all scored physical models required')
    index = {}
    for alias in models:
        rows = eight['scored_image_observations'][alias]
        require([(r['sample_id'], r['camera_id']) for r in rows] == expected
            and len({r['archive'] for r in rows}) == 1524
            and all(r['source_cluster_id'] == clusters[r['sample_id']]
                and eight['input_SHA256'].get(r['archive']) == r['archive_SHA256'] for r in rows),
            'all1524 ordered actual scored sample/camera/source/archive identities required')
        index[alias] = {(r['sample_id'], r['camera_id']): r for r in rows}
    return index


def read_scored_image(expected, check):
    """Use the same float arrays and official conversion as the ten metrics."""
    path = Path(expected['archive'])
    require(path.stem == f"camera{expected['camera_id']:02d}", 'displayed camera must match the actual scored archive name')
    digest = check.check(path, path.parent / 'file4-closure.json')
    require(digest == expected['archive_SHA256'], 'scored archive differs from the actual metric input')
    with np.load(path, allow_pickle=False) as arrays:
        pixels = {k: arrays[k].copy() for k in ['prediction', 'GT_shared_renderer', 'GT_alpha']}
    observation = scored_image_observation(pixels, sample_id=expected['sample_id'], camera_id=expected['camera_id'],
        source_cluster_id=expected['source_cluster_id'], archive=str(path), archive_SHA256=digest)
    require(exact(observation, expected), 'actual displayed float RGB, alpha or uint8 conversion differs from scoring')
    return pixels, observation


def case_images(sid, *, models, index, check):
    images, references, observed = {}, {}, []
    for alias in models:
        for camera in [8, 9, 10]:
            pixels, observation = read_scored_image(index[alias][sid, camera], check)
            reference = (observation['reference_float_RGB_SHA256'], observation['reference_alpha_SHA256'])
            require(references.setdefault(camera, reference) == reference, 'every displayed model must use identical GT RGB and alpha')
            images.setdefault(('GT', camera), render_uint8(pixels['GT_shared_renderer']).transpose(1, 2, 0))
            images[alias, camera] = render_uint8(pixels['prediction']).transpose(1, 2, 0)
            observed.append(dict(alias=alias, **observation))
    return images, observed


def render_case(case, *, images, models, output, number, selected_alias, comparator):
    """Uniform full-frame scientific plot; annotations never recompute metrics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'zero-source4': 'Zero-shot\nsource RGB4', 'zero-edited4': 'Zero-shot\nedited RGB4',
        'V1': 'Vanilla V1', 'RETARGET': 'Top-1 R', 'SELECTED': 'Selected\nmodel'}
    aliases = [None, *models]; figure, axes = plt.subplots(3, len(aliases),
        figsize=(7.2, 5.4))
    # Reserve a common caption/title band. Repeated constrained-layout passes
    # across PDF/PNG backends can otherwise move the selection title off-page.
    figure.subplots_adjust(left=.07, right=.99, bottom=.15, top=.85, wspace=.18, hspace=.18)
    for row, camera in enumerate([8, 9, 10]):
        for col, alias in enumerate(aliases):
            ax = axes[row, col]
            pixels = images['GT', camera] if alias is None else images[alias, camera]
            ax.imshow(pixels, interpolation='nearest', vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title('GT: M1\noriginal material' if alias is None else labels[alias], fontsize=7)
            if col == 0:
                ax.set_ylabel(('Target' if camera == 8 else 'Source') + f' camera {camera}', fontsize=7)
            if row == 2 and alias is not None:
                values = case['per_model_metrics'][alias]
                ax.set_xlabel(f"LPIPS {values['LPIPS']:.4f}\nDS {values['DreamSim']:.4f}\n"
                    f"CD {values['CD']:.4f}\nF {values['FScore']:.4f}", fontsize=6.5)
    figure.suptitle(textwrap.fill(case['sample_id'], 110) + '\nSelection: ' + ', '.join(case['categories'])
        + f'; contrast {selected_alias} minus {comparator}', fontsize=8)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer(); width, height = figure.canvas.get_width_height()
    text = [figure._suptitle] + [item for ax in axes.flat for item in [ax.title, ax.xaxis.label, ax.yaxis.label] if item.get_text()]
    require(all((box := item.get_window_extent(renderer)).x0 >= 0 and box.y0 >= 0
        and box.x1 <= width and box.y1 <= height for item in text), 'qualitative labels must fit the common full-page canvas')
    stem = f'example-{number:02d}'
    figure.savefig(Path(output) / (stem + '.pdf'), dpi=300)
    figure.savefig(Path(output) / (stem + '.png'), dpi=300)
    plt.close(figure)
    return [stem + '.pdf', stem + '.png']


def export(path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit CPU-only qualitative export required')
    path = Path(path); context = json.loads(path.read_text())
    require(context.get('repo') == str(Path(__file__).resolve().parents[2]), 'use the current actual source checkout')
    affinity = context['CPU_affinity']
    require(bool(affinity) and all(type(c) is int for c in affinity)
        and set(affinity) <= os.sched_getaffinity(0), 'explicit available metadata CPU affinity required')
    os.sched_setaffinity(0, set(affinity)); os.nice(10)
    base = Path(context['canonical_local_root']); check = ClosedInputs(base)
    check.check(path, path.parent / 'file4-closure.json')
    parent, controls, models, contract, declared = frozen_admission(context, check)
    inputs = context['source_inputs']
    eight = read_file(inputs['eight_metrics'], check); ten = read_file(inputs['ten_metrics'], check)
    index = report_admission(eight, ten, parent=parent, models=models, contract=contract, inputs=inputs)
    selection = select_cases(declared=declared, population=parent['population'],
        selected_alias=parent['selected_alias'], models=models, statistics=eight['statistics'])
    output = Path(context['output'])
    require(output.is_absolute() and output != base and output.resolve() == output
        and output.is_relative_to(base) and not output.exists(), 'new owned canonical figure output required')
    output.mkdir(parents=True)
    observed = []; figures = []
    for number, case in enumerate(selection['cases'], 1):
        sid = case['sample_id']
        images, case_observations = case_images(sid, models=models, index=index, check=check)
        observed.extend(case_observations)
        figures.append(dict(sample_id=sid, categories=case['categories'], outputs=render_case(case,
            images=images, models=models, output=output, number=number,
            selected_alias=selection['selected_alias'], comparator=selection['comparator'])))
    def write(name, value):
        with (output / name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    for source, digest in context['source_SHA256'].items():
        require(sha(source) == digest, 'qualitative source changed during export')
    for source, digest in check.inputs.items():
        require(sha(source) == digest, 'closed qualitative input changed during export')
    write('selection.json', selection)
    caption = ('Actual frozen TEST examples: rows are target camera8 and source cameras9/10; '
        'columns show the common original-material M1 reference and every distinct frozen model. '
        'Full-frame pixels use the official rounded/clamped uint8 conversion. '
        'No model-specific crop, alignment or additional rendering is used. '
        'Source LPIPS/DreamSim annotations average cameras9/10; CD/F-Score describe the generated surface. '
        'The two outcome-blind examples are retained regardless of quality. '
        'Conditional lower-median examples illustrate stated conditions and do not estimate success rates.\n')
    for category, row in selection['categories'].items():
        caption += f"{category}: {row['eligible_cases']} eligible; selected {row['selected_sample_id']}.\n"
    (output / 'figure-captions.txt').write_text(caption)
    audit = dict(status=STATUS, figures=figures, actual_scored_image_observations=observed,
        source_inputs=inputs, source_SHA256=context['source_SHA256'], verified_inputs=check.inputs,
        exact_original_float_and_official_uint8_images=True, shared_GT_RGB_and_alpha_exact=True,
        source_metric_cameras=[9, 10], displayed_cameras=[8, 9, 10],
        model_specific_crop_alignment_or_new_render=False, CPU_model_forward_or_training_updates=0,
        conditional_examples_not_population_success_rates=True, full_goal_complete=False)
    write('audit.json', audit)
    manifest = Path(context['manifest_directory']) / (context['name'] + '-files.json')
    require(not manifest.exists(), 'new owned figure closure manifest required')
    manifest.write_text(json.dumps([str(p.relative_to(base)) for p in sorted(output.rglob('*')) if p.is_file()], indent=2) + '\n')
    subprocess.run([*context['closure_command'], '--manifest', str(manifest),
        '--receipt', str((output / 'file4-closure.json').relative_to(base)),
        '--scope', 'Actual frozen TEST qualitative examples under predeclared metadata/conditional selection; exact scored images, no new NN/render or population-effect claim'], check=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan', type=Path, required=True)
    result = export(parser.parse_args().plan)
    print(json.dumps(dict(status=result['status'],figures=result['figures'])), flush=True)


if __name__ == '__main__':
    main()
