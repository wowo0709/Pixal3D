"""Readable fixed-order sheets from existing SS projection panels, no inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def fingerprint(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def render(root, output):
    if output.exists():
        raise FileExistsError('Preserve the previous final presentation.')
    study = root/'ss-seed-confirmation-v1'
    confirmation = study/'final-seed101-report.json'
    final = read(confirmation)
    if final['status'] != 'PASS_MATCHED_SS_SEED101_CONFIRMATION' or final['arms'] != ['v1','b3','c1']:
        raise ValueError('The full matched three-arm final is not complete.')
    primary_path = root/'module-study-native-v4/final-primary-report.json'
    primary = read(primary_path)
    if primary['status'] != 'primary_matrix_complete':
        raise ValueError('The fixed zero-shot baseline authority is incomplete.')
    sources = [(name, Path(primary['models'][name]['evaluations'][0]))
               for name in ['zero-source-only','zero-edited-rgb']]
    sources += [(arm, study/'evaluations'/arm/'noise-0/evaluation.json') for arm in ['v1','b3','c1']]
    records, images, inputs = {}, {}, [fingerprint(confirmation),fingerprint(primary_path)]
    identifiers, authority = None, None
    for name, path in sources:
        value = read(path)
        rows = value['per_asset']
        ids = [r['sample_id'] for r in rows]
        if len(rows) != 128 or len(set(ids)) != 128:
            raise ValueError('Expected the complete fixed128 panel.')
        pixels = np.asarray(Image.open(path.parent/'panel.png').convert('RGB'))
        if pixels.shape != (128*64,3*64,3):
            raise ValueError('Expected original128 row GT/Pred/Diff SS64 panels.')
        if identifiers is None:
            identifiers, authority = ids, rows
        if ids != identifiers or any((r['seed'],r['source_cluster_id'],r['tp']+r['fn']) !=
                (a['seed'],a['source_cluster_id'],a['tp']+a['fn']) for r,a in zip(rows,authority)):
            raise ValueError('Baseline/matched panel identity, noise or target counts differ.')
        if images and not np.array_equal(pixels[:,:64],next(iter(images.values()))[:,:64]):
            raise ValueError('Actual GT projection pixels differ between methods.')
        records[name], images[name] = rows, pixels
        inputs += [fingerprint(path),fingerprint(path.parent/'panel.png')]
    for case in range(128):
        hashes = [records[arm][case].get('target_voxels_sha256') for arm in ['v1','b3','c1']]
        if not hashes[0] or len(set(hashes)) != 1:
            raise ValueError('Matched methods require exact actual3D target voxel hashes.')
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    names = [name for name,_ in sources]
    labels = ['GT','Zero: source4','Zero: edited1+source3','V1','B3','C1']
    with PdfPages(output/'all128-fixed-order.pdf') as pdf:
        for start in range(0,128,8):
            fig, axes = plt.subplots(8,6,figsize=(10,12),layout='constrained')
            for row_index, case in enumerate(range(start,start+8)):
                first = images[names[0]][case*64:(case+1)*64]
                axes[row_index,0].imshow(first[:,:64],interpolation='nearest')
                tag = identifiers[case].split('_')[1][:8]
                axes[row_index,0].set_ylabel(f'{case+1}: {tag}',fontsize=8)
                for column,name in enumerate(names,1):
                    tile = images[name][case*64:(case+1)*64,64:128]
                    axes[row_index,column].imshow(tile,interpolation='nearest')
                    record = records[name][case]
                    iou = record['tp']/(record['tp']+record['fp']+record['fn'])
                    axes[row_index,column].set_xlabel(f'IoU64 {iou:.3f}',fontsize=8)
                for column in range(6):
                    axis = axes[row_index,column]
                    axis.set_xticks([])
                    axis.set_yticks([])
                    if row_index == 0:
                        axis.set_title(labels[column],fontsize=9)
            fig.suptitle('Same fixed cases, noise0: SS occupancy max-projection\n'
                         'Numbers are actual3D IoU; silhouettes alone do not establish3D alignment.',fontsize=11)
            pdf.savefig(fig)
            if start == 0:
                fig.savefig(output/'first8-fixed-order.png',dpi=160)
            plt.close(fig)
    manifest = {'status':'PASS_FIXED_FINAL_COMPARISON_PRESENTATION','inputs':inputs,
        'models':names,'cases':128,'sampling_noise_index':0,'PDF_pages':16,
        'display_order':'all fixed128 cases in evaluation order,8 per page; PNG is first8,never selected by gain',
        'sample_ids':identifiers,'all_actual_GT_projection_pixels_exact':True,
        'all_model_sampling_seeds_clusters_and_target_counts_exact':True,
        'matched_three_arm_actual_GT_voxel_hashes_exact':True,
        'zero_shot_3D_GT_hash_coverage':{name:sum(bool(r.get('target_voxels_sha256')) for r in records[name])
                                       for name in names[:2]},
        'presentation_is_2D_projection_not_new_3D_metric':True,
        'physical_axis_name_not_inferred_from_dense_tensor_layout':True,
        'renderer_source':fingerprint(Path(__file__)),
        'optimizer_updates':0,'new_inference':False,
        'scope':'existing source tiles displayed with nearest-neighbor interpolation and labels; no occupancy edits; full3D metrics and all3-noise evaluation remain authoritative'}
    (output/'summary.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':manifest['status'],'cases':128,'PDF_pages':16}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    render(args.root,args.output)


if __name__ == '__main__':
    main()
