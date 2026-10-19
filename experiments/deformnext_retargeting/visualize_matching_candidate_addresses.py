"""Plot actual fixed-panel semantic candidate addresses, without GT-match labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from pixal3d.modules.r9_matching import PromotedR9Matcher, categorical_weights
from pixal3d.modules.task_matching_attention import TaskCandidateAttention, contextual_query


def fingerprint(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def local(path, path_map):
    for prefix, replacement in path_map.items():
        if path == prefix or path.startswith(prefix + '/'):
            return Path(replacement + path[len(prefix):])
    raise ValueError('visualization input has no declared path mapping')


def rgb(path):
    with Image.open(path) as image:
        if image.size != (512, 512):
            raise ValueError('prepared 512-pixel canvas required; no implicit crop/resize')
        rgba = np.asarray(image.convert('RGBA'), dtype=np.float32) / 255
    return rgba[..., :3] * rgba[..., 3:] + 1 - rgba[..., 3:]


def run(args):
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    root = args.root
    output = root / 'matching-address-first8-v1'
    if output.exists():
        raise FileExistsError('preserve actual correspondence inspection')
    qa_path = root / 'attention-statistics-v1/task-full/summary.json'
    qa = json.loads(qa_path.read_text())
    cfg_path = root / 'configs/task-full.json'
    cfg = json.loads(cfg_path.read_text())
    reference_path = root / 'matching-prior-drift-panel128-v1/summary.json'
    reference = json.loads(reference_path.read_text())
    if qa['status'] != 'PASS_TRAINED_SS_CACHE_AMP_CFG_CONTRACT':
        raise ValueError('actual saved-weight GPU proof required')
    for name in ['panel', 'manifest', 'trained_conditioner']:
        if fingerprint(qa[name]['path']) != qa[name]:
            raise ValueError('saved proof input bytes changed: ' + name)
    panel = json.loads(Path(qa['panel']['path']).read_text())['sample_ids']
    if len(panel) != len(set(panel)) or len(panel) != 128:
        raise ValueError('original fixed128 panel required')
    rows = {row['sample_id']: row for row in map(json.loads, open(qa['manifest']['path']))}
    reference_cases = {row['sample_id']: row for row in reference['per_case_contrasts']}
    saved = torch.load(qa['trained_conditioner']['path'], map_location='cpu', weights_only=True)
    matcher = PromotedR9Matcher().eval()
    matcher.load_state_dict({key.removeprefix('q2c.'): value for key, value in saved.items()
                             if key.startswith('q2c.')}, strict=True)
    attention = TaskCandidateAttention().eval()
    attention.load_state_dict({key.removeprefix('task_attention.'): value for key, value in saved.items()
                               if key.startswith('task_attention.')}, strict=True)
    path_map = cfg['validation']['dataset_args']['path_map']
    output.mkdir(parents=True)
    cases = []
    with PdfPages(output / 'correspondences.pdf') as pdf:
        for number, sid in enumerate(panel[:8], 1):
            row = rows[sid]
            if row['conditioning_camera_ids'] != [8, 9, 10, 11] or row['target_anchor_kind'] != 'rgb':
                raise ValueError('four RGB camera contract differs')
            transport_path = local(row['transport_cache'], path_map)
            match_path = Path(cfg['validation']['dataset_args']['match_root']) / (sid + '.npz')
            image_paths = [local(path, path_map) for path in [row['target_rgb_input'], *row['source_images']]]
            if len(image_paths) != 4 or image_paths[0] != local(row['target_anchor'], path_map):
                raise ValueError('prepared RGB target/source identity differs')
            with np.load(transport_path) as archive:
                if str(archive['conditioning_input_sha256']) != row['conditioning_input_sha256']:
                    raise ValueError('transport RGB identity differs')
                query = torch.from_numpy(archive['target_keys'].copy()).float()[..., :1024]
                keys = torch.from_numpy(archive['source_keys'].copy()).float()[..., :1024].flatten(0, 1)
                values = torch.from_numpy(archive['source_values'].copy()).float().flatten(0, 1)
                native = torch.from_numpy(archive['target_query'].copy()).float()
                source_valid = torch.from_numpy(archive['source_valid'].copy()).bool().flatten()
            with np.load(match_path) as archive:
                if str(archive['conditioning_input_sha256']) != row['conditioning_input_sha256']:
                    raise ValueError('match RGB identity differs')
                indices = torch.from_numpy(archive['semantic_indices'].copy()).long()
                valid = torch.from_numpy(archive['candidate_valid'].copy()).bool() & source_valid[indices]
                cached = torch.from_numpy(archive['matching_logits'].copy()).float()
            with Image.open(image_paths[0]) as image:
                if image.size != (512, 512):
                    raise ValueError('target canvas differs')
                alpha = np.asarray(image.convert('RGBA'))[8::16, 8::16, 3].reshape(-1) > 127
            valid &= torch.from_numpy(alpha.copy())[:, None] & (source_valid.sum() >= 64)
            active = valid.any(-1)
            query_ids = active.nonzero().flatten()
            ids, mask = indices[active], valid[active]
            if ids.shape != (len(query_ids), 64) or not bool(mask.all()) or keys.shape != (3072, 1024):
                raise ValueError('expected foreground64 and3x1024 source patch maps')
            context = contextual_query(matcher.decoder, query[active], keys[ids], mask)
            nq, nk = F.normalize(context, dim=-1), F.normalize(keys[ids], dim=-1)
            cosine = (nq[:, None] * nk).sum(-1)
            score = matcher.fusion_head(nq, nk, cosine[..., None], cosine)
            _, learned, _ = attention(native[active], context, values[ids], mask, score / .05)
            frozen005 = categorical_weights(cached[active], mask, .005)
            frozen05 = categorical_weights(cached[active], mask, .05)
            changed = float((learned.argmax(-1) != frozen05.argmax(-1)).float().mean())
            old = reference_cases[sid]
            expected = old['contrasts']['trained_total_T05-vs-cached_static_T05']['candidate_argmax_changed_fraction']
            if old['active_queries'] != len(query_ids) or abs(changed - expected) > 1e-7:
                raise ValueError('actual address choices do not replay the saved128 diagnostic')
            chosen = np.floor((np.arange(4) + .5) * len(query_ids) / 4).astype(int)
            images = [rgb(path) for path in image_paths]
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle(f'{number:02d} {row["category"]} | green=frozen Top1; cyan=frozen ranks2-4; magenta=trained Full Top1\n'
                         'Original panel order; four fixed foreground quantiles. Locations are model choices, not GT matches.', fontsize=11)
            selections = []
            for plot_row, chosen_index in enumerate(chosen):
                query_id = int(query_ids[chosen_index])
                frozen_rank = frozen005[chosen_index].argsort(descending=True)[:4]
                full_rank = learned[chosen_index].argsort(descending=True)[:4]
                selected = {'query_patch': query_id, 'query_row_col': [query_id // 32, query_id % 32],
                            'frozen_T005_top4': [], 'trained_Full_T05_top4': []}
                for group, ranks, weights in [('frozen_T005_top4', frozen_rank, frozen005),
                                              ('trained_Full_T05_top4', full_rank, learned)]:
                    for rank in ranks:
                        address = int(ids[chosen_index, rank])
                        selected[group].append({'candidate_rank_in_semantic64': int(rank), 'source_camera_id': 9 + address // 1024,
                                                'source_patch_row_col': [(address % 1024) // 32, address % 32],
                                                'weight': float(weights[chosen_index, rank]),
                                                'frozen_T05_weight': float(frozen05[chosen_index, rank])})
                for column, image in enumerate(images):
                    ax = axes[plot_row, column]
                    ax.imshow(image)
                    ax.set_axis_off()
                    if column == 0:
                        y, x = selected['query_row_col']
                        ax.add_patch(Rectangle((16*x, 16*y), 16, 16, fill=False, edgecolor='orange', linewidth=2))
                        ax.set_title(f'target M1 camera8 | q=({y},{x})', fontsize=9)
                    else:
                        ax.set_title(f'source M0 camera{8+column}', fontsize=9)
                        for rank_number, candidate in enumerate(selected['frozen_T005_top4'], 1):
                            if candidate['source_camera_id'] != 8 + column:
                                continue
                            y, x = candidate['source_patch_row_col']
                            color = '#159447' if rank_number == 1 else '#00a9c9'
                            ax.add_patch(Rectangle((16*x, 16*y), 16, 16, fill=False, edgecolor=color,
                                                   linewidth=2 if rank_number == 1 else 1))
                            ax.text(16*x+8, 16*y-4, str(rank_number), color=color, fontsize=8,
                                    ha='center', bbox={'facecolor': 'white', 'alpha': .7, 'edgecolor': 'none', 'pad': .2})
                        full = selected['trained_Full_T05_top4'][0]
                        if full['source_camera_id'] == 8 + column:
                            y, x = full['source_patch_row_col']
                            ax.add_patch(Rectangle((16*x-2, 16*y-2), 20, 20, fill=False,
                                                   edgecolor='magenta', linewidth=1.5, linestyle='--'))
                selections.append(selected)
            fig.tight_layout(rect=(0, 0, 1, .94))
            filename = f'case{number:02d}.png'
            fig.savefig(output / filename, dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
            cases.append({'sample_id': sid, 'source_cluster_id': row['source_cluster_id'],
                          'category': row['category'], 'active_queries': len(query_ids),
                          'all_query_Full_vs_frozen_argmax_changed_fraction': changed,
                          'actual_saved128_address_choice_replay': True,
                          'displayed_query_selection': 'four mid-quantiles of row-major active foreground patches',
                          'displayed_queries': selections, 'figure': filename,
                          'inputs': [fingerprint(path) for path in [transport_path, match_path, *image_paths]]})
            print(json.dumps({'case': number, 'category': row['category'], 'active_queries': len(query_ids),
                              'argmax_changed_fraction': changed}), flush=True)
    result = {'status': 'PASS_ACTUAL_FIRST8_MATCHING_ADDRESS_VISUALIZATION',
              'cases': cases, 'panel': qa['panel'], 'saved_conditioner': qa['trained_conditioner'],
              'manifest': qa['manifest'], 'original_saved128_diagnostic': fingerprint(reference_path),
              'driver': fingerprint(Path(__file__)), 'optimizer_updates': 0, 'GPU_used': False,
              'scope': 'First8 original development cases in fixed order;fixed4foreground-query quantiles. Frozen andactualjoint-trained Full choices on same semantic64 support;no GT correspondence labels or reserved121 inputs. Address disagreement is not matching accuracy or a quality contribution.',
              'model_adopted': False, 'full_goal_complete': False}
    with (output / 'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(result['status'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
