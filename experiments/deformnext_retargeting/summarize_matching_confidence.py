"""Plot fixed final head confidence without fitting or selecting thresholds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def fingerprint(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def summarize(root, output, publish=False):
    if output.exists():
        raise FileExistsError('Preserve the earlier confidence diagnosis.')
    output.mkdir(parents=True)
    rows, arrays, inputs = [], {}, []
    selection = read(root/'selected-head.json')
    for arm, audit_name in [('h2', 'audit-final-h1-h2.json'), ('h3', 'audit-final-h3.json')]:
        directory = root/'heads'/arm
        value = read(directory/'validation.json')
        audit_path = root/'head-audit'/audit_name
        audit = next(r for r in read(audit_path)['heads'] if r['arm'] == arm)
        frozen = audit['confidence_calibration']
        expected = audit['confidence']
        logits = torch.tensor(value['confidence_logits'], dtype=torch.float32)
        probability = torch.sigmoid(frozen['slope'] * logits + frozen['bias']).numpy()
        labels = np.asarray(value['selected_correct'], dtype=np.int64)
        if (not np.isfinite(probability).all() or not set(labels.tolist()) <= {0, 1}
                or len(probability) != len(labels) or value['usable'] != value['selected_correct']
                or len(labels) != expected['evaluated_queries']):
            raise ValueError('Actual fixed confidence population differs.')
        accepted = probability >= frozen['threshold']
        precision = float(labels[accepted].mean())
        coverage = float(accepted.mean())
        brier = float(np.mean((probability.astype(np.float64) - labels)**2))
        if (int(accepted.sum()) != expected['accepted'] or precision != expected['precision']
                or coverage != expected['coverage'] or abs(brier-expected['brier']) > 1e-8):
            raise ValueError('CPU recomputation does not reproduce the frozen GPU operating point.')
        if arm == 'h3':
            if (fingerprint(directory/'final.pt')['sha256'] != selection['head_sha256']
                    or frozen['threshold'] != selection['reliability']['threshold']):
                raise ValueError('The selected H3 checkpoint or frozen threshold changed.')
        order = np.argsort(-probability, kind='stable')
        sorted_p = probability[order]
        # A threshold must accept all ties; no artificial within-tie ranking.
        ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(sorted_p)-1]
        counts = ends + 1
        positives = np.cumsum(labels[order])[ends]
        curve_coverage = counts / len(labels)
        curve_precision = positives / counts
        curve_recall = positives / max(1, int(labels.sum()))
        arrays.update({arm+'_threshold': sorted_p[ends], arm+'_accepted': counts,
                       arm+'_true_positive': positives, arm+'_coverage': curve_coverage,
                       arm+'_precision': curve_precision, arm+'_correct_match_recall': curve_recall})
        points = []
        for fraction in [.01, .05, .10, .20, .50, 1.]:
            index = int(np.searchsorted(curve_coverage, fraction))
            points.append({'requested_coverage': fraction, 'threshold': float(sorted_p[ends[index]]),
                'coverage': float(curve_coverage[index]), 'precision': float(curve_precision[index]),
                'accepted': int(counts[index]), 'correct_matches': int(positives[index])})
        rows.append({'arm': arm, 'queries': len(labels), 'correct_matches': int(labels.sum()),
            'frozen_threshold': frozen['threshold'], 'frozen_precision': precision,
            'frozen_coverage': coverage, 'frozen_accepted': int(accepted.sum()), 'brier': brier,
            'all_query_precision': float(labels.mean()), 'curve_points': len(ends),
            'illustrative_coverage_points': points, 'calibration': frozen,
            'checkpoint': fingerprint(directory/'final.pt')})
        inputs += [fingerprint(directory/n) for n in ['validation.json', 'calibration.json', 'recipe.json']]
        inputs.append(fingerprint(audit_path))
    np.savez_compressed(output/'precision-coverage.npz', **arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    for row in rows:
        arm = row['arm']
        axes[0].plot(arrays[arm+'_coverage'], arrays[arm+'_precision'], label=arm.upper())
        axes[0].scatter(row['frozen_coverage'], row['frozen_precision'], s=30, zorder=3)
        axes[1].plot(arrays[arm+'_coverage'], arrays[arm+'_correct_match_recall'], label=arm.upper())
    axes[0].set(xlabel='Accepted fraction of trustworthy foreground queries',
                ylabel='Exact-match precision', title='Fixed head precision vs coverage')
    axes[1].set(xlabel='Accepted fraction of trustworthy foreground queries',
                ylabel='Recall of all correct selected matches', title='Coverage vs correct-match recall')
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=.2)
        axis.legend()
    fig.savefig(output/'precision-coverage.png', dpi=180)
    fig.savefig(output/'precision-coverage.pdf')
    plt.close(fig)
    result = {'status': 'PASS_FIXED_HEAD_PRECISION_COVERAGE_DIAGNOSIS',
        'heads': rows, 'inputs': inputs, 'selected_head': fingerprint(root/'selected-head.json'),
        'probability': 'CPU float32 sigmoid of frozen train-calibration slope*confidence-logit+bias',
        'denominator': 'same trustworthy foreground queries as saved final-head GPU audit; includes visible misses and certified unobservable queries',
        'positive_label': 'selected Top1 exact correctness; not soft-attention utility',
        'tie_policy': 'accept complete equal-probability groups',
        'frozen_operating_points_exact_to_actual_GPU_audits': True,
        'H0_H1_curves_absent_reason': 'confidence prediction was not a trained objective of these arms',
        'threshold_fitting': False, 'validation_selection_use': False, 'optimizer_updates': 0,
        'cached_inference_coverage_not_replaced': True,
        'scope': 'final-epoch head validation confidence diagnostic, not SS generation or unseen test performance'}
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if publish:
        import wandb
        run = wandb.init(entity='youngwoojeon-unist', project='pixal3d-deformnext-retargeting',
            id='ss-matching-v2-confidence-coverage-v1', name='Fixed final head precision coverage',
            resume='never', dir=str(output), config={'scope': result['scope'], 'threshold_fitting': False})
        artifact = wandb.Artifact('ss-matching-v2-confidence-coverage-v1', type='matching-diagnostic')
        for name in ['summary.json','precision-coverage.npz','precision-coverage.png','precision-coverage.pdf']:
            artifact.add_file(str(output/name))
        for row in rows:
            run.log({row['arm']+'/'+key: row[key] for key in
                     ['queries','correct_matches','frozen_precision','frozen_coverage','frozen_accepted','brier']})
        run.log({'precision_coverage': wandb.Image(str(output/'precision-coverage.png'))})
        logged = run.log_artifact(artifact)
        logged.wait()
        version = logged.version
        run.summary.update({'status': result['status'], 'threshold_fitting': False})
        url = run.url
        run.finish()
        (output/'wandb-publication.json').write_text(json.dumps({'status':'PASS_WANDB_FINISHED',
            'run_url':url, 'artifact_version':version, 'independent_readback_required':True}, indent=2)+'\n')
    print(json.dumps({'status':result['status'], 'heads':len(rows), 'threshold_fitting':False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    summarize(args.root, args.output, args.wandb)


if __name__ == '__main__':
    main()
