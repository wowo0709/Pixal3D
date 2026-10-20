"""Label-free inputs for predicting whether a native ray has a usable candidate."""
import numpy as np


def inference_features(native, context, prior, valid, addresses):
    """Use only existing inference observations; no GT or visibility arguments."""
    n, k = valid.shape
    if (native.shape != (n, 1024) or context.shape != native.shape
            or prior.shape != valid.shape or addresses.shape != valid.shape
            or valid.dtype != np.bool_ or k != 64):
        raise ValueError('native target rays and aligned semantic Top64 required')
    if ((addresses < 0) | (addresses >= 3072)).any():
        raise ValueError('candidate addresses must belong to three source views')
    if not (np.isfinite(native).all() and np.isfinite(context).all()
            and np.isfinite(prior[valid]).all()):
        raise ValueError('finite actual inference observations required')
    def normalize(x):
        x = x.astype(np.float32)
        return (x-x.mean(-1, keepdims=True))/np.sqrt(x.var(-1, keepdims=True)+1e-5)
    masked = np.where(valid, prior, -np.inf).astype(np.float32)
    maximum = masked.max(-1, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.)
    probability = np.where(valid, np.exp(masked-maximum), 0.)
    probability /= np.maximum(probability.sum(-1, keepdims=True), 1e-8)
    entropy = -(probability*np.log(np.maximum(probability, 1e-30))).sum(-1)
    count = valid.sum(-1)
    ordered = np.sort(masked, axis=-1)
    top_two = np.where((count >= 2)[:, None], ordered[:, -2:], 0.)
    gap = top_two[:, 1]-top_two[:, 0]
    mean = np.where(valid, prior, 0.).sum(-1)/np.maximum(count, 1)
    view_mass = np.stack([(probability*(addresses//1024 == view)).sum(-1)
                          for view in range(3)], -1)
    extra = np.stack([maximum[:, 0]/20., gap/20., mean/20., entropy/np.log(64),
        probability.max(-1), np.exp(entropy)/64., count/64., view_mass.max(-1),
        -(view_mass*np.log(np.maximum(view_mass, 1e-30))).sum(-1)/np.log(3),
        np.sqrt(np.square(native.astype(np.float32)).mean(-1)),
        np.sqrt(np.square(context.astype(np.float32)).mean(-1)),
        (view_mass > 0).sum(-1)/3.], -1).astype(np.float32)
    result = np.concatenate([normalize(native), normalize(context), extra], -1)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite reliability inputs')
    return result


def binary_metrics(labels, probability, threshold):
    y, p = np.asarray(labels, bool), np.asarray(probability, np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite(p).all():
        raise ValueError('nonempty finite binary observations required')
    if ((p < 0) | (p > 1)).any() or not 0 <= threshold <= 1:
        raise ValueError('probabilities and threshold must lie in [0,1]')
    pos, neg = int(y.sum()), int((~y).sum())
    order = np.argsort(p, kind='stable')
    sorted_p = p[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_p))+1, len(p)]
    ranks = np.empty(len(p), np.float64)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[lo:hi]] = (lo+1+hi)/2
    auc = ((ranks[y].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None
    accept = p >= threshold
    safe = np.clip(p, 1e-7, 1-1e-7)
    tpr = float(accept[y].mean()) if pos else None
    fpr = float(accept[~y].mean()) if neg else None
    return dict(rays=len(y), positives=pos, negatives=neg, prevalence=float(y.mean()),
        AUROC=auc, Brier=float(np.square(p-y).mean()),
        NLL=float(-(y*np.log(safe)+(~y)*np.log1p(-safe)).mean()),
        threshold=threshold, positive_recall=tpr, false_acceptance=fpr,
        balanced_accuracy=(tpr+1-fpr)/2 if pos and neg else None,
        accepted_fraction=float(accept.mean()))


def calibration_threshold(labels, probability):
    # A fixed finite grid avoids tuning thresholds on actual validation.
    grid = np.linspace(.05, .95, 19)
    return float(min(grid, key=lambda t: (
        -binary_metrics(labels, probability, float(t))['balanced_accuracy'], abs(t-.5), t)))
