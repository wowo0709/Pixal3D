"""Explicit saved-model origins; historical labels never imply copied weights."""
from pathlib import Path


SEMANTIC_CANDIDATES = ['soft-topk', 'static-c1', 'task-full',
                       'task-no-decoder', 'task-no-prior', 'previous-c1']


def candidate_source(root, name):
    root = Path(root)
    if name == 'previous-c1':
        return root.parent/'ss-seed-confirmation-v1', 'c1'
    if name in SEMANTIC_CANDIDATES or name == 'top1-reference':
        return root, name
    raise ValueError('unknown semantic TopK candidate: '+name)


def additional_development_label(name):
    if name not in SEMANTIC_CANDIDATES:
        raise ValueError('unknown semantic TopK candidate: '+name)
    return name if name == 'previous-c1' else 's101-'+name
