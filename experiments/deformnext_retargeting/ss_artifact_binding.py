"""Bind nested SS artifacts to the evaluation producer's declared receipt."""
from pathlib import Path

from .cascade_view_roles import require


def closed_ss_artifact_spec(evaluation, artifact, *, check):
    """Use the same receipt for evaluation.json and its recorded NPZ.

    Receipts belong to producer outputs, not necessarily to each subdirectory.
    Never synthesize a receipt or search ancestors for a permissive substitute.
    """
    relative = Path(artifact['artifact'])
    require(not relative.is_absolute() and '..' not in relative.parts
            and relative.suffix == '.npz', 'relative recorded SS NPZ required')
    source, receipt = Path(evaluation['path']), Path(evaluation['receipt'])
    require(check.check(source, receipt) == evaluation['SHA256'],
            'actual evaluation differs from its declared producer receipt')
    path = source.parent / relative
    digest = check.check(path, receipt)
    require(digest == artifact['artifact_SHA256'],
            'actual SS artifact differs from the evaluation record')
    return dict(path=str(path), SHA256=digest, receipt=str(receipt))
