"""Decode only the declared camera-map keys at a cascade JSON boundary."""
from copy import deepcopy


def decode_cascade_records(records):
    """Restore cameras 8/9/10 without changing measured values or row order.

    JSON object keys are strings. The numerical reducer intentionally accepts
    integer camera IDs. Accept either complete representation, reject mixed
    or ambiguous maps, and leave the original records untouched.
    """
    if not isinstance(records, dict) or not records:
        raise ValueError('nonempty model-to-records mapping required')
    result = deepcopy(records)
    for model, rows in result.items():
        if not isinstance(model, str) or not model or not isinstance(rows, list) or not rows:
            raise ValueError('named models with nonempty record lists required')
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('views'), dict):
                raise ValueError('each record requires its explicit views mapping')
            views = row['views']
            integer = set(views) == {8, 9, 10} and all(type(k) is int for k in views)
            serialized = set(views) == {'8', '9', '10'} and all(type(k) is str for k in views)
            if not (integer or serialized):
                raise ValueError('exact homogeneous cameras 8/9/10 required; no camera inference')
            if serialized:
                row['views'] = {int(key): value for key, value in views.items()}
    return result
