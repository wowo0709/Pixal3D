"""Retain legacy render settings while binding their original environment map."""
from copy import deepcopy

from .cascade_view_roles import file_sha256, require


def bind_render_template_inputs(template, original_case):
    """Import the exact input pin from the actual original case assignment."""
    environment = template['environment_map']
    original = original_case['input_SHA256']
    require(environment in original,
            'original render environment must be pinned in the original case')
    digest = original[environment]
    require(file_sha256(environment) == digest,
            'actual environment differs from the original render input')
    existing = template.get('input_SHA256', {})
    require(isinstance(existing, dict)
            and (environment not in existing or existing[environment] == digest),
            'render template conflicts with the original environment identity')
    result = deepcopy(template)
    result['input_SHA256'] = dict(existing, **{environment: digest})
    return result
