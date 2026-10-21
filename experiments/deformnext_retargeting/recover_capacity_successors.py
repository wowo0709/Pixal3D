"""Append fresh successors for unstarted jobs held by a failed capacity probe.

Original failures, HELD records and descriptors stay unchanged. New names use
new successful/prospective capacity jobs; artifact destinations may be reused
only because none of these successors ever started or produced their outputs.
"""
from copy import deepcopy


PIN_MAPS = {'source_SHA256', 'input_SHA256'}


def descendants(jobs, roots):
    result = set(roots)
    while True:
        added = {j['name'] for j in jobs if result.intersection(j.get('depends_on', []))}
        if added <= result:
            return result-set(roots)
        result |= added


def replacement_successors(jobs, state, roots, *, suffix, output_exists):
    by_name = {j['name']: j for j in jobs}
    if len(by_name) != len(jobs) or not roots or not suffix:
        raise ValueError('unique original jobs and explicit failed-capacity replacements required')
    for old, new in roots.items():
        if state.get(old, {}).get('status') != 'FAILED' or old == new or new not in by_name:
            raise ValueError('original failed capacity and separately declared replacement required')
    blocked = descendants(jobs, roots)
    mapping = dict(roots)
    for name in blocked:
        row = state.get(name, {})
        if row.get('status') != 'HELD' or any(k in row for k in ['pid', 'started', 'finished']):
            raise ValueError('only never-started HELD successors can receive fresh aliases: '+name)
        if any(output_exists(p) for p in by_name[name].get('outputs', [])):
            raise ValueError('preserve pre-existing successor outputs: '+name)
        mapping[name] = name+suffix
    if len(set(mapping.values())) != len(mapping) or set(mapping.values())-set(roots.values()) & by_name.keys():
        raise ValueError('fresh unique successor names required')
    new = []
    for original in jobs:
        if original['name'] not in blocked:
            continue
        job = deepcopy(original)
        job['name'] = mapping[original['name']]
        job['depends_on'] = [mapping.get(name, name) for name in original.get('depends_on', [])]
        new.append(job)
    return new, mapping


def rebind_exact(value, replacements, *, owned_receipts=None):
    """Rewrite complete identifiers/paths; pin maps are refreshed separately."""
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [rebind_exact(x, replacements, owned_receipts=owned_receipts) for x in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {}
    for key, item in value.items():
        if key in PIN_MAPS:
            result[key] = deepcopy(item)
        elif key == 'closed_inputs' and owned_receipts is not None:
            result[key] = {}
            for path, receipt in item.items():
                changed = replacements.get(path, path)
                result[key][changed] = owned_receipts.get(changed, replacements.get(receipt, receipt))
        else:
            changed = replacements.get(key, key)
            result[changed] = rebind_exact(item, replacements, owned_receipts=owned_receipts)
    return result
