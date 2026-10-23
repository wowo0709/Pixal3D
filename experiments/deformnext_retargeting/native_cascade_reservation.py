"""Validate operational wrappers without changing native cascade arguments."""
from pathlib import Path


INNER = {
    'case': ('run_native_matched_fullcascade_case_v1.py',
             'PASS_ACTUAL_NATIVE_SS480_AND_MATCHED_PBR1900_FULLCASCADE_CASE'),
    'official': ('run_native_fullcascade_analysis_v1.py',
                 'PASS_ACTUAL_NATIVE_FOUR_SS_PAIRED_PBR768_OFFICIAL_METRICS'),
}


def validate_envelope(plan, inner_document, *, physical_gpu, allowed_gpus, sha):
    kind = plan.get('inner_kind')
    if (kind not in INNER or physical_gpu not in {'0','1','2','3'}
            or physical_gpu not in allowed_gpus):
        raise ValueError('one explicitly authorized GPU0-3 and native GPU producer required')
    command = plan.get('inner_command', [])
    if (len(command) != 7 or command[:2] != [plan['python'],'-u']
            or command[3:6:2] != ['--plan','--expected-self-sha']
            or Path(command[2]).name != INNER[kind][0]
            or command[4] != plan['inner_plan']
            or command[6] != plan['source_SHA256'].get(command[2])
            or sha(command[2]) != command[6]
            or sha(command[4]) != plan['inner_plan_SHA256']):
        raise ValueError('unchanged pinned original worker and exact argument vector required')
    if plan.get('inner_read_contract') != inner_document:
        raise ValueError('pending consumer scan must retain the complete original read contract')
    if kind == 'official' and inner_document.get('phase') != 'perceptual':
        raise ValueError('only the native GPU official metric phase may use a GPU wrapper')
    base, owned, out = [Path(plan[k]) for k in ['canonical_local_root','owned_guard_root','output']]
    inner_out = Path(inner_document['output'])
    if (not all(p.is_absolute() for p in [base,owned,out,inner_out])
            or owned == base or not owned.is_relative_to(base)
            or out == owned or not out.is_relative_to(owned)
            or inner_out == out or out.is_relative_to(inner_out) or inner_out.is_relative_to(out)
            or any(p.resolve() != p for p in [base,owned,out,inner_out])):
        raise ValueError('separate case-owned operational output under the canonical root required')
    expected = [dict(path=str(inner_out/'audit.json'),receipt=str(inner_out/'file4-closure.json'),
                     status=INNER[kind][1])]
    if plan.get('expected_inner_artifacts') != expected:
        raise ValueError('actual original producer audit and canonical receipt required')
    return expected
