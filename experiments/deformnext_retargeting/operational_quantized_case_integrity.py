"""Adapt one frozen final-case GT raster admission expression after TEST GT QA.

Every original case check still runs. Only the continuous .99 containment call
is replaced with verification of the separately archived, uniform pixel gate.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys

from .selected_pipeline_model_contract import require


def adapt_case_validator(original, gate_path):
    require(original.__module__ == 'experiments.deformnext_retargeting.selected_final_case_integrity'
            and original.__name__ == 'validate_case_reports',
            'adapt only the frozen final-case validator')
    source = inspect.getsource(original)
    tree = ast.parse(source)
    before = ast.dump(tree, include_attributes=False)
    expression = "all(r.get('GT_geometry_RGBA_containment',0) >= .99 for r in render['per_view'])"
    expected = ast.dump(ast.parse(expression, mode='eval').body, include_attributes=False)
    targets = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and ast.dump(node, include_attributes=False) == expected]
    require(len(targets) == 1, 'exactly one frozen continuous GT raster gate required')
    target = targets[0]
    replacement = ast.parse('_operational_gt_raster_admission(render[\'per_view\'])', mode='eval').body
    for node in ast.walk(tree):
        for field, value in ast.iter_fields(node):
            if value is target:
                setattr(node, field, replacement)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    if child is target:
                        value[index] = replacement
    require(not any(node is target for node in ast.walk(tree)), 'replace the exact GT gate')
    after = ast.dump(tree, include_attributes=False)
    ast.fix_missing_locations(tree)
    proof = dict(changed_GT_raster_admission_expressions=1,
                 all_other_original_case_checks_unchanged=True,
                 original_validator_AST_SHA256=hashlib.sha256(before.encode()).hexdigest(),
                 adapted_validator_AST_SHA256=hashlib.sha256(after.encode()).hexdigest())
    path = Path(gate_path)

    def accepted(rows):
        gate = json.loads(path.read_text())
        observed = gate.get('observations', [])
        require(gate.get('status') == 'PASS_UNIFORM_INTEGER_PIXEL_GT_RASTER_QA'
                and len(rows) == len(observed) == 3
                and [row['camera_id'] for row in rows] == [8, 9, 10],
                'three original camera reports and real uniform GT QA proof required')
        for row, item in zip(rows, observed):
            n, outside, allowance = (item[k] for k in
                ('GT_alpha_pixels', 'outside_one_pixel_raster', 'integer_ceil_one_percent_allowance'))
            require(type(n) is int and type(outside) is int and type(allowance) is int
                    and n > 0 and 0 <= outside <= allowance == (n + 99) // 100
                    and row['GT_geometry_RGBA_mask_iou'] == item['IoU']
                    and row['GT_geometry_RGBA_containment'] == item['containment'],
                    'real report values and all-view integer pixel GT admission must agree')
        return True

    namespace = dict(sys.modules[original.__module__].__dict__,
                     _operational_gt_raster_admission=accepted)
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), namespace)
    return namespace['validate_case_reports'], proof
