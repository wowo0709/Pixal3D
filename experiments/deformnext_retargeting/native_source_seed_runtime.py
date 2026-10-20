"""Run seed202 confirmation without editing the sealed seed101 entrypoints.

Scientific trainers already accept202. Their frozen diagnostic entrypoints
pin101, so only their explicit seed guard and initial RNG call are rebound in
this process. The reverse AST must equal the entire original module exactly.
No optimizer, data, sampler, model, precision or mathematical operation changes.
"""
import argparse
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys


PROBES = {
    'native-prior': ('probe_native_prior_source_ray', 'probe_target_first_fullbatch'),
    'member-posterior': ('probe_native_prior_source_ray', 'probe_target_first_fullbatch'),
    'member-feature': ('probe_member_feature_source_ray', 'probe_member_feature_fullbatch'),
    'member-feature-dq': ('probe_member_feature_deformation_source_ray', 'probe_member_feature_deformation_fullbatch'),
    'feature-control': ('probe_member_feature_controls', 'probe_member_feature_controls_fullbatch'),
}
NATIVE = dict(native='probe_native_prior_source_ray', capacity='probe_native_prior_fullbatch',
              science='train_native_prior_source_ray', evaluation='evaluate_native_prior_source_ray')
ROUTERS = {'member-posterior': 'member_posterior_runtime', 'member-feature': 'member_feature_runtime',
           'member-feature-dq': 'member_feature_deformation_runtime',
           'feature-control': 'member_feature_control_runtime'}


def _seed_lookup(node):
    return (
        isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
        and node.value.id == 'cfg' and isinstance(node.slice, ast.Constant)
        and node.slice.value == 'training_seed'
    ) or (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'cfg'
        and node.func.attr == 'get' and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'training_seed'
    )


def _transform(tree, before, after):
    counts = dict(seed_guard=0, initial_RNG=0)

    class Seeds(ast.NodeTransformer):
        def visit_Compare(self, node):
            if (_seed_lookup(node.left) and len(node.ops) == len(node.comparators) == 1
                    and isinstance(node.ops[0], ast.NotEq)
                    and isinstance(node.comparators[0], ast.Constant)
                    and node.comparators[0].value == before):
                node.comparators[0].value = after
                counts['seed_guard'] += 1
            return self.generic_visit(node)

        def visit_Call(self, node):
            if (isinstance(node.func, ast.Name) and node.func.id == 'setup_rng'
                    and len(node.args) == 2 and not node.keywords
                    and all(isinstance(arg, ast.Constant) for arg in node.args)
                    and [arg.value for arg in node.args] == [0, before]):
                node.args[1].value = after
                counts['initial_RNG'] += 1
            return self.generic_visit(node)

    return Seeds().visit(tree), counts


def seed202_ast(source, *, native_probe):
    """Accept exactly the known one guard and optional native-probe RNG site."""
    original = ast.parse(source)
    transformed, counts = _transform(copy.deepcopy(original), 101, 202)
    expected = dict(seed_guard=1, initial_RNG=int(native_probe))
    if counts != expected:
        raise ValueError(f'Frozen diagnostic seed sites differ: {counts}, expected {expected}')
    restored, reverse_counts = _transform(copy.deepcopy(transformed), 202, 101)
    if (reverse_counts != expected
            or ast.dump(restored, include_attributes=False) != ast.dump(original, include_attributes=False)):
        raise ValueError('Seed substitution changes code outside the two declared seed sites')
    ast.fix_missing_locations(transformed)
    return transformed, counts


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--family', choices=list(PROBES), required=True)
    parser.add_argument('--phase', choices=list(NATIVE), required=True)
    args, remaining = parser.parse_known_args()
    config_flag = '--arm-config' if args.phase == 'evaluation' else '--config'
    cfg = json.loads(Path(remaining[remaining.index(config_flag)+1]).read_text())
    if cfg['training_seed'] != 202:
        raise ValueError('This separate confirmation entry requires actual training_seed202')
    # The selected train/evaluation entry enforces its full original recipe.
    # Probe patches exist only while its original phase entry runs.
    replacements = []
    bindings = []
    if args.phase in {'native', 'capacity'}:
        module_name = PROBES[args.family][0 if args.phase == 'native' else 1]
        module = importlib.import_module('.'+module_name, __package__)
        path = Path(module.__file__)
        raw = path.read_bytes()
        transformed, counts = seed202_ast(raw.decode(), native_probe=args.phase == 'native')
        namespace = dict(module.__dict__)
        exec(compile(transformed, str(path)+'[training_seed202]', 'exec'), namespace)
        replacements.append((module, module.main))
        module.main = namespace['main']
        bindings.append(dict(module=module.__name__, original_source_SHA256=hashlib.sha256(raw).hexdigest(),
                             actual_seed_substitutions=counts,
                             substituted_AST_SHA256=hashlib.sha256(ast.dump(transformed, include_attributes=False).encode()).hexdigest(),
                             reverse_AST_equals_original=True))
    previous = sys.argv
    try:
        if args.family == 'native-prior':
            entry = importlib.import_module('.'+NATIVE[args.phase], __package__)
            sys.argv = [sys.argv[0], *remaining]
        else:
            entry = importlib.import_module('.'+ROUTERS[args.family], __package__)
            sys.argv = [sys.argv[0], '--phase', args.phase, *remaining]
        entry.main()
    finally:
        sys.argv = previous
        for module, original_main in replacements:
            module.main = original_main
    output = (Path(remaining[remaining.index('--output')+1])
              if args.phase in {'native', 'evaluation'} else Path(cfg['output_dir']))
    for binding in bindings:
        path = Path(importlib.import_module(binding['module']).__file__)
        if hashlib.sha256(path.read_bytes()).hexdigest() != binding['original_source_SHA256']:
            raise ValueError('Frozen probe source changed during seed confirmation')
    (output/'seed-confirmation-runtime.json').write_text(json.dumps(dict(
        status='PASS_COMPLETED_NATIVE_SOURCE_SEED202_PHASE', phase=args.phase, family=args.family,
        actual_config_training_seed=202, probe_seed_bindings=bindings,
        original_entry_files_not_modified=True,
        source_initialization_seed_and_training_rng_are_not_generation_noise=True,
        optimizer_sampler_AMP_LoRA_data_camera_and_latent_protocol_unchanged=True,
        independent_training_quality_not_yet_proven=True, full_goal_complete=False,
    ), indent=2)+'\n')


if __name__ == '__main__':
    main()
