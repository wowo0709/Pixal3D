"""Explicit original case roots for completed, immutable TEST asset groups.

Only metadata lineage differs from the original single-root loader. Native
cases, terminals, archived pixels, numerical metrics and all their guards
remain unchanged. No case folder or terminal is relocated or synthesized.
"""
import ast
import hashlib
import inspect
from pathlib import Path

from .cascade_view_roles import file_sha256 as sha
from .collect_module_seed_confirmation import read_file, verify_terminal
from .selected_final_case_integrity import CASE_STATUS
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


STATUS = 'PASS_ACTUAL_IMMUTABLE_COMPLETE_TEST_ASSET_GROUP_REUSE'


def validate_manifest(manifest, *, context, parent, models):
    ids = parent['population']['sample_ids']
    reused = manifest.get('reused_sample_ids', [])
    require(manifest.get('status') == STATUS and reused
        and len(reused) == len(set(reused)) and reused == [sid for sid in ids if sid in set(reused)]
        and set(manifest.get('groups', {})) == set(reused)
        and exact(manifest.get('models'), models)
        and exact(manifest.get('original_source_SHA256'), context['source_SHA256'])
        and len(context['source_SHA256']) == 505
        and all(exact(manifest.get(key), context['source_inputs'][key]) for key in ['parent_freeze', 'control_freeze'])
        and manifest.get('actual_same_scored_archive_bytes_verified') is True
        and manifest.get('no_case_or_terminal_relocated_or_synthesized') is True,
        'exact ordered original-model complete-group reuse with actual pixel checks required')
    old = Path(manifest['original_campaign_root'])
    base = Path(context['canonical_local_root'])
    require(old.is_absolute() and old.resolve() == old and old != base and old.is_relative_to(base)
        and old != Path(context['campaign_root']), 'separate original immutable case root required')
    for sid, group in manifest['groups'].items():
        token = hashlib.sha256(sid.encode()).hexdigest()[:24]
        require(set(group['cases']) == set(models)
            and group['group_audit']['path'] == str(old / 'groups' / token / 'audit.json'),
            'actual complete original asset group required')
        for alias, record in group['cases'].items():
            require(record['audit']['path'] == str(old / 'cases' / token / alias / 'audit.json'),
                'reused case keeps its original actual path')
    return reused


def load_manifest(context, check, parent, models, *, verify_groups=False):
    manifest = read_file(context['source_inputs']['case_reuse_manifest'], check)
    validate_manifest(manifest, context=context, parent=parent, models=models)
    for path, digest in manifest['operational_source_SHA256'].items():
        require(sha(path) == digest, 'separately pinned operational reuse source changed')
    require(str(Path(__file__).resolve()) in manifest['operational_source_SHA256'],
        'this metadata adapter must be separately pinned')
    if verify_groups:
        release = read_file(manifest['original_drained_release'], check)
        failure = read_file(manifest['original_failure'], check)
        require(release.get('all_actual_campaign_children_drained') is True
            and release.get('status') == failure.get('status') == 'FAILED_SELECTED_FINAL_CAMPAIGN_PRESERVED',
            'actual preserved failed campaign release required')
        for sid, record in manifest['groups'].items():
            group = read_file(record['group_audit'], check)
            require(group.get('status') == 'PASS_ACTUAL_COMPLETE_ASSET_GROUP_NATIVE_CASES_AND_RAW_RETIREMENT'
                and group.get('sample_id') == sid and exact(group.get('cases'), record['cases'])
                and group.get('raw_duplicates_only') is True, 'actual closed original group lineage required')
            for alias, value in record['cases'].items():
                case = read_file(value['audit'], check)
                terminal = read_file(value['worker_terminal'], check)
                verify_terminal(terminal)
                require(case.get('status') == CASE_STATUS and case.get('actual_completed_native_case') is True
                    and case.get('alias') == alias and case.get('sample_id') == sid
                    and exact(case.get('frozen_model_binding'), models[alias])
                    and terminal.get('case_output') == case.get('case_output') == str(Path(value['audit']['path']).parent)
                    and terminal.get('actual_session_drained') is True,
                    'actual original model and terminal cannot be relabeled')
                for phase in case['terminals']:
                    verify_terminal(phase)
    return manifest


def physical_case_root(runtime, manifest, sid, alias):
    """Resolve only a declared original group or a fresh successor group."""
    root = manifest['original_campaign_root'] if sid in manifest['groups'] else runtime['campaign_root']
    token = hashlib.sha256(sid.encode()).hexdigest()[:24]
    return Path(root) / 'cases' / token / alias


def adapt_case_root(source):
    """Replace one exact metadata path expression and prove every other AST node.

    The frozen loader's actual case, phase terminal, model, noise, GT and pixel
    checks still execute. Unexpected source structure refuses the adaptation.
    """
    tree = ast.parse(source)
    original_expression = "Path(runtime['campaign_root'])/'cases'/hashlib.sha256(sid.encode()).hexdigest()[:24]/alias"
    expected = ast.dump(ast.parse(original_expression, mode='eval').body, include_attributes=False)
    targets = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'root'
        and ast.dump(n.value, include_attributes=False) == expected]
    require(len(targets) == 1, 'exactly one original case-root metadata expression required')
    before = ast.dump(tree, include_attributes=False)
    target = targets[0]
    original = target.value
    target.value = ast.parse('_actual_case_root(runtime, result, sid, alias)', mode='eval').body
    after = ast.dump(tree, include_attributes=False)
    target.value = original
    require(ast.dump(tree, include_attributes=False) == before, 'all other original loader statements must remain identical')
    target.value = ast.parse('_actual_case_root(runtime, result, sid, alias)', mode='eval').body
    ast.fix_missing_locations(tree)
    proof = dict(changed_case_root_expressions=1, all_other_AST_statements_unchanged=True,
        original_AST_SHA256=hashlib.sha256(before.encode()).hexdigest(),
        adapted_AST_SHA256=hashlib.sha256(after.encode()).hexdigest())
    return tree, proof


def loader(context, check, *, pixels=True):
    """Keep the frozen full508 loader and all its numerical calls intact."""
    from . import selected_final_metric_inputs as original
    parent, controls, models = original.frozen_inputs(context, check)
    manifest = load_manifest(context, check, parent, models, verify_groups=True)
    tree, proof = adapt_case_root(inspect.getsource(original.load_campaign))
    require(exact(proof, manifest['original_metric_loader_AST_proof']),
        'actual original metadata-only loader adaptation must match the prepared proof')
    def resolve(runtime, result, sid, alias):
        if sid in manifest['groups']:
            require(exact(result['actual_cases'][sid], manifest['groups'][sid]['cases']),
                'actual aggregate cannot substitute a reused completed case')
        return physical_case_root(runtime, manifest, sid, alias)
    namespace = dict(original.__dict__, _actual_case_root=resolve)
    exec(compile(tree, str(Path(original.__file__).resolve()), 'exec'), namespace)
    return namespace['load_campaign'](context, check, pixels=pixels)
