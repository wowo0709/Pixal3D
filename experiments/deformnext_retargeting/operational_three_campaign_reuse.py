"""Join immutable complete TEST groups from three drained physical campaigns.

The scientific loader is unchanged. Only its single case-root metadata
expression resolves to the actual immutable producer of each asset group.
"""
import hashlib
import inspect
from pathlib import Path

from .cascade_view_roles import file_sha256 as sha
from .collect_module_seed_confirmation import read_file, verify_terminal
from .operational_selected_case_reuse import adapt_case_root
from .selected_final_case_integrity import CASE_STATUS
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


STATUS = 'PASS_ACTUAL_IMMUTABLE_THREE_CAMPAIGN_COMPLETE_TEST_GROUP_REUSE'


def validate_manifest(manifest, *, context, parent, models):
    ids = parent['population']['sample_ids']
    reused = manifest.get('reused_sample_ids', [])
    roots = manifest.get('source_campaigns', [])
    source_roots = [row.get('campaign_root') for row in roots]
    base = Path(context['canonical_local_root'])
    require(manifest.get('status') == STATUS and len(roots) == 3
        and len(set(source_roots)) == 3
        and all(Path(p).is_absolute() and Path(p).resolve() == Path(p)
            and Path(p).is_relative_to(base) and p != context['campaign_root'] for p in source_roots)
        and reused and len(reused) == len(set(reused))
        and reused == [sid for sid in ids if sid in set(reused)]
        and set(manifest.get('groups', {})) == set(reused)
        and exact(manifest.get('models'), models)
        and exact(manifest.get('original_source_SHA256'), context['source_SHA256'])
        and len(context['source_SHA256']) == 505
        and all(exact(manifest.get(key), context['source_inputs'][key])
            for key in ['parent_freeze', 'control_freeze'])
        and manifest.get('actual_same_scored_archive_bytes_verified') is True
        and manifest.get('no_case_or_terminal_relocated_or_synthesized') is True,
        'exact ordered, three-producer complete-group reuse required')
    counts = {p: 0 for p in source_roots}
    for sid, group in manifest['groups'].items():
        root = group.get('source_campaign_root')
        token = hashlib.sha256(sid.encode()).hexdigest()[:24]
        require(root in counts and set(group['cases']) == set(models)
            and group['group_audit']['path'] == str(Path(root) / 'groups' / token / 'audit.json'),
            'actual complete group must keep declared physical producer')
        counts[root] += 1
        for alias, record in group['cases'].items():
            require(record['audit']['path'] == str(Path(root) / 'cases' / token / alias / 'audit.json'),
                'reused case keeps exact physical producer path')
    require(all(counts[p] == row['complete_groups'] for p, row in zip(source_roots, roots))
        and sum(counts.values()) == len(reused), 'each producer complete-group count required')
    return reused


def load_manifest(context, check, parent, models, *, verify_groups=False):
    manifest = read_file(context['source_inputs']['case_reuse_manifest'], check)
    validate_manifest(manifest, context=context, parent=parent, models=models)
    for path, digest in manifest['operational_source_SHA256'].items():
        require(sha(path) == digest, 'separately pinned multi-campaign reuse source changed')
    require(str(Path(__file__).resolve()) in manifest['operational_source_SHA256'],
        'multi-producer metadata resolver must be pinned')
    if verify_groups:
        for producer in manifest['source_campaigns']:
            release = read_file(producer['drained_release'], check)
            failure = read_file(producer['failure'], check)
            require(release.get('all_actual_campaign_children_drained') is True
                and release.get('status') == failure.get('status') == 'FAILED_SELECTED_FINAL_CAMPAIGN_PRESERVED',
                'each original failed campaign and actual drain required')
        for sid, record in manifest['groups'].items():
            group = read_file(record['group_audit'], check)
            require(group.get('status') == 'PASS_ACTUAL_COMPLETE_ASSET_GROUP_NATIVE_CASES_AND_RAW_RETIREMENT'
                and group.get('sample_id') == sid and exact(group.get('cases'), record['cases'])
                and group.get('raw_duplicates_only') is True,
                'actual closed source group lineage required')
            for alias, value in record['cases'].items():
                case = read_file(value['audit'], check)
                terminal = read_file(value['worker_terminal'], check)
                verify_terminal(terminal)
                require(case.get('status') == CASE_STATUS and case.get('actual_completed_native_case') is True
                    and case.get('alias') == alias and case.get('sample_id') == sid
                    and exact(case.get('frozen_model_binding'), models[alias])
                    and terminal.get('case_output') == case.get('case_output') == str(Path(value['audit']['path']).parent)
                    and terminal.get('actual_session_drained') is True,
                    'actual source model and terminal cannot be relabeled')
                for phase in case['terminals']:
                    verify_terminal(phase)
    return manifest


def physical_case_root(runtime, manifest, sid, alias):
    root = (manifest['groups'][sid]['source_campaign_root'] if sid in manifest['groups']
        else runtime['campaign_root'])
    token = hashlib.sha256(sid.encode()).hexdigest()[:24]
    return Path(root) / 'cases' / token / alias


def loader(context, check, *, pixels=True):
    from . import selected_final_metric_inputs as original
    parent, controls, models = original.frozen_inputs(context, check)
    manifest = load_manifest(context, check, parent, models, verify_groups=True)
    tree, proof = adapt_case_root(inspect.getsource(original.load_campaign))
    require(exact(proof, manifest['original_metric_loader_AST_proof']),
        'exact original loader one-expression adaptation proof required')
    def resolve(runtime, result, sid, alias):
        if sid in manifest['groups']:
            require(exact(result['actual_cases'][sid], manifest['groups'][sid]['cases']),
                'aggregate cannot substitute a reused complete-group case')
        return physical_case_root(runtime, manifest, sid, alias)
    namespace = dict(original.__dict__, _actual_case_root=resolve)
    exec(compile(tree, str(Path(original.__file__).resolve()), 'exec'), namespace)
    return namespace['load_campaign'](context, check, pixels=pixels)
