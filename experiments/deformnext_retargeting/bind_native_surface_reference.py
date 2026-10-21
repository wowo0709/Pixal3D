"""Join post-generation native surface evaluation to shared stored GT points.

Only native producer metadata and evaluation-only source/target geometry are
read. The native prediction, sampler and scores are not changed or recomputed.
Large remote prediction archives need not be transferred for this GT join.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .cascade_remote_results import array_identities
from .cascade_view_roles import require
from .evaluate_bound_population import pinned, sha
from .prepare_surface_reference import surface_reference


def target_identity(reference, *, source_vertices, source_faces, target_vertices,
                    reference_arrays, native_geometry_arrays):
    """Reproduce exact GT samples and match the actual producer's array hashes."""
    expected = surface_reference(source_vertices, source_faces, target_vertices, source_faces)
    require(reference['point_count'] == 20000 and reference['surface_seed'] == 101
            and reference['target_faces_equal_source'] and reference['final_target_vertices_exact'],
            'fixed20k/101 source-topology GT reference required')
    for key in ['target_points','source_center','source_extent']:
        require(reference_arrays[key].dtype == expected[key].dtype
                and np.array_equal(reference_arrays[key], expected[key]),
                'shared GT point/normalization bytes differ from the native surface sampler: '+key)
    require(expected['source_center'].tolist() == reference['source_center']
            and float(expected['source_extent']) == reference['source_extent'],
            'source-defined normalization differs from the stored reference')
    expected_arrays = array_identities(dict(target_world=target_vertices, target_faces=source_faces,
        source_center=expected['source_center'], source_extent=expected['source_extent']))
    for key, value in expected_arrays.items():
        actual = native_geometry_arrays[key]
        require(all(actual[field] == value[field] for field in ['dtype','shape','values_C_SHA256']),
                'actual native geometry producer used different GT/world normalization: '+key)
    return dict(target_points_SHA256=hashlib.sha256(expected['target_points'].tobytes()).hexdigest(),
                native_GT_array_identities=expected_arrays)


class SurfaceReferences:
    """Cache one verified common reference per sample across every frozen model."""
    def __init__(self, spec, *, receipt, closed, ids, clusters):
        self.spec, self.receipt, self.closed = spec, receipt, closed
        audit_path = pinned(spec); closed.check(audit_path, receipt)
        audit = json.loads(audit_path.read_text())
        require(audit['status'] == 'PASS_ALL508_GT_SURFACE_REFERENCES'
                and audit['samples'] == 508 and audit['source_clusters'] == 134
                and audit['surface_samples'] == 20000 and audit['surface_seed'] == 101
                and audit['FScore_threshold'] == .02 and audit['source_defined_normalization']
                and audit['per_output_alignment'] is False and audit['target_recentered_or_rescaled'] is False
                and audit['geometry_frame'] == 'source-defined normalized WORLD'
                and audit['all508_source_target_topologies_and_final_vertices_exact']
                and audit['reference_points_readback_exact'], 'complete common native-world GT surface protocol required')
        records = audit['records']
        require(len(records) == 508 and len({r['sample_id'] for r in records}) == 508
                and {r['sample_id'] for r in records} == set(ids)
                and all(r['source_cluster_id'] == clusters[r['sample_id']] for r in records),
                'shared GT references must retain all frozen508 samples/source clusters')
        for path, digest in audit['source_SHA256'].items(): pinned(dict(path=path, SHA256=digest))
        self.root = audit_path.parent
        self.records = {row['sample_id']:row for row in records}
        self.cache = {}

    def expected(self, native):
        sid = native['row']['sample_id']; reference = self.records[sid]
        require(native['row']['source_cluster_id'] == reference['source_cluster_id']
                and native['row']['source_uid'] == reference['source_uid']
                and all(native['row'][key] == reference[key] for key in ['source_geometry','target_vertices'])
                and native['surface_seed'] == reference['surface_seed'] == 101,
                'native generated sample/geometry/surface noise differs from the shared GT')
        paths = {key:Path(native[key]) for key in ['source_geometry','target_vertices']}
        for key, path in paths.items():
            require(native['inputs_SHA256'][str(path)] == reference[key+'_SHA256'],
                    'native input geometry bytes differ from the shared GT reference')
            pinned(dict(path=str(path), SHA256=reference[key+'_SHA256']))
        reference_path = Path(reference['reference'])
        require(not reference_path.is_absolute() and '..' not in reference_path.parts,
                'explicit relative shared GT archive path required')
        path = self.root/reference_path
        require(self.closed.check(path, self.receipt) == reference['reference_SHA256'], 'shared GT archive bytes differ')
        if sid not in self.cache:
            with np.load(paths['source_geometry'], allow_pickle=False) as z:
                vertices, faces = z['V'].copy(), z['F'].copy()
            target = np.load(paths['target_vertices'], allow_pickle=False)
            with np.load(path, allow_pickle=False) as z:
                require(str(z['sample_id']) == sid and str(z['source_cluster_id']) == reference['source_cluster_id']
                        and str(z['schema']) == 'source-defined-GT-surface-reference-v1'
                        and int(z['point_count']) == 20000 and int(z['surface_seed']) == 101,
                        'actual shared GT archive identity/schema differs')
                arrays = {key:z[key].copy() for key in ['target_points','source_center','source_extent']}
            # The same native geometry is expected independently of prediction.
            identity = array_identities(dict(target_world=target,target_faces=faces,
                source_center=arrays['source_center'],source_extent=arrays['source_extent']))
            result = target_identity(reference,source_vertices=vertices,source_faces=faces,target_vertices=target,
                reference_arrays=arrays,native_geometry_arrays=identity)
            self.cache[sid] = result
        return reference, dict(path=str(path), SHA256=reference['reference_SHA256']), self.cache[sid]

    def join(self, native, generated, *, generation_root):
        require(generated['sample_id'] == native['row']['sample_id']
                and generated['GT_loaded_only_after_generation']
                and not generated['GT_geometry_or_coords_used_in_generation']
                and generated['source_defined_normalization'] and generated['per_output_alignment'] is False
                and generated['actual_resolution'] == 1024,
                'GT reference joins only completed native predictions, after generation')
        metrics = generated['metrics']
        require(metrics['prediction_samples'] == metrics['target_samples'] == 20000
                and metrics['threshold'] == .02 and metrics['per_output_alignment'] is False,
                'actual native20k F@.02 surface metric protocol differs')
        root = Path(generation_root)
        manifest_path = root/'remote-npz.json'
        remote = generated['remote_raw_archives']
        require(remote['manifest'] == str(manifest_path)
                and remote['manifest_SHA256'] == self.closed.check(manifest_path,root/'file4-closure.json'),
                'actual native geometry producer manifest binding differs')
        manifest = json.loads(manifest_path.read_text())
        require(manifest['schema'] == 'native-cascade-remote-npz-v1'
                and manifest['archives'] == remote['archives']
                and manifest['remote_root'] == remote['remote_root']
                and manifest['prefix'] == remote['prefix'], 'actual native producer archive identities differ')
        module = 'cascade_zero_shot_from_ss_artifact.py' if generated.get('original_zero_shot_full_stack',False) else 'cascade_from_ss_artifact.py'
        source = Path(__file__).with_name(module)
        require(manifest['source_contract']['reverse_AST_exact']
                and manifest['source_contract']['source_SHA256'] == sha(source),
                'GT join requires the unchanged native numerical producer source')
        reference, reference_file, expected = self.expected(native)
        actual = manifest['archives']['geometry-evaluation.npz']['arrays']
        for key, value in expected['native_GT_array_identities'].items():
            require(all(actual[key][field] == value[field] for field in ['dtype','shape','values_C_SHA256']),
                    'native generated geometry used different common GT arrays: '+key)
        return dict(status='PASS_ACTUAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN',
            sample_id=generated['sample_id'],source_cluster_id=reference['source_cluster_id'],
            GT_reference_audit=self.spec,GT_reference=reference_file,
            native_geometry_manifest=dict(path=str(manifest_path),SHA256=sha(manifest_path)),
            native_numeric_source=dict(path=str(source),SHA256=sha(source)),
            surface_metric_source=dict(path=str(Path(__file__).with_name('surface_metrics.py')),
                                       SHA256=sha(Path(__file__).with_name('surface_metrics.py'))),
            target_points_SHA256=expected['target_points_SHA256'],
            native_GT_array_identities=expected['native_GT_array_identities'],
            native_metrics=metrics,actual_existing_native_scores_unchanged=True,
            no_prediction_archive_download_or_neural_forward=True,
            GT_reference_not_used_for_model_conditioning=True)

