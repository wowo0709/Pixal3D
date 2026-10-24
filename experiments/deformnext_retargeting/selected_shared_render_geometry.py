"""Reuse actual owner render triangles without creating another raw archive."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .fixed_pbr_render_geometry import render_only_prediction, verify_render_only_prediction
from .selected_pipeline_model_contract import require


SCHEMA = 'selected-canonical-owner-render-geometry-v1'
FIELD = 'frozen_generated_render_geometry'


def required_sources():
    root = Path(__file__).resolve().parent
    return [str(root/name) for name in ['selected_shared_render_geometry.py',
                                      'fixed_pbr_render_geometry.py']]


def _spec(path, check):
    path = Path(path)
    receipt = path.parent/'file4-closure.json'
    return dict(path=str(path), SHA256=check.check(path, receipt), receipt=str(receipt))


def bind_owner_render_geometry(case, owner, check, source_SHA256):
    """Keep phase count, raw roots, sampled outputs and non-reuse cases intact."""
    if case['geometry_reference'] is None:
        require(owner is None, 'a geometry producer cannot borrow a render mesh')
        return case
    ref = case['geometry_reference']
    require(owner is not None and owner.get('actual_completed_native_case') is True
            and owner['alias'] == ref['alias'] and owner['sample_id'] == case['sample_id']
            and owner['geometry_group'] == case['geometry_group']
            and owner['geometry_source_root'] == case['geometry_source_root'],
            'completed canonical owner of this sample and exact SS group required')
    require(set(required_sources()) <= set(source_SHA256),
            'freeze the actual shared render-input sources before TEST')
    result = deepcopy(case)
    phases = [p for p in result['phases'] if p['name'] == 'render']
    require(len(phases) == 1 and FIELD not in phases[0]['plan'],
            'one original render phase without a previous override required')
    record = dict(schema=SCHEMA, sample_id=case['sample_id'], owner_alias=owner['alias'],
        geometry_group=case['geometry_group'],
        common_prediction=_spec(Path(owner['geometry_source_root'])/'prediction.npz', check),
        render_mesh=_spec(Path(owner['case_output'])/'render/render-geometry.npz', check),
        source_SHA256={p:source_SHA256[p] for p in required_sources()})
    phases[0]['plan'][FIELD] = record
    result['render_geometry_reference'] = record
    return result


def _checked(spec, check):
    require(set(spec) == {'path', 'SHA256', 'receipt'}, 'explicit closed render input required')
    path = Path(spec['path'])
    require(check.check(path, spec['receipt']) == spec['SHA256'],
            'actual canonical render input changed')
    return path


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k:archive[k] for k in archive.files}


class _Archive:
    def __init__(self, arrays):self.arrays = arrays
    def __enter__(self):return self.arrays
    def __exit__(self, *args):return False


class FrozenGeometryNumpy:
    """A renderer-local NumPy proxy; global NumPy and original files stay intact."""
    def __init__(self, original_numpy, native):
        self.original = original_numpy
        self.native = native
        self.reference = native[FIELD]
        require(self.reference.get('schema') == SCHEMA
                and self.reference['sample_id'] == native['row']['sample_id'],
                'same explicit sample and frozen generated render input required')
        require(set(self.reference['source_SHA256']) == set(required_sources())
                and all(sha(p) == h for p,h in self.reference['source_SHA256'].items()),
                'actual frozen render-input adapter sources required')
        self.check = ClosedInputs(native['canonical_local_root'])
        self.prediction = Path(native['generation_root'])/'prediction.npz'
        self.prediction_reads = 0
        self.derived = None

    def __getattr__(self, name):return getattr(self.original, name)

    def load(self, path, *args, **kwargs):
        if Path(path).resolve() != self.prediction.resolve():
            return self.original.load(path, *args, **kwargs)
        require(not args and not kwargs and self.prediction_reads == 0,
                'one explicit original renderer prediction read required')
        self.prediction_reads += 1
        self.model_spec = _spec(self.prediction, self.check)
        generated_path = self.prediction.parent/'generation.json'
        _spec(generated_path, self.check)
        generated = json.loads(generated_path.read_text())
        require(generated['sample_id'] == self.reference['sample_id']
                and generated['GT_geometry_or_coords_used_in_generation'] is False,
                'actual generated own prediction, without GT input, required')
        common = _checked(self.reference['common_prediction'], self.check)
        frozen = _checked(self.reference['render_mesh'], self.check)
        self.derived = render_only_prediction(_arrays(self.prediction), _arrays(common), _arrays(frozen))
        return _Archive(self.derived)

    def verify(self):
        require(self.prediction_reads == 1 and self.derived is not None,
                'the actual original renderer must consume the fixed geometry input')
        model = _checked(self.model_spec, self.check)
        frozen = _checked(self.reference['render_mesh'], self.check)
        rendered = Path(self.native['output'])/'render-geometry.npz'
        proof = verify_render_only_prediction(self.derived, _arrays(model), _arrays(frozen), _arrays(rendered))
        proof.update(reference=deepcopy(self.reference), input_override='renderer-local NumPy archive read',
                     original_prediction_SHA256=self.model_spec['SHA256'],
                     new_raw_archive_files=0, prediction_reads=self.prediction_reads)
        return proof


def verify_case_render_geometry(case, check):
    """Check the actual GPU phase proof and mesh before closing a reused case."""
    reference = case.get('render_geometry_reference')
    if reference is None:
        require(case['geometry_reference'] is None, 'reused geometry requires its frozen render mesh')
        return None
    root = Path(next(p['plan']['output'] for p in case['phases'] if p['name'] == 'render'))
    path = root/'selected-phase-entry.json'
    check.check(path, root/'file4-closure.json')
    phase = json.loads(path.read_text())
    proof = phase.get('proof', {}).get(FIELD, {})
    require(phase.get('status') == 'PASS_ACTUAL_ORIGINAL_NUMERICAL_SELECTED_CASE_PHASE'
            and phase.get('kind') == 'assigned-native-render'
            and proof.get('reference') == reference and proof.get('new_raw_archive_files') == 0
            and proof.get('status') == 'PASS_EXACT_FROZEN_GENERATED_PBR_RENDER_GEOMETRY',
            'actual source-bound in-memory fixed render phase proof required')
    mesh = root/'render-geometry.npz';check.check(mesh, root/'file4-closure.json')
    expected = _arrays(_checked(reference['render_mesh'], check)); actual = _arrays(mesh)
    require(set(actual) == set(expected) == {'vertices','faces'}
            and all(actual[k].dtype == expected[k].dtype and np.array_equal(actual[k], expected[k])
                    for k in actual), 'actual reused rendering triangles differ from the canonical owner')
    return proof
