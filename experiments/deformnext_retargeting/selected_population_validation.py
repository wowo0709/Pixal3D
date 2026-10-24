"""Verify two actual validation36 loads without repeating baseline NN work.

The three original native families may retain their old binding schema. Their
physical config, source count, saved bytes and predictions must match the new
binding exactly before its metadata can be used. Other families require their
own current schema. No architecture or measured array is relabelled.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .collect_module_seed_confirmation import read_file, verify_terminal
from .evaluate_ss_artifact_subset import validate_subset_reference
from .native_population_contract import (
    ENGINEERING, model_binding as legacy_binding, validate_repeatability,
)
from .pipeline_population_contract_v2 import model_binding, required_sources as population_sources
from .selected_pipeline_model_contract import require, validate_ss_evaluation


SCHEMA = 'selected-physical-validation36-two-load-closed-inputs-v1'
LEGACY_ARMS = {'V1', 'C_DECODER', 'FULL'}
# These later metadata-only entries were not used by the original three-family
# evaluator. Every original numerical/runtime source is still mandatory.
LEGACY_METADATA_ADDITIONS = {
    'cascade_view_roles.py','evaluate_pipeline_bound_population_v2.py','evaluate_unified_bound_population.py',
    'module_pipeline_protocol.py','module_pipeline_protocol_v2.py','pipeline_population_contract_v2.py',
    'unified_module_summary.py','unified_population_contract.py',
}


def exact(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def validate_records(results, audits, reference, *, arm, bundle, config, hashes):
    """Keep actual numerical records and compare only legitimate schema lineage."""
    own = model_binding(bundle, config, hashes, arm=arm)
    validate_ss_evaluation(reference, own, arm)
    require(reference.get('evaluation_mode') == 'final128' and len(reference['per_asset']) == 128,
            'actual own final128 reference required for original prefix32')
    require(len(results) == len(audits) == 2, 'two independent complete validation36 records required')
    normalized_results, normalized_audits, lineage = [], [], []
    for result, audit in zip(results, audits):
        validate_ss_evaluation(result, own, arm)
        require(result.get('population_scope') == ENGINEERING and result.get('diagnostic_only') is True
            and result.get('native_task_checkpoint_bundle') == reference.get('native_task_checkpoint_bundle')
            and result.get('manifest_sha256') == reference.get('manifest_sha256')
            and result.get('native_SS_artifacts', {}).get('status') == 'PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION_ARCHIVE',
            'own actual bound validation36 archive, config, manifest and bundle required')
        old = result.get('frozen_native_model_binding')
        require(exact(audit.get('model_binding'), old), 'actual evaluator and audit disagree about loaded model')
        if not exact(old, own):
            require(arm in LEGACY_ARMS, 'a non-native physical architecture cannot use legacy binding metadata')
            expected = legacy_binding(bundle, config, hashes)
            require(exact(old, expected), 'legacy metadata refers to different physical model bytes or variant')
        artifacts = result['native_SS_artifacts']['artifacts']
        actual = dict(result, per_asset=result['per_asset'][:32])
        validate_subset_reference(actual, reference, artifacts[:32], count=32)
        require(result['quick_panel_ids'] == reference['quick_panel_ids'][:36]
            and [r['seed'] for r in result['per_asset']] == [r['seed'] for r in reference['per_asset'][:36]],
            'same prescribed36 IDs and noise required including the tail batch')
        r, a = deepcopy(result), deepcopy(audit)
        r['frozen_native_model_binding'] = deepcopy(own); a['model_binding'] = deepcopy(own)
        normalized_results.append(r); normalized_audits.append(a)
        lineage.append(dict(original_binding=old, physical_binding=own,
            metadata_schema_only=not exact(old, own), actual_measurements_or_architecture_changed=False))
    proof = validate_repeatability(normalized_results, normalized_audits)
    proof.update(module_arm=arm, original_evaluator_binding_lineage=lineage,
        actual_reference32_checked_from_records=True, tail36_noise_checked_from_reference=True,
        no_new_neural_forward_or_training=True, no_architecture_alias_or_measurement_change=True)
    return proof


def verify_archive(path, artifact, check, receipt):
    """Hash actual latent and occupancy data; never infer them from JSON flags."""
    path = Path(path)
    require(check.check(path, receipt) == artifact['artifact_SHA256'], 'actual SS archive file bytes differ')
    with np.load(path, allow_pickle=False) as arrays:
        require(str(arrays['sample_id']) == artifact['sample_id']
            and int(arrays['seed']) == artifact['seed']
            and arrays['occupancy_shape'].tolist() == [64,64,64], 'actual saved SS sample/noise/grid differs')
        for key in ['prediction_latent', 'target_latent']:
            value = arrays[key]
            digest = hashlib.sha256((str(value.dtype)+repr(value.shape)).encode())
            digest.update(value.tobytes())
            require(value.dtype == np.float32 and digest.hexdigest() == artifact[key+'_SHA256'],
                    'actual saved native latent array differs: '+key)
        decoded = []
        for prefix in ['prediction','target']:
            value = arrays[prefix+'_occupancy_packed']
            require(value.dtype == np.uint8 and value.shape == (64**3//8,)
                and hashlib.sha256(value.tobytes()).hexdigest() == artifact[prefix+'_voxels_SHA256'],
                'actual saved decoded occupancy differs: '+prefix)
            decoded.append(np.unpackbits(value, count=64**3).astype(bool))
        pred, gt = decoded
        counts = dict(tp=int((pred & gt).sum()), fp=int((pred & ~gt).sum()), fn=int((~pred & gt).sum()))
        require(counts == artifact['occupancy_counts'], 'actual occupancy intersection/union counts differ')
    return dict(actual_archived_latent_and_occupancy_bytes_checked=True, sample_id=artifact['sample_id'])


def required_sources(arm):
    root = Path(__file__).resolve().parent
    return sorted(set(population_sources(arm)) | {str(root/(name+'.py')) for name in
        ['selected_population_validation','selected_pipeline_model_contract','collect_module_seed_confirmation']})


def execute(path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit CPU-only validation reduction required')
    path = Path(path); plan = json.loads(path.read_text())
    require(plan.get('schema') == SCHEMA, 'actual closed two-load validation plan required')
    check = ClosedInputs(plan['canonical_local_root']); check.check(path, path.parent/'file4-closure.json')
    arm = plan['module_arm']
    require(set(required_sources(arm)) <= set(plan['source_SHA256']), 'pin all original numerical/metadata validators')
    for source, digest in plan['source_SHA256'].items(): require(sha(source) == digest, 'validation source changed')
    bundle = read_file(plan['bundle'], check); hashes = {}
    for key in ['native_config','checkpoint','conditioner_checkpoint']:
        if bundle[key] is not None: hashes[bundle[key]] = check.check(bundle[key], plan['bundle']['receipt'])
    config = json.loads(Path(bundle['native_config']).read_text())
    physical_binding = model_binding(bundle, config, hashes, arm=arm)
    reference = read_file(plan['reference'], check)
    require(reference.get('native_task_checkpoint_bundle') == plan['bundle']['path'], 'actual own bundle required')
    require(len(plan['repetitions']) == 2, 'exactly two actual native loading processes required')
    results, audits, terminals, folders = [], [], [], []; evaluated_sources = {}
    for item in plan['repetitions']:
        result = read_file(item['evaluation'], check); audit = read_file(item['audit'], check)
        terminal = read_file(item['terminal'], check); terminals.append(verify_terminal(terminal))
        folder = Path(item['evaluation']['path']).parent
        require(Path(item['audit']['path']).parent == folder, 'own actual evaluator audit directory required')
        native_plan = read_file(item['population_plan'], check)
        require(native_plan['population_scope'] == ENGINEERING and native_plan['expected_assets'] == 36
            and native_plan['generation_batch_size'] == 8 and native_plan['deterministic_algorithms'] is True
            and native_plan['mode'] == 'bound-population' and native_plan['bundle'] == plan['bundle']['path']
            and native_plan['output'] == str(folder)
            and native_plan['wrapper_output'] == str(Path(item['terminal']['path']).parent)
            and exact(native_plan['model_binding'], result['frozen_native_model_binding'])
            and result.get('verified_population_inputs', {}).get(item['population_plan']['path'])
                == item['population_plan']['SHA256'],
            'actual loaded population plan, output, wrapper and evaluated source provenance required')
        needed = set(population_sources(arm))
        if not exact(native_plan['model_binding'], physical_binding):
            require(arm in LEGACY_ARMS, 'only the actual original three numerical families permit older metadata')
            needed = {p for p in needed if Path(p).name not in LEGACY_METADATA_ADDITIONS}
        require(needed <= set(native_plan['source_SHA256']), 'pin every originally evaluated physical numerical source')
        for source, digest in native_plan['source_SHA256'].items():
            require(sha(source) == digest, 'original actually evaluated population source changed')
            require(source not in evaluated_sources or evaluated_sources[source] == digest,
                    'the two loads used conflicting numerical source versions')
            evaluated_sources[source] = digest
        require(result['frozen_native_model_binding']['checkpoint']['SHA256'] == hashes[bundle['checkpoint']],
                'actual loading used another saved denoiser')
        for artifact in result['native_SS_artifacts']['artifacts']:
            stored = folder/artifact['artifact']
            require(stored.resolve().is_relative_to(folder.resolve()), 'own archive path required')
            verify_archive(stored, artifact, check, item['evaluation']['receipt'])
        results.append(result); audits.append(audit); folders.append(str(folder))
    require(len(set(folders)) == len({(t['PID'],t['birth']) for t in terminals}) == 2,
            'two different actual saved outputs and model-loading process identities required')
    proof = validate_records(results, audits, reference, arm=arm, bundle=bundle, config=config, hashes=hashes)
    proof.update(actual_saved_archive_files_verified=72, actual_saved_latent_and_occupancy_array_bytes_verified=True,
        actual_repetitions=folders,
        actual_terminals=terminals, verified_closed_inputs=check.inputs,
        source_SHA256=plan['source_SHA256'], full_goal_complete=False)
    for source, digest in {**evaluated_sources, **plan['source_SHA256'], **check.inputs}.items():
        require(sha(source) == digest, 'actual evidence changed during validation reduction')
    out = Path(plan['output']); out.mkdir(parents=True, exist_ok=False)
    (out/'audit.json').write_text(json.dumps(proof, indent=2, allow_nan=False)+'\n')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); value = execute(args.plan)
    print(json.dumps(dict(status=value['status'], module_arm=value['module_arm'], new_neural_forwards=0)), flush=True)


if __name__ == '__main__': main()
