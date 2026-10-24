"""Camera-role statistics on one paired validation128 generation realization."""
from .cascade_view_roles import summarize_view_roles
from .dev128_pipeline_contract import ARMS
from .selected_pipeline_model_contract import require


def summarize(records, *, sample_ids, clusters, draws=4000):
    require(set(records) == set(ARMS) and len(sample_ids) == len(set(sample_ids)) == 128
        and set(clusters) == set(sample_ids) and len(set(clusters.values())) == 93,
        'the same three previously measured pipelines and full128/93-source panel required')
    for arm, rows in records.items():
        require([r['sample_id'] for r in rows] == sample_ids
            and all(r['source_cluster_id'] == clusters[r['sample_id']] for r in rows),
            'complete ordered paired population required for '+arm)
    result = summarize_view_roles(records, [
        ('E_FIXED_RAW', 'C_DECODER'), ('E_LEARNED_RAW', 'C_DECODER'),
        ('E_LEARNED_RAW', 'E_FIXED_RAW'),
    ], draws=draws, seed=20261024)
    result.update(status='PASS_ACTUAL_PAIRED_THREE_PIPELINE_DEV128_EIGHT_METRICS',
        independent_training_seeds=1, generation_noises=1,
        exploratory_extension_after_DEV32_screening=True,
        original_finite12_and_SS128x3_studies_unchanged=True,
        statistics_scope='source-cluster paired exploratory intervals; not held-out confirmation',
        source_preservation=['LPIPS', 'DreamSim', 'CLIP'],
        target_alignment=['PSNR', 'SSIM', 'M_IoU', 'CD', 'FScore'],
        final_FID_KID_TEST508_and_independent_confirmation_still_required=True,
        candidate_selected=False, full_goal_complete=False)
    return result
