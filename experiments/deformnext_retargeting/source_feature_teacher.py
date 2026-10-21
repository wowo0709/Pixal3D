"""Train-only native source-feature supervision; never an inference input."""
import torch

from pixal3d.modules.native_prior_source_ray import source_ray_delta


def source_feature_teacher_loss(output, original_source, teacher, *, split, camera_ids):
    """Equal-weight MSE over both complete source9/10 patch maps.

    No foreground, visibility or matching-success mask changes the objective.
    Camera11 has no teacher and receives no fabricated identity label. The
    original module still decides where it can emit a correction.
    """
    batch = original_source.shape[0]
    if (split != 'train' or tuple(camera_ids) != (9, 10)
            or teacher.shape != (batch, 2, 1024, 1024)
            or not teacher.is_floating_point() or teacher.requires_grad
            or teacher.device != original_source.device
            or not torch.isfinite(teacher).all()):
        raise ValueError('detached finite TRAIN M1 RGB teachers for cameras9/10 required')
    delta = source_ray_delta(output.source_numerator, output.semantic_weight, original_source)
    prediction = original_source[:, :2].float() + delta[:, :2]
    return (prediction - teacher.float()).square().mean()


def validate_teacher_records(rows, forbidden_source_clusters):
    """Reject mixed splits, repeated sources and accidental validation replays."""
    if not rows or not forbidden_source_clusters:
        raise ValueError('explicit training records and held-out source identities required')
    samples, clusters = set(), set()
    for row in rows:
        sid, cluster = row['sample_id'], row['source_cluster_id']
        if (row['split'] != 'train' or row.get('not_for_training', False)
                or sid in samples or cluster in clusters
                or cluster in forbidden_source_clusters
                or row['conditioning_camera_ids'] != [8, 9, 10, 11]
                or len(row['evaluation_target_images']) != 2):
            raise ValueError('unique TRAIN source clusters disjoint from held-out data required')
        samples.add(sid); clusters.add(cluster)
    return samples, clusters
