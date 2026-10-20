"""Diagnose the actual SS64-to-Shape512 support conversion, without sampling."""
import hashlib
import numpy as np

from .cascade_artifact_inputs import native_shape_support


def pooled_support(occupancy):
    value=np.asarray(occupancy)
    if value.shape!=(64,64,64) or value.dtype!=np.bool_:
        raise ValueError('actual binary SS64 occupancy required')
    result=np.zeros((32,32,32),dtype=bool)
    if value.any():
        coords=native_shape_support(value).cpu().numpy()
        if coords.dtype!=np.int32 or coords.ndim!=2 or coords.shape[1]!=4 or coords[:,0].any():
            raise ValueError('native single-sample Shape512 integer support changed')
        result[tuple(coords[:,1:].T)]=True
    else:
        # Preserve a real failed/empty SS prediction for analysis. Generation
        # separately requires a nonempty support; do not substitute GT points.
        coords=np.empty((0,4),dtype=np.int32)
    return result,coords


def pooled_support_metrics(prediction,target):
    pred,coords=pooled_support(prediction);truth,gt_coords=pooled_support(target)
    tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
    return dict(native_shape512_SS32_tokens=len(coords),decoded_GT_SS32_tokens=len(gt_coords),
        actual_native_SS32_coordinate_SHA256=hashlib.sha256(coords.tobytes()).hexdigest(),
        decoded_GT_SS32_coordinate_SHA256=hashlib.sha256(gt_coords.tobytes()).hexdigest(),
        tp=tp,fp=fp,fn=fn,voxel_iou=tp/max(1,tp+fp+fn),
        voxel_precision=tp/max(1,tp+fp),voxel_recall=tp/max(1,tp+fn),voxel_f1=2*tp/max(1,2*tp+fp+fn),
        nonempty_generation_input=bool(len(coords)),native_maxpool64_to32_used=True,
        GT_is_decoded_latent_reference_not_raw_mesh=True,GT_support_is_for_scoring_only=True)


def artifact_support_metrics(path):
    with np.load(path) as archive:
        if archive['occupancy_shape'].tolist()!=[64,64,64]:
            raise ValueError('actual SS64 archive shape required')
        values=[]
        for key in ['prediction_occupancy_packed','target_occupancy_packed']:
            packed=archive[key]
            if packed.dtype!=np.uint8 or packed.shape!=(64**3//8,):
                raise ValueError('complete actual packed SS64 occupancy required')
            values.append(np.unpackbits(packed).reshape(64,64,64).astype(bool))
    return pooled_support_metrics(*values)
