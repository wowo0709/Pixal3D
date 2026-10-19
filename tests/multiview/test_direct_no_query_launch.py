import json
import multiprocessing
import sys

import pytest


def _check_rank_registration(queue,rank):
    from experiments.deformnext_retargeting import train_direct_no_query_joint as entry
    from experiments.deformnext_retargeting import ray_feature_joint_runtime as runtime
    from experiments.deformnext_retargeting.direct_no_query_joint_runtime import DirectNoQueryJointTrainer
    from pixal3d.modules.direct_ray_retargeting_no_query import DirectRayRetargetingNoQueryConditioner
    old_trainer,old_module = runtime.RayFeatureJointTrainer,runtime.RayFeatureAttentionConditioner

    def observed(local_rank,config):
        assert local_rank==rank
        assert runtime.RayFeatureJointTrainer is DirectNoQueryJointTrainer
        assert runtime.RayFeatureAttentionConditioner is DirectRayRetargetingNoQueryConditioner

    entry.native_worker = observed
    entry.worker(rank,{})
    assert runtime.RayFeatureJointTrainer is old_trainer
    assert runtime.RayFeatureAttentionConditioner is old_module
    queue.put(rank)


def test_spawned_ranks_install_and_restore_actual_NoDQ_factories():
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    processes = [context.Process(target=_check_rank_registration,args=(queue,rank)) for rank in [0,1]]
    for process in processes:
        process.start()
    try:
        assert {queue.get(timeout=90),queue.get(timeout=90)}=={0,1}
        for process in processes:
            # Native sparse/render imports have a substantial interpreter
            # teardown on the bounded two-core test allocation. Registration
            # has already been observed from both actual spawned children.
            process.join(timeout=60)
            assert process.exitcode==0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        queue.close()


def test_parent_dispatches_spawnable_new_worker_and_restores_original(monkeypatch,tmp_path):
    from experiments.deformnext_retargeting import train_direct_no_query_joint as entry
    from experiments.deformnext_retargeting import train_ray_feature_joint as native
    old_worker = native.worker
    config = tmp_path/'recipe.json'
    config.write_text(json.dumps(dict(dataset=dict(name='DeformNeXtSS64',args={}),
        validation=dict(dataset_name='DeformNeXtSS64'),
        trainer=dict(args=dict(anchor_stage=dict(variant='direct-physical-NoDQ'))))))
    monkeypatch.setattr(sys,'argv',['entry','--config',str(config)])

    class Observed(Exception):
        pass

    def main():
        assert native.worker is entry.worker
        raise Observed

    monkeypatch.setattr(native,'main',main)
    with pytest.raises(Observed):
        entry.main()
    assert native.worker is old_worker
