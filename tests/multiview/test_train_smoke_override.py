from easydict import EasyDict as edict

from train import apply_smoke_overrides


def test_ten_step_smoke_logs_every_step_samples_five_and_ten_and_saves_ten():
    config = edict({"trainer": {"args": {
        "max_steps": 1_000_000,
        "i_log": 5,
        "i_sample": 250,
        "i_save": 1000,
        "snapshot_batch_size": 8,
        "snapshot_num_samples": 64,
        "num_workers": 14,
        "prefetch_data": True,
    }}})
    apply_smoke_overrides(config, 10)
    assert config.trainer.args.max_steps == 10
    assert config.trainer.args.i_log == 1
    assert config.trainer.args.i_sample == 5
    assert config.trainer.args.i_save == 10
    assert config.trainer.args.snapshot_batch_size == 1
    assert config.trainer.args.snapshot_num_samples == 1
    assert config.trainer.args.num_workers == 0
    assert config.trainer.args.prefetch_data is False


def test_one_step_online_gate_samples_and_saves_step_one():
    config = edict({"trainer": {"args": {
        "max_steps": 1_000_000,
        "i_log": 5,
        "i_sample": 250,
        "i_save": 1000,
        "snapshot_batch_size": 8,
        "snapshot_num_samples": 64,
        "num_workers": 14,
        "prefetch_data": True,
    }}})
    apply_smoke_overrides(config, 1)
    assert config.trainer.args.max_steps == 1
    assert config.trainer.args.i_log == 1
    assert config.trainer.args.i_sample == 1
    assert config.trainer.args.i_save == 1
    assert config.trainer.args.snapshot_batch_size == 1
    assert config.trainer.args.snapshot_num_samples == 1
    assert config.trainer.args.num_workers == 0
    assert config.trainer.args.prefetch_data is False
