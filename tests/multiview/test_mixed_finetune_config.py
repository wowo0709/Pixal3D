import importlib.util
from pathlib import Path


def test_new_recipe_preserves_objective_and_derives_update_budget():
    path = Path("experiments/multiview_finetune/configure.py")
    assert path.exists(), "mixed-source config builder is not implemented"
    spec = importlib.util.spec_from_file_location("mixed_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for stage, ebs, lr in [("ss64", 96, 5e-5), ("shape512", 96, 5e-5),
                           ("shape1024", 24, 1e-5), ("pbr1024", 24, 1e-5)]:
        config = module.stage_config(stage, 21500, microbatch=1)
        trainer = config["trainer"]["args"]
        assert trainer["max_steps"] == (21500 // ebs) * 5
        # Native run() only saves on i_save boundaries, including the final update.
        assert trainer["max_steps"] % trainer["i_save"] == 0
        assert trainer["batch_size_per_gpu"] == ebs
        assert trainer["batch_split"] == ebs
        assert trainer["optimizer"]["args"]["lr"] == lr
        assert trainer["ema_rate"] == [0.9999]
        assert trainer["mix_precision_dtype"] == "bfloat16"
        assert trainer["lr_scheduler"]["name"] == "LinearWarmupLRScheduler"
        assert config["dataset"]["args"]["min_condition_views"] == 1
        assert config["dataset"]["args"]["max_condition_views"] == 6
        assert config["validation"]["dataset_name"] == config["dataset"]["name"]
        assert config["validation"]["assets_per_source"] is None
        assert config["validation"]["roots_file"].endswith(f"validation_{stage}_roots.json")
