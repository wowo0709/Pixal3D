from data_toolkit.pipeline.camera import build_condition_views


def test_views_are_deterministic_and_bounded(config):
    first = build_condition_views("d" * 64, config.render)

    assert first == build_condition_views("d" * 64, config.render)
    assert len(first) == 8
    assert all(10 <= item["fov_degrees"] <= 70 for item in first)
    assert first != build_condition_views("e" * 64, config.render)
