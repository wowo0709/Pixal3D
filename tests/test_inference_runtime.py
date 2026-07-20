import importlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


class FakeImageConditioner:
    use_naf_upsample = False

    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(torch.device(device))
        return self


class FakePipeline:
    loaded_paths = []

    def __init__(self):
        self.devices = []

    @classmethod
    def from_pretrained(cls, path):
        cls.loaded_paths.append(path)
        return cls()

    def to(self, device):
        self.devices.append(torch.device(device))


def _import_inference_without_model_weights(monkeypatch):
    import pixal3d.pipelines as pipelines

    monkeypatch.setitem(
        pipelines.__dict__, "Pixal3DImageTo3DPipeline", FakePipeline
    )
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "o_voxel", SimpleNamespace())
    sys.modules.pop("inference", None)
    return importlib.import_module("inference")


def test_production_runtime_bootstraps_low_vram_without_model_weights(monkeypatch):
    inference = _import_inference_without_model_weights(monkeypatch)
    monkeypatch.setattr(
        inference, "build_image_cond_model", lambda config: FakeImageConditioner()
    )

    init_signature = inspect.signature(inference.init_pipeline)
    export_signature = inspect.signature(inference.export_glb)
    assert list(init_signature.parameters) == ["model_path", "device", "low_vram"]
    assert list(export_signature.parameters) == [
        "pipeline",
        "mesh",
        "resolution",
        "output_path",
        "decimation_target",
        "texture_size",
    ]

    pipeline = inference.init_pipeline(
        "missing-model-weights", device="cpu", low_vram=True
    )

    assert FakePipeline.loaded_paths[-1] == "missing-model-weights"
    assert pipeline.low_vram is True
    assert pipeline.devices == [torch.device("cpu")]
    for attribute in inference.IMAGE_COND_CONFIGS:
        model = getattr(pipeline, f"image_cond_model_{attribute}")
        assert model.devices == []


def test_full_vram_bootstrap_places_image_conditioners_on_requested_device(
    monkeypatch,
):
    inference = _import_inference_without_model_weights(monkeypatch)
    monkeypatch.setattr(
        inference, "build_image_cond_model", lambda config: FakeImageConditioner()
    )

    pipeline = inference.init_pipeline(
        "missing-model-weights", device="cpu", low_vram=False
    )

    assert pipeline.low_vram is False
    assert pipeline.devices == [torch.device("cpu")]
    for attribute in inference.IMAGE_COND_CONFIGS:
        model = getattr(pipeline, f"image_cond_model_{attribute}")
        assert model.devices == [torch.device("cpu")]


def test_export_glb_uses_tracked_orientation_and_export_settings(
    tmp_path, monkeypatch
):
    inference = _import_inference_without_model_weights(monkeypatch)

    class FakeGlb:
        def __init__(self):
            self.transform = None
            self.export_call = None

        def apply_transform(self, transform):
            self.transform = transform

        def export(self, path, *, extension_webp):
            self.export_call = (Path(path), extension_webp)

    glb = FakeGlb()
    conversion_calls = []

    def to_glb(**kwargs):
        conversion_calls.append(kwargs)
        return glb

    monkeypatch.setattr(
        inference,
        "o_voxel",
        SimpleNamespace(postprocess=SimpleNamespace(to_glb=to_glb)),
    )
    pipeline = SimpleNamespace(pbr_attr_layout={"base_color": slice(0, 3)})
    mesh = SimpleNamespace(
        vertices=object(), faces=object(), attrs=object(), coords=object()
    )
    output_path = tmp_path / "nested" / "result.glb"

    inference.export_glb(
        pipeline,
        mesh,
        1024,
        output_path,
        decimation_target=123_456,
        texture_size=1024,
    )

    assert conversion_calls[0]["grid_size"] == 1024
    assert conversion_calls[0]["decimation_target"] == 123_456
    assert conversion_calls[0]["texture_size"] == 1024
    np.testing.assert_array_equal(
        glb.transform,
        np.array(
            [
                [-1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        ),
    )
    assert output_path.parent.is_dir()
    assert glb.export_call == (output_path, True)
