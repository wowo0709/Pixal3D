import importlib
import os

import pytest
import torch


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        return
    original_get_device_name = torch.cuda.get_device_name
    torch.cuda.get_device_name = lambda _device=None: "A100"
    try:
        importlib.import_module("flex_gemm.ops.grid_sample")
    finally:
        torch.cuda.get_device_name = original_get_device_name
