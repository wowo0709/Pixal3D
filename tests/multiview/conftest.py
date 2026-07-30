import os

import pytest
import torch


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_collection(session):
    original_get_device_name = torch.cuda.get_device_name
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        torch.cuda.get_device_name = lambda _device=None: "A100"
    try:
        yield
    finally:
        torch.cuda.get_device_name = original_get_device_name
