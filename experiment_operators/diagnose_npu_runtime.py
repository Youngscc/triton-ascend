#!/usr/bin/env python3
"""Print the exact runtime stage reached before an NPU hang."""

from __future__ import annotations

import importlib
import os


def stage(name: str, value: object = "") -> None:
    suffix = f" {value}" if value != "" else ""
    print(f"RUNTIME_STAGE {name}{suffix}", flush=True)


stage("import_torch_begin")
torch = importlib.import_module("torch")
importlib.import_module("torch_npu")

stage("import_torch_done")
stage("ASCEND_VISIBLE_DEVICES", os.environ.get("ASCEND_VISIBLE_DEVICES", "<unset>"))
stage(
    "ASCEND_RT_VISIBLE_DEVICES",
    os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>"),
)
stage("device_count", torch.npu.device_count())
stage("set_device_begin")
torch.npu.set_device(0)
stage("set_device_done", torch.npu.get_device_name(0))
stage("allocate_begin")
x = torch.ones(16, dtype=torch.float32, device="npu:0")
stage("allocate_done")
stage("torch_add_begin")
y = x + x
stage("torch_add_done")
stage("synchronize_begin")
torch.npu.synchronize()
stage("synchronize_done")
stage("copy_to_cpu_begin")
stage("result", y.cpu().tolist())
print("NPU_RUNTIME_OK", flush=True)
