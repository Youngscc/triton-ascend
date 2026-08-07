# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import hashlib
import json

from triton.backends.ascend.runtime.adapter_overflow_screen import (
    STRESS_CONFIGS,
    _write_artifacts,
    infer_kernel_type,
)


def test_stress_configs_are_unique_and_high_pressure():
    payloads = [json.dumps(config.values, sort_keys=True) for config in STRESS_CONFIGS]
    assert len(STRESS_CONFIGS) == 8
    assert len(set(payloads)) == len(payloads)
    assert any(config.values["num_stages"] == 2 for config in STRESS_CONFIGS)
    assert any(config.values["tile_mix_vector_loop"] == 8 for config in STRESS_CONFIGS)
    assert any(config.values["tile_mix_cube_loop"] == 8 for config in STRESS_CONFIGS)
    assert all(config.values["enable_ubuf_saving"] is False for config in STRESS_CONFIGS)


def test_infer_kernel_type_from_adapter_metadata(tmp_path):
    vector = tmp_path / "vector.ttadapter"
    vector.write_text('func.func @f() attributes {mix_mode = "aiv"}\n', encoding="utf-8")
    cube = tmp_path / "cube.ttadapter"
    cube.write_text('func.func @f() attributes {mix_mode = "aic"}\n', encoding="utf-8")
    mix = tmp_path / "mix.ttadapter"
    mix.write_text('func.func @f() attributes {mix_mode = "mix"}\n', encoding="utf-8")
    assert infer_kernel_type(vector) == "vector"
    assert infer_kernel_type(cube) == "cube"
    assert infer_kernel_type(mix) == "mixcv"


def test_artifacts_only_include_planmemory_confirmed_cases(tmp_path):
    config_ids = [
        hashlib.sha256(json.dumps(config.values, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        for config in STRESS_CONFIGS
    ]

    def row(mode, status, adapter="positive.ttadapter", config=config_ids[0]):
        return {
            "adapter_path": str(tmp_path / adapter),
            "adapter_digest": adapter,
            "config_id": config,
            "kernel_type": "mixcv",
            "status": status,
            "normalized_config": STRESS_CONFIGS[0].values,
            "required_bits": 64,
            "capacity_bits": 32,
            "mode": mode,
        }

    screen = [
        row("prune", "predicted_ub_overflow_final"),
        row("prune", "predicted_ub_overflow_final", "mismatch.ttadapter", config_ids[1]),
    ]
    verify = [row("baseline", "native_ub_overflow_final")]
    manifest, cases, by_adapter, mismatches = _write_artifacts(
        tmp_path,
        tmp_path,
        (),
        screen,
        verify,
    )
    assert set(by_adapter) == {"positive.ttadapter"}
    assert mismatches == 1
    assert len(json.loads(cases.read_text(encoding="utf-8"))) == 1
    assert json.loads(
        manifest.read_text(encoding="utf-8"))["adapters"] == [{"kernel_type": "mixcv", "path": "positive.ttadapter"}]
