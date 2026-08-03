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

import json
from pathlib import Path

from triton.backends.ascend.runtime.adapter_compile_only import (
    AdapterCompileExecutor,
    AdapterSpec,
    CompileOnlyOptions,
    CompileTask,
    KERNEL_TYPE_PARAMS,
    _experiment_metrics,
    _load_completed,
    _status,
    build_all_configs,
    config_to_bisheng_options,
    mode_to_bisheng_options,
    normalize_config,
    parse_fallback_actions,
    parse_model_results,
    parse_native_ub_result,
    run_adapter_compile_only,
)
from triton.backends.ascend.runtime.autotuner import (
    AutoTilingTuner,
    _CUBE_PARAMS,
    _MIXCV_PARAMS,
    _VECTOR_PARAMS,
)


def _marker():
    pass


def _make_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_all_search_space_counts_and_uniqueness():
    assert set(KERNEL_TYPE_PARAMS["vector"]) == _VECTOR_PARAMS
    assert set(KERNEL_TYPE_PARAMS["cube"]) == _CUBE_PARAMS
    assert set(KERNEL_TYPE_PARAMS["mixcv"]) == _MIXCV_PARAMS
    expected = {"vector": 4, "cube": 8, "mixcv": 1152}
    for kernel_type, count in expected.items():
        configs = build_all_configs(kernel_type)
        normalized = [normalize_config(config, kernel_type) for config in configs]
        assert len(configs) == count
        assert len({json.dumps(value, sort_keys=True) for value in normalized}) == count
        assert all(value["multibuffer"] == (value["num_stages"] != 1) for value in normalized)


def test_config_and_modes_map_to_explicit_bisheng_options():
    config = {
        "num_stages": 2,
        "multibuffer": True,
        "unit_flag": True,
        "limit_auto_multi_buffer_only_for_local_buffer": True,
        "limit_auto_multi_buffer_of_local_buffer": "no-limit",
        "set_workspace_multibuffer": 4,
        "enable_hivm_auto_cv_balance": True,
        "tile_mix_vector_loop": 8,
        "tile_mix_cube_loop": 4,
        "enable_ubuf_saving": True,
    }
    arguments = config_to_bisheng_options(config)
    assert "--enable-auto-multi-buffer=true" in arguments
    assert "--enable-hivm-unit-flag-sync=true" in arguments
    assert "--limit-auto-multi-buffer-only-for-local-buffer=true" in arguments
    assert "--limit-auto-multi-buffer-of-local-buffer=no-limit" in arguments
    assert "--set-workspace-multibuffer=4" in arguments
    assert "--enable-hivm-auto-cv-balance=true" in arguments
    assert "--tile-mix-vector-loop=8" in arguments
    assert "--tile-mix-cube-loop=4" in arguments
    assert "--enable-ubuf-saving=true" in arguments
    assert mode_to_bisheng_options("baseline") == [
        "--enable-ub-overflow-prediction=false",
        "--prune-predicted-ub-overflow=false",
    ]
    assert mode_to_bisheng_options("shadow") == [
        "--enable-ub-overflow-prediction=true",
        "--prune-predicted-ub-overflow=false",
    ]
    assert mode_to_bisheng_options("prune") == [
        "--enable-ub-overflow-prediction=true",
        "--prune-predicted-ub-overflow=true",
    ]


def test_model_and_fallback_parser_preserves_all_attempts():
    stderr = """\
BISHENGIR_UB_MODEL_RESULT contract_version=1 status=overflow precision=exact overflow=true ub_peak_bits=32 required_bits=64 capacity_bits=48 selected_seed=1 decision_path=full_plan non_overflow_upper_bound_proven=false conservative_upper_bound_bits=unknown serialize_ns=2 model_ns=3 pipeline_fingerprint=test-pipeline diagnostic_category=predicted_ub_overflow
[BISHENG][FALLBACK][RETRY] ub overflow detected; automatically set enable-code-motion to false and retrying compilation.
BISHENGIR_UB_MODEL_RESULT contract_version=1 status=success precision=exact overflow=false ub_peak_bits=unknown required_bits=unknown capacity_bits=48 selected_seed=unknown decision_path=non_overflow_upper_bound non_overflow_upper_bound_proven=true conservative_upper_bound_bits=24 serialize_ns=4 model_ns=5 pipeline_fingerprint=test-pipeline diagnostic_category=none
"""
    attempts = parse_model_results(stderr)
    fallbacks = parse_fallback_actions(stderr)
    assert len(attempts) == 2
    assert attempts[0]["overflow"] is True
    assert attempts[1]["status"] == "success"
    assert attempts[1]["decision_path"] == "non_overflow_upper_bound"
    assert attempts[1]["non_overflow_upper_bound_proven"] is True
    assert attempts[1]["conservative_upper_bound_bits"] == 24
    assert fallbacks == [{"cause": "ub overflow", "option": "enable-code-motion", "value": "false"}]
    assert _status("prune", 1, False, stderr, attempts[:1], fallbacks) == (
        "predicted_ub_overflow_final",
        False,
    )
    assert _status("shadow", 1, False, stderr, attempts[:1], fallbacks) == (
        "native_ub_overflow_final",
        True,
    )
    assert _status("prune", 1, False, stderr, [attempts[1], attempts[0]], fallbacks) == (
        "predicted_ub_overflow_final",
        True,
    )
    assert parse_native_ub_result("error: ub overflow, requires 64 bits while 32 bits available") == {
        "required_bits": 64,
        "capacity_bits": 32,
    }


def test_executor_stops_after_plan_memory_and_accepts_no_binary(tmp_path):
    compiler = _make_executable(
        tmp_path / "bishengir-compile",
        """#!/usr/bin/env python3
import os
import sys
assert os.environ["BISHENGIR_STOP_AFTER_LOCAL_PLAN_MEMORY"] == "1"
assert os.environ["BISHENGIR_UB_MODEL_EMIT_RESULT"] == "1"
assert "--enable-ub-overflow-prediction=true" in sys.argv
assert "--prune-predicted-ub-overflow=true" in sys.argv
sys.stderr.write("BISHENGIR_UB_MODEL_RESULT contract_version=1 status=overflow precision=exact overflow=true ub_peak_bits=64 required_bits=64 capacity_bits=32 selected_seed=1 serialize_ns=2 model_ns=3 diagnostic_category=predicted_ub_overflow\\n")
sys.stderr.write("[BISHENG][FALLBACK][RETRY] ub overflow detected; automatically set enable-code-motion to false and retrying compilation.\\n")
sys.stderr.write("BISHENGIR_UB_MODEL_RESULT contract_version=1 status=success precision=exact overflow=false ub_peak_bits=16 required_bits=16 capacity_bits=32 selected_seed=2 serialize_ns=4 model_ns=5 diagnostic_category=none\\n")
""",
    )
    adapter = tmp_path / "input.ttadapter"
    adapter.write_text("module {}\n", encoding="utf-8")
    spec = AdapterSpec(adapter, "vector")
    options = CompileOnlyOptions(
        adapters=[spec],
        compiler=compiler,
        report_dir=tmp_path / "report",
        modes=("prune", ),
        repeat=1,
        fixed_bisheng_options=(),
        limit_configs=1,
        progress_interval=0,
    )
    config = normalize_config(build_all_configs("vector")[0], "vector")
    task = CompileTask(
        run_id="run",
        candidate_id="candidate",
        adapter=spec,
        adapter_digest="adapter",
        mode="prune",
        repeat=0,
        order_index=0,
        config_id="config",
        normalized_config=config,
    )
    result = AdapterCompileExecutor(options).run(task)
    assert result.status == "success_after_fallback"
    assert result.reached_plan_memory is True
    assert result.attempt_count == 2
    assert result.fallback_count == 1
    assert result.model_ns == 8
    assert not (tmp_path / "output.o").exists()


def test_checkpoint_resume_and_autotuner_branch(tmp_path, monkeypatch):
    compiler = _make_executable(tmp_path / "bishengir-compile", "#!/bin/sh\nexit 0\n")
    adapter = tmp_path / "input.ttadapter"
    adapter.write_text("module {}\n", encoding="utf-8")
    options = CompileOnlyOptions(
        adapters=[AdapterSpec(adapter, "vector")],
        compiler=compiler,
        report_dir=tmp_path / "report",
        modes=("baseline", ),
        repeat=1,
        fixed_bisheng_options=(),
        limit_configs=1,
        progress_interval=0,
    )
    first = run_adapter_compile_only(options)
    assert first.total_candidates == 1
    assert first.executed_candidates == 1
    assert first.mode_summaries["baseline"]["reached_plan_memory"] == 1

    options.resume = True
    resumed = run_adapter_compile_only(options)
    assert resumed.executed_candidates == 0
    assert resumed.resumed_candidates == 1

    sentinel = object()
    monkeypatch.setattr(
        "triton.backends.ascend.runtime.adapter_compile_only.run_adapter_compile_only",
        lambda received: sentinel if received is options else None,
    )
    tuner = AutoTilingTuner(
        _marker,
        [],
        [],
        [],
        None,
        None,
        compile_only=True,
        compile_only_options=options,
    )
    assert tuner.run() is sentinel
    assert tuner.best_config is None


def test_checkpoint_repairs_a_partial_final_line(tmp_path):
    checkpoint = tmp_path / "results.jsonl"
    checkpoint.write_text(
        '{"run_id":"run","candidate_id":"done"}\n{"run_id":"run"',
        encoding="utf-8",
    )
    completed, rows = _load_completed(checkpoint, "run")
    assert completed == {"done"}
    assert len(rows) == 1
    assert checkpoint.read_text(encoding="utf-8") == '{"run_id":"run","candidate_id":"done"}\n'


def test_experiment_metrics_use_paired_candidates_and_exclude_overflow():
    def row(mode, config_id, wall_ns, overflow=False):
        return {
            "adapter_digest": "adapter",
            "config_id": config_id,
            "repeat": 0,
            "mode": mode,
            "candidate_wall_ns": wall_ns,
            "compiler_returncode": 1 if overflow else 0,
            "timed_out": False,
            "fallback_count": 0,
            "overflow": overflow,
            "attempt_results": [{"overflow": overflow}] if mode != "baseline" else [],
        }

    rows = [
        row("baseline", "ok", 100),
        row("shadow", "ok", 110),
        row("prune", "ok", 108),
        row("baseline", "overflow", 200, overflow=True),
        row("shadow", "overflow", 205, overflow=True),
        row("prune", "overflow", 50, overflow=True),
    ]
    metrics = _experiment_metrics(rows)
    overhead = metrics["no_overflow_model_overhead"]
    assert overhead["paired_candidate_runs"] == 1
    assert overhead["average_overhead_per_candidate_ns"] == 10
    speedup = metrics["overall_prune_speedup"]
    assert speedup["paired_candidate_runs"] == 2
    assert speedup["time_saved_ns"] == 142
    assert speedup["time_saved_percent"] == 100.0 * 142 / 300
