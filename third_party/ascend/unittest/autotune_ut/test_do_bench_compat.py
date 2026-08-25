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

import sys
from types import MethodType, SimpleNamespace

import pytest
import triton
import triton.backends.ascend.testing as ascend_testing
from triton.runtime.autotuner import Config
from triton.backends.ascend.runtime.autotuner import AutoTilingTuner


class _FakeProfilerContext:

    def __init__(self):
        self.steps = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def step(self):
        self.steps += 1


def _make_tuner(do_bench):
    tuner = object.__new__(AutoTilingTuner)
    tuner.compile_parallel = False
    tuner.do_bench = do_bench
    tuner.user_defined_do_bench = True

    def _make_kernel_call(self, *args, config, **meta):

        def kernel_call(warmup):
            return None

        return kernel_call

    tuner._make_kernel_call = MethodType(_make_kernel_call, tuner)
    return tuner


def test_do_bench_npu_completes_scheduled_profile(monkeypatch, tmp_path):
    schedule_args = {}
    profile_args = {}
    profile = _FakeProfilerContext()
    synchronize_calls = []
    function_calls = [0, 0]
    cleanup_args = []

    def schedule(**kwargs):
        schedule_args.update(kwargs)
        return "complete-schedule"

    def make_profile(**kwargs):
        profile_args.update(kwargs)
        return profile

    fake_profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kwargs: kwargs,
        AiCMetrics=SimpleNamespace(PipeUtilization="pipe-utilization"),
        ProfilerLevel=SimpleNamespace(Level1="level1"),
        ProfilerActivity=SimpleNamespace(NPU="npu"),
        schedule=schedule,
        profile=make_profile,
        tensorboard_trace_handler=lambda path: ("trace-handler", path),
    )
    fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: synchronize_calls.append(True)))
    fake_torch_npu = SimpleNamespace(profiler=fake_profiler)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    monkeypatch.setattr(ascend_testing, "_collect_prof_result", lambda *args, **kwargs: [1.0, 2.0])
    monkeypatch.setattr(ascend_testing, "_rm_dic", lambda *args, **kwargs: cleanup_args.append((args, kwargs)))

    def first():
        function_calls[0] += 1

    def second():
        function_calls[1] += 1

    result = ascend_testing.do_bench_npu(
        [first, second],
        warmup=2,
        active=3,
        prof_dir=str(tmp_path),
        target_kernel_name="target",
    )

    assert result == [1.0, 2.0]
    assert schedule_args == {"wait": 0, "warmup": 1, "active": 10, "repeat": 1}
    assert profile_args["schedule"] == "complete-schedule"
    assert profile.steps == 11
    assert function_calls == [7, 7]
    assert len(synchronize_calls) == 14
    assert cleanup_args == [((False, str(tmp_path)), {})]


def test_do_bench_npu_preserves_profile_when_collection_fails(monkeypatch, tmp_path):
    profile = _FakeProfilerContext()
    cleanup_args = []
    fake_profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kwargs: kwargs,
        AiCMetrics=SimpleNamespace(PipeUtilization="pipe-utilization"),
        ProfilerLevel=SimpleNamespace(Level1="level1"),
        ProfilerActivity=SimpleNamespace(NPU="npu"),
        schedule=lambda **kwargs: "complete-schedule",
        profile=lambda **kwargs: profile,
        tensorboard_trace_handler=lambda path: ("trace-handler", path),
    )
    fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
    fake_torch_npu = SimpleNamespace(profiler=fake_profiler)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    def fail_collection(*args, **kwargs):
        raise ascend_testing.ProfilerDataUnavailableError(str(tmp_path))

    monkeypatch.setattr(ascend_testing, "_collect_prof_result", fail_collection)
    monkeypatch.setattr(ascend_testing, "_rm_dic", lambda *args, **kwargs: cleanup_args.append((args, kwargs)))

    with pytest.raises(ascend_testing.ProfilerDataUnavailableError, match="profile data preserved"):
        ascend_testing.do_bench_npu(lambda: None, warmup=1, active=1, prof_dir=str(tmp_path))

    assert cleanup_args == [((True, str(tmp_path)), {})]


def test_do_bench_npu_can_fall_back_to_event_timing(monkeypatch, tmp_path, capsys):
    profile = _FakeProfilerContext()
    cleanup_args = []
    fallback_args = []
    fake_profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kwargs: kwargs,
        AiCMetrics=SimpleNamespace(PipeUtilization="pipe-utilization"),
        ProfilerLevel=SimpleNamespace(Level1="level1"),
        ProfilerActivity=SimpleNamespace(NPU="npu"),
        schedule=lambda **kwargs: "complete-schedule",
        profile=lambda **kwargs: profile,
        tensorboard_trace_handler=lambda path: ("trace-handler", path),
    )
    fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
    fake_torch_npu = SimpleNamespace(profiler=fake_profiler)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    def fail_collection(*args, **kwargs):
        raise ascend_testing.ProfilerDataUnavailableError(str(tmp_path))

    def event_fallback(*args, **kwargs):
        fallback_args.append((args, kwargs))
        return 1.25

    monkeypatch.setattr(ascend_testing, "_collect_prof_result", fail_collection)
    monkeypatch.setattr(ascend_testing, "_do_bench_npu_events", event_fallback)
    monkeypatch.setattr(ascend_testing, "_rm_dic", lambda *args, **kwargs: cleanup_args.append((args, kwargs)))

    fn = lambda: None
    result = ascend_testing.do_bench_npu(
        fn,
        warmup=2,
        active=3,
        prof_dir=str(tmp_path),
        fallback_to_event_timing=True,
    )

    assert result == 1.25
    assert fallback_args == [(([fn], 2, 3, False), {})]
    assert cleanup_args == [((True, str(tmp_path)), {})]
    assert "NPU_BENCHMARK_METHOD=npu_event_fallback" in capsys.readouterr().out


def test_batch_bench_supports_do_bench_with_quantiles():
    record = {}

    def _do_bench(fn, quantiles):
        record["quantiles"] = quantiles
        fn()
        return (1.0, 1.0, 1.0)

    tuner = _make_tuner(_do_bench)
    cfg = Config({})

    result = tuner._batch_bench(configs=[cfg])

    assert result[cfg] == (1.0, 1.0, 1.0)
    assert record["quantiles"] == (0.5, 0.2, 0.8)


def test_batch_bench_requires_do_bench_quantiles_parameter():

    def _do_bench(fn):
        fn()
        return (2.0, 2.0, 2.0)

    tuner = _make_tuner(_do_bench)
    cfg = Config({})

    with pytest.raises(TypeError):
        tuner._batch_bench(configs=[cfg])


def test_batch_bench_npu_env_respects_user_do_bench(monkeypatch):
    calls = {"do_bench": 0}

    def _do_bench(fn, quantiles):
        calls["do_bench"] += 1
        fn()
        return (3.0, 3.0, 3.0)

    def _unexpected_do_bench_npu(*args, **kwargs):
        raise AssertionError("do_bench_npu should not be used when user do_bench is provided")

    tuner = _make_tuner(_do_bench)
    cfg0 = Config({"ID": 0})
    cfg1 = Config({"ID": 1})
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    monkeypatch.setattr("triton.backends.ascend.testing.do_bench_npu", _unexpected_do_bench_npu)

    result = tuner._batch_bench(configs=[cfg0, cfg1])

    assert calls["do_bench"] == 2
    assert result[cfg0] == (3.0, 3.0, 3.0)
    assert result[cfg1] == (3.0, 3.0, 3.0)


def test_batch_bench_npu_env_uses_do_bench_npu_without_user_do_bench(monkeypatch):

    def _do_bench(fn, quantiles):
        raise AssertionError("self.do_bench should not be used when no user do_bench is provided")

    calls = {"do_bench_npu": 0}

    def _do_bench_npu(funcs, clear_l2_cache=False, warmup=5, active=30, target_kernel_name=None, **kwargs):
        calls["do_bench_npu"] += 1
        assert len(funcs) == 2
        return [1.0, 2.0]

    tuner = _make_tuner(_do_bench)
    tuner.user_defined_do_bench = False
    cfg0 = Config({"ID": 0})
    cfg1 = Config({"ID": 1})
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    monkeypatch.setattr("triton.backends.ascend.testing.do_bench_npu", _do_bench_npu)

    result = tuner._batch_bench(configs=[cfg0, cfg1])

    assert calls["do_bench_npu"] == 1
    assert result[cfg0] == 1.0
    assert result[cfg1] == 2.0


def test_autotilingtuner_marks_user_defined_do_bench():
    marker = {"called": False}

    def _do_bench(fn, quantiles):
        marker["called"] = True
        return (0.0, 0.0, 0.0)

    def _dummy_kernel():
        return None

    _dummy_kernel.arg_names = []

    tuner = AutoTilingTuner(
        _dummy_kernel,
        [],
        [Config({})],
        [],
        None,
        None,
        do_bench=_do_bench,
    )

    assert tuner.user_defined_do_bench is True
    assert marker["called"] is False


def test_ascend_autotune_decorator_forwards_do_bench(monkeypatch):
    import triton.backends.ascend.runtime.autotuner as ascend_autotuner

    captured = {}

    class DummyAutoTilingTuner:

        def __init__(self, *args, **kwargs):
            captured["do_bench"] = kwargs.get("do_bench")

    monkeypatch.setattr(ascend_autotuner, "AutoTilingTuner", DummyAutoTilingTuner)

    def _dummy_kernel():
        return None

    _dummy_kernel.arg_names = []
    my_do_bench = lambda kernel_call, quantiles: (0.0, 0.0, 0.0)

    ascend_autotuner.autotune(configs=[object()], key=[], do_bench=my_do_bench)(_dummy_kernel)

    assert captured["do_bench"] is my_do_bench
