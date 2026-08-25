import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

TESTING_PATH = Path(__file__).resolve().parents[1] / "third_party/ascend/backend/testing.py"


class FakeProfilerContext:

    def __init__(self):
        self.steps = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def step(self):
        self.steps += 1


def load_testing_module():
    triton = ModuleType("triton")
    runtime = ModuleType("triton.runtime")
    runtime.driver = SimpleNamespace(active=SimpleNamespace())
    knobs = ModuleType("triton.knobs")
    knobs.cache = SimpleNamespace(get_triton_dir=lambda name: f"/unused/{name}")
    triton.runtime = runtime
    triton.knobs = knobs
    spec = importlib.util.spec_from_file_location("isolated_ascend_testing", TESTING_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
            "triton": triton,
            "triton.runtime": runtime,
            "triton.knobs": knobs,
    }):
        spec.loader.exec_module(module)
    return module


def fake_torch_modules(profile, schedule_args):

    def schedule(**kwargs):
        schedule_args.update(kwargs)
        return "scheduled-profile"

    profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kwargs: kwargs,
        AiCMetrics=SimpleNamespace(PipeUtilization="pipe-utilization"),
        ProfilerLevel=SimpleNamespace(Level1="level1"),
        ProfilerActivity=SimpleNamespace(NPU="npu"),
        schedule=schedule,
        profile=lambda **kwargs: profile,
        tensorboard_trace_handler=lambda path: ("trace-handler", path),
    )
    torch = ModuleType("torch")
    torch.npu = SimpleNamespace(synchronize=lambda: None)
    torch_npu = ModuleType("torch_npu")
    torch_npu.profiler = profiler
    return torch, torch_npu


class ProfilerFallbackTest(unittest.TestCase):

    def test_profiler_schedule_completes_and_collects(self):
        testing = load_testing_module()
        profile = FakeProfilerContext()
        schedule_args = {}
        torch, torch_npu = fake_torch_modules(profile, schedule_args)
        cleanup = []
        calls = []
        testing._collect_prof_result = lambda *args, **kwargs: 1.5
        testing._rm_dic = lambda *args: cleanup.append(args)

        with patch.dict(sys.modules, {"torch": torch, "torch_npu": torch_npu}):
            result = testing.do_bench_npu(
                lambda: calls.append(True),
                warmup=2,
                active=3,
                prof_dir="/profile",
            )

        self.assertEqual(result, 1.5)
        self.assertEqual(schedule_args, {"wait": 0, "warmup": 1, "active": 5, "repeat": 1})
        self.assertEqual(profile.steps, 6)
        self.assertEqual(len(calls), 7)
        self.assertEqual(cleanup, [(False, "/profile")])

    def test_missing_profile_can_fall_back_to_events(self):
        testing = load_testing_module()
        profile = FakeProfilerContext()
        schedule_args = {}
        torch, torch_npu = fake_torch_modules(profile, schedule_args)
        cleanup = []
        fallback = []

        def fail_collection(*args, **kwargs):
            raise testing.ProfilerDataUnavailableError("/profile")

        def event_fallback(*args, **kwargs):
            fallback.append((args, kwargs))
            return 2.5

        testing._collect_prof_result = fail_collection
        testing._do_bench_npu_events = event_fallback
        testing._rm_dic = lambda *args: cleanup.append(args)
        output = io.StringIO()
        fn = lambda: None
        with patch.dict(sys.modules, {"torch": torch, "torch_npu": torch_npu}), contextlib.redirect_stdout(output):
            result = testing.do_bench_npu(
                fn,
                warmup=2,
                active=3,
                prof_dir="/profile",
                fallback_to_event_timing=True,
            )

        self.assertEqual(result, 2.5)
        self.assertEqual(fallback, [(([fn], 2, 3, False), {})])
        self.assertEqual(cleanup, [(True, "/profile")])
        self.assertIn("NPU_BENCHMARK_METHOD=npu_event_fallback", output.getvalue())


if __name__ == "__main__":
    unittest.main()
