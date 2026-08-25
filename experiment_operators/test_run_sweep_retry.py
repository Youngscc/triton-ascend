import ast
import builtins
from contextlib import ExitStack, redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from experiment_operators import experiment_config
from experiment_operators import generate_experiment_report
from experiment_operators import run_sweep


class TtyBuffer(io.StringIO):

    def isatty(self):
        return True

    def fileno(self):
        raise OSError("test stream has no file descriptor")


class SweepRetryTest(unittest.TestCase):

    @staticmethod
    def compile_options_function(candidate: Path):
        tree = ast.parse(candidate.read_text(encoding="utf-8"), filename=str(candidate))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "_experiment_compile_options")
        namespace = {"os": os}
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(module, str(candidate), "exec"), namespace)
        return namespace["_experiment_compile_options"]

    def test_recovered_timeout_requires_overwrite_confirmation(self):
        self.assertFalse(run_sweep.row_is_timeout({
            "timed_out": False,
            "diagnostic": "曾超时，补测后成功",
        }))

    def test_interactive_progress_reuses_two_terminal_lines(self):
        stream = TtyBuffer()
        row = {
            "status": "measured",
            "latency_ms": 1.25,
            "required_ub_kib": 64.0,
            "log_path": "logs/d1-boff-m0.log",
        }
        with patch.object(run_sweep.sys, "stdout", stream):
            progress = run_sweep.SweepProgress("retry_test", 2)
            progress.begin(1, run_sweep.SweepConfig(False, 1, None, 0), "depth", 1, "initial")
            progress.finish("d1-boff-m0", row)
            progress.close()

        output = stream.getvalue()
        self.assertIn("current: running 1/2 depth=1 multibuffer_num=off", output)
        self.assertIn("\033[1A", output)
        self.assertIn("current: finished case=d1-boff-m0 status=measured", output)
        self.assertNotIn("CASE_START", output)
        self.assertTrue(output.endswith("\n"))

    def write_candidate(self, root: Path) -> Path:
        candidate = root / "retry_candidate.py"
        candidate.write_text(
            textwrap.dedent("""
                import os
                from pathlib import Path
                import time

                state_dir = Path(os.environ["RETRY_TEST_STATE"])
                state_dir.mkdir(parents=True, exist_ok=True)
                marker = state_dir / "first-config-timed-out"
                if os.environ["EXPERIMENT_VF_MERGE_LEVEL"] == "0" and not marker.exists():
                    marker.write_text("timeout\\n")
                    print("RETRY_TEST_INITIAL_TIMEOUT", flush=True)
                    time.sleep(60)

                if os.environ["EXPERIMENT_VF_MERGE_LEVEL"] == "1":
                    print("NPU_BENCHMARK_METHOD=npu_event_fallback", flush=True)
                print("BENCHMARK operator=retry_test latency_ms=1.25 warmup=1 active=1", flush=True)
                """),
            encoding="utf-8",
        )
        return candidate

    def settings(self, root: Path, retries: int) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(experiment_config, "A3_DEPTH_VALUES", (1, )))
        stack.enter_context(patch.object(experiment_config, "A5_INTRA_CACHE_NUM_VALUES", ("off", 1)))
        stack.enter_context(patch.object(experiment_config, "MULTIBUFFER_NUM_VALUES", (1, )))
        stack.enter_context(patch.object(experiment_config, "VF_MERGE_LEVEL_VALUES", (0, 1)))
        stack.enter_context(patch.object(experiment_config, "WARMUP", 1))
        stack.enter_context(patch.object(experiment_config, "ACTIVE", 1))
        stack.enter_context(patch.object(experiment_config, "CASE_TIMEOUT_SECONDS", 0.2))
        stack.enter_context(patch.object(experiment_config, "TIMEOUT_RETRIES", retries))
        stack.enter_context(patch.object(run_sweep, "RESULTS_ROOT", root / "results"))
        stack.enter_context(patch.object(run_sweep.sys, "stderr", io.StringIO()))
        stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "BISHENGIR_NATIVE_A5_REGBASE": "",
                    "RETRY_TEST_STATE": str(root / "state"),
                    "TRITON_CACHE_DIR": str(root / "cache"),
                },
            ))
        return stack

    def test_default_config_includes_real_disabled_states(self):
        a3_first, multibuffer, vf_merge = run_sweep.configured_values(False)
        self.assertEqual(a3_first, (1, 2, 3, 4))
        self.assertEqual(multibuffer, ("off", 1, 2, 3, 4))
        a3_configs = list(run_sweep.requested_configs(False, a3_first, multibuffer, vf_merge))
        self.assertEqual(len(a3_configs), 40)
        self.assertEqual(a3_configs[0], run_sweep.SweepConfig(False, 1, None, 0))

        a5_first, multibuffer, vf_merge = run_sweep.configured_values(True)
        self.assertEqual(a5_first, ("off", 1, 2, 3, 4))
        a5_configs = list(run_sweep.requested_configs(True, a5_first, multibuffer, vf_merge))
        self.assertEqual(len(a5_configs), 50)
        self.assertEqual(a5_configs[0], run_sweep.SweepConfig(False, None, None, 0))
        self.assertFalse(any(config.dynamic_cv_pipeline for config in a5_configs[:10]))
        self.assertEqual(a5_configs[10], run_sweep.SweepConfig(True, 1, None, 0))

    def test_disabled_state_environment_and_manual_case(self):
        config = run_sweep.parse_manual_config("off", "off", "0", True)
        self.assertEqual(config, run_sweep.SweepConfig(False, None, None, 0))
        environment = run_sweep.candidate_environment(config, True)
        self.assertEqual(environment["EXPERIMENT_DYNAMIC_CV"], "0")
        self.assertEqual(environment["EXPERIMENT_DEPTH"], "1")
        self.assertEqual(environment["EXPERIMENT_MULTIBUFFER"], "0")
        self.assertNotIn("EXPERIMENT_INTRA_CACHE_NUM", environment)
        self.assertNotIn("EXPERIMENT_MULTIBUFFER_NUM", environment)

        enabled = run_sweep.parse_manual_config("2", "3", "1", True)
        environment = run_sweep.candidate_environment(enabled, True)
        self.assertEqual(environment["EXPERIMENT_DYNAMIC_CV"], "1")
        self.assertEqual(environment["EXPERIMENT_INTRA_CACHE_NUM"], "2")
        self.assertEqual(environment["EXPERIMENT_MULTIBUFFER"], "1")
        self.assertEqual(environment["EXPERIMENT_MULTIBUFFER_NUM"], "3")

    def test_every_candidate_forwards_multibuffer_disabled_state(self):
        candidates = (
            "fused_attention.py",
            "flash_attention_npu_v8.py",
            "hstu_attention.py",
            "unified_attention.py",
        )
        candidate_dir = Path(__file__).with_name("candidates")
        for name in candidates:
            compile_options = self.compile_options_function(candidate_dir / name)
            with self.subTest(candidate=name), patch.dict(
                    os.environ, {
                        "EXPERIMENT_DYNAMIC_CV": "0",
                        "EXPERIMENT_DEPTH": "1",
                        "EXPERIMENT_MULTIBUFFER": "0",
                        "EXPERIMENT_VF_MERGE_LEVEL": "0",
                    }, clear=True):
                options = compile_options()
                self.assertFalse(options["enable_dynamic_cv_pipeline"])
                self.assertFalse(options["multibuffer"])
                self.assertNotIn("multibuffer_num", options)

            with self.subTest(candidate=name), patch.dict(
                    os.environ, {
                        "EXPERIMENT_DYNAMIC_CV": "1",
                        "EXPERIMENT_INTRA_CACHE_NUM": "2",
                        "EXPERIMENT_MULTIBUFFER": "1",
                        "EXPERIMENT_MULTIBUFFER_NUM": "3",
                        "EXPERIMENT_VF_MERGE_LEVEL": "1",
                    }, clear=True):
                options = compile_options()
                self.assertTrue(options["enable_dynamic_cv_pipeline"])
                self.assertTrue(options["multibuffer"])
                self.assertEqual(options["multibuffer_num"], 3)

    def result_dir(self, root: Path) -> Path:
        directories = list((root / "results").glob("*-retry_test"))
        self.assertEqual(len(directories), 1)
        return directories[0]

    @staticmethod
    def load_rows(directory: Path) -> list[dict]:
        return [json.loads(line) for line in (directory / "measurements.jsonl").read_text().splitlines()]

    def test_timeout_retry_runs_after_initial_sweep(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.write_candidate(root)
            output = io.StringIO()
            with self.settings(root, retries=1), redirect_stdout(output):
                self.assertEqual(run_sweep.main([str(candidate)]), 0)

            text = output.getvalue()
            retry_message = ("initial sweep complete; retrying 1 timed-out configuration(s)")
            self.assertEqual(text.count("CASE_START "), 3)
            self.assertEqual(text.count("CASE_RESULT "), 2)
            self.assertNotIn("[retry_test] [", text)
            self.assertLess(text.index("running 1/2"), text.index("running 2/2"))
            self.assertLess(text.index("running 2/2"), text.index(retry_message))
            self.assertLess(text.index(retry_message), text.index("attempt=2(automatic_retry)"))

            directory = self.result_dir(root)
            rows = self.load_rows(directory)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["attempt_count"], 2)
            self.assertEqual(rows[0]["timeout_retries_used"], 1)
            self.assertEqual(rows[0]["benchmark_method"], "npu_profiler")
            self.assertTrue(rows[0]["initial_timed_out"])
            self.assertFalse(rows[0]["timed_out"])
            self.assertEqual(
                [attempt["timed_out"] for attempt in rows[0]["attempt_history"]],
                [True, False],
            )
            self.assertEqual(rows[1]["attempt_count"], 1)
            self.assertEqual(rows[1]["benchmark_method"], "npu_event_fallback")
            self.assertTrue((directory / "results.csv").is_file())
            first_log = (directory / "logs/d1-b1-m0.log").read_text()
            self.assertIn("[EXPERIMENT] TIMEOUT_PROCESS_SNAPSHOT", first_log)
            self.assertLess(
                first_log.index("attempt 1 (initial)"),
                first_log.index("attempt 2 (automatic_retry)"),
            )

    def test_manual_case_refill_and_overwrite_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.write_candidate(root)
            output = io.StringIO()
            with self.settings(root, retries=0), redirect_stdout(output):
                self.assertEqual(run_sweep.main([str(candidate)]), 0)
                directory = self.result_dir(root)
                self.assertTrue(self.load_rows(directory)[0]["timed_out"])

                # A timeout row is refilled without asking.
                self.assertEqual(
                    run_sweep.main(["--case", str(candidate), "1", "1", "0"]),
                    0,
                )
                rows = self.load_rows(directory)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["attempt_count"], 2)
                self.assertEqual(rows[0]["manual_rerun_count"], 1)
                self.assertFalse(rows[0]["timed_out"])

                # Declining leaves a non-timeout row untouched.
                second_attempts = rows[1]["attempt_count"]
                with patch.object(builtins, "input", return_value="n"):
                    self.assertEqual(
                        run_sweep.main(["--case", str(candidate), "1", "1", "1"]),
                        0,
                    )
                self.assertEqual(self.load_rows(directory)[1]["attempt_count"], second_attempts)

                # Confirmation reruns and overwrites that same row.
                with patch.object(builtins, "input", return_value="y"):
                    self.assertEqual(
                        run_sweep.main(["--case", str(candidate), "1", "1", "1"]),
                        0,
                    )
                rows = self.load_rows(directory)
                self.assertEqual(rows[1]["attempt_count"], second_attempts + 1)
                self.assertEqual(rows[1]["manual_rerun_count"], 1)
                with (directory / "results.csv").open(newline="", encoding="utf-8") as handle:
                    csv_rows = list(csv.DictReader(handle))
                self.assertEqual(len(csv_rows), 2)
                self.assertEqual(csv_rows[0]["手动补测次数"], "1")
                self.assertEqual(csv_rows[1]["手动补测次数"], "1")

            self.assertIn("case_update=timeout_refill", output.getvalue())

    def test_report_accepts_existing_results_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "20260819T120000+0800-retry_test"
            (result / "logs").mkdir(parents=True)
            rows = [{
                "depth": 1,
                "enable_dynamic_cv_pipeline": False,
                "enable_auto_multi_buffer": bool(merge),
                "multibuffer_num": 1 if merge else "off",
                "resolved_local_multibuffer_num": 1 if merge else None,
                "vf_merge_level": merge,
                "status": "measured",
                "correctness_status": "passed",
                "diagnostic": "",
                "latency_ms": 1.0 + merge,
                "benchmark_method": "npu_profiler" if merge == 0 else "npu_event_fallback",
                "required_ub_kib": 64.0,
                "wall_time_s": 2.0,
                "attempt_count": 1,
                "timed_out": False,
                "log_path": str(result / f"logs/d1-b1-m{merge}.log"),
            } for merge in (0, 1)]
            run_sweep.write_results(rows, result, "depth")
            manifest = {
                "run_id": "20260819T120000+0800",
                "operator": "retry_test",
                "experiment_schema": "existing-results-csv",
                "requested_configuration_count": 2,
                "executed_configuration_count": 2,
            }
            (result / "manifest.json").write_text(json.dumps(manifest))

            latest = generate_experiment_report.find_latest_report_runs(root)
            self.assertEqual(set(latest), {"retry_test"})
            self.assertEqual(latest["retry_test"]["source_format"], "results.csv")
            self.assertEqual(latest["retry_test"]["manifest"]["axes"]["multibuffer_num"], ["off", 1])
            report = generate_experiment_report.report_data(latest)
            self.assertEqual(report["operators"][0]["row_count"], 2)
            self.assertEqual(report["operators"][0]["measured_count"], 2)
            self.assertEqual(report["operators"][0]["rows"][0]["multibuffer_num"], "off")
            self.assertFalse(report["operators"][0]["rows"][0]["enable_auto_multi_buffer"])
            self.assertEqual(report["operators"][0]["rows"][1]["benchmark_method"], "npu_event_fallback")

            combined_path = root / "latest-summary/combined-results.csv"
            self.assertEqual(generate_experiment_report.write_combined_results_csv(latest, combined_path), 2)
            with combined_path.open(newline="", encoding="utf-8-sig") as handle:
                combined_rows = list(csv.DictReader(handle))
            self.assertEqual(len(combined_rows), 2)
            self.assertEqual(combined_rows[0]["算子"], "retry_test")
            self.assertEqual(combined_rows[0]["run_id"], "20260819T120000+0800")
            self.assertEqual(combined_rows[0]["结果目录"], result.name)
            self.assertEqual(combined_rows[0]["multibuffer_num"], "off")
            self.assertEqual(combined_rows[1]["vf_merge_level"], "1")

    def test_combined_report_csv_contains_all_four_operators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            operators = [
                "fused_attention",
                "flash_attention_npu_v8",
                "hstu_attention",
                "unified_attention",
            ]
            for operator in operators:
                result = root / f"20260825T120000+0800-{operator}"
                result.mkdir()
                rows = [{
                    "depth": 1,
                    "enable_dynamic_cv_pipeline": False,
                    "enable_auto_multi_buffer": False,
                    "multibuffer_num": "off",
                    "vf_merge_level": merge,
                    "status": "measured",
                    "correctness_status": "passed",
                    "diagnostic": "",
                    "latency_ms": 1.0 + merge,
                    "required_ub_kib": 64.0,
                    "wall_time_s": 2.0,
                    "attempt_count": 1,
                    "timed_out": False,
                    "log_path": str(result / f"d1-boff-m{merge}.log"),
                } for merge in (0, 1)]
                run_sweep.write_results(rows, result, "depth")
                (result / "manifest.json").write_text(
                    json.dumps({
                        "run_id": "20260825T120000+0800",
                        "operator": operator,
                        "requested_configuration_count": 2,
                        "executed_configuration_count": 2,
                    }))

            latest = generate_experiment_report.find_latest_report_runs(root)
            combined_path = root / "latest-summary/combined-results.csv"
            self.assertEqual(generate_experiment_report.write_combined_results_csv(latest, combined_path), 8)
            with combined_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames[:3], ["算子", "run_id", "结果目录"])
                combined_rows = list(reader)
            self.assertEqual(len(combined_rows), 8)
            self.assertEqual(
                [row["算子"] for row in combined_rows],
                [operator for operator in sorted(operators) for _ in range(2)],
            )
            self.assertEqual([row["vf_merge_level"] for row in combined_rows[:2]], ["0", "1"])

    def test_case_refill_selects_only_current_experiment_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_result(run_id: str, schema: str) -> Path:
                result = root / f"{run_id}-retry_test"
                result.mkdir()
                (result / "manifest.json").write_text(
                    json.dumps({
                        "run_id": run_id,
                        "operator": "retry_test",
                        "experiment_schema": schema,
                        "requested_configuration_count": 1,
                        "executed_configuration_count": 1,
                        "axes": {"depth": [1]},
                    }))
                (result / "measurements.jsonl").write_text(json.dumps({"status": "measured"}) + "\n")
                return result

            expected = write_result("20260825T120000+0800", run_sweep.A3_EXPERIMENT_SCHEMA)
            write_result("20260825T130000+0800", "incompatible-newer-schema")
            with patch.object(run_sweep, "RESULTS_ROOT", root):
                selected, _, _ = run_sweep.find_latest_operator_run("retry_test", "depth",
                                                                    run_sweep.A3_EXPERIMENT_SCHEMA)

            self.assertEqual(selected, expected)


if __name__ == "__main__":
    unittest.main()
