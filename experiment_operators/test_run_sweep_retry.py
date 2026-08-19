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


class SweepRetryTest(unittest.TestCase):

    def test_recovered_timeout_requires_overwrite_confirmation(self):
        self.assertFalse(run_sweep.row_is_timeout({
            "timed_out": False,
            "diagnostic": "曾超时，补测后成功",
        }))

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

                print("BENCHMARK operator=retry_test latency_ms=1.25 warmup=1 active=1", flush=True)
                """),
            encoding="utf-8",
        )
        return candidate

    def settings(self, root: Path, retries: int) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(experiment_config, "FIRST_AXIS_VALUES", (1, )))
        stack.enter_context(patch.object(experiment_config, "MULTIBUFFER_NUM_VALUES", (1, )))
        stack.enter_context(patch.object(experiment_config, "VF_MERGE_LEVEL_VALUES", (0, 1)))
        stack.enter_context(patch.object(experiment_config, "WARMUP", 1))
        stack.enter_context(patch.object(experiment_config, "ACTIVE", 1))
        stack.enter_context(patch.object(experiment_config, "CASE_TIMEOUT_SECONDS", 0.2))
        stack.enter_context(patch.object(experiment_config, "TIMEOUT_RETRIES", retries))
        stack.enter_context(patch.object(run_sweep, "RESULTS_ROOT", root / "results"))
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
            self.assertLess(text.index("running 1/2"), text.index("running 2/2"))
            self.assertLess(text.index("running 2/2"), text.index(retry_message))
            self.assertLess(text.index(retry_message), text.index("attempt=2(automatic_retry)"))

            directory = self.result_dir(root)
            rows = self.load_rows(directory)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["attempt_count"], 2)
            self.assertEqual(rows[0]["timeout_retries_used"], 1)
            self.assertTrue(rows[0]["initial_timed_out"])
            self.assertFalse(rows[0]["timed_out"])
            self.assertEqual(
                [attempt["timed_out"] for attempt in rows[0]["attempt_history"]],
                [True, False],
            )
            self.assertEqual(rows[1]["attempt_count"], 1)
            self.assertTrue((directory / "results.csv").is_file())
            first_log = (directory / "logs/d1-b1-m0.log").read_text()
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
                "multibuffer_num": 1,
                "vf_merge_level": merge,
                "status": "measured",
                "correctness_status": "passed",
                "diagnostic": "",
                "latency_ms": 1.0 + merge,
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
                "axes": {
                    "depth": [1],
                    "multibuffer_num": [1],
                    "vf_merge_level": [0, 1],
                },
            }
            (result / "manifest.json").write_text(json.dumps(manifest))

            latest = generate_experiment_report.find_latest_report_runs(root)
            self.assertEqual(set(latest), {"retry_test"})
            self.assertEqual(latest["retry_test"]["source_format"], "results.csv")
            report = generate_experiment_report.report_data(latest)
            self.assertEqual(report["operators"][0]["row_count"], 2)
            self.assertEqual(report["operators"][0]["measured_count"], 2)


if __name__ == "__main__":
    unittest.main()
