import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]
RUN_SWEEP = ROOT / "experiment_operators/run_sweep.py"


def run_retry_sweep(tmp_path: Path, *, simple_output: bool = False):
    candidate = tmp_path / "retry_candidate.py"
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
    result_dir = tmp_path / "results"
    env = os.environ.copy()
    env.pop("BISHENGIR_NATIVE_A5_REGBASE", None)
    env.update({
        "RETRY_TEST_STATE": str(tmp_path / "state"),
        "SWEEP_PROGRESS_MODE": "off",
        "TRITON_CACHE_DIR": str(tmp_path / "cache"),
    })
    command = [
        sys.executable,
        "-u",
        str(RUN_SWEEP),
        "--operator-file",
        str(candidate),
        "--operator-name",
        "retry_test",
        "--warmup",
        "1",
        "--active",
        "1",
        "--timeout",
        "0.2",
        "--timeout-retries",
        "1",
        "--limit",
        "2",
        "--output-dir",
        str(result_dir),
    ]
    if simple_output:
        command.append("--simple-output")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    return completed, result_dir


def assert_retry_order(completed: subprocess.CompletedProcess) -> None:
    assert completed.returncode == 0, completed.stdout + completed.stderr
    expected_retry_message = "initial sweep complete; retrying 1 timed-out configuration(s)"
    assert expected_retry_message in completed.stdout, completed.stdout + completed.stderr
    first_initial = completed.stdout.index("[1/2 initial]")
    second_initial = completed.stdout.index("[2/2 initial]")
    retry_phase = completed.stdout.index(expected_retry_message)
    first_retry = completed.stdout.index("[1/2 retry 1/1]")
    assert first_initial < second_initial < retry_phase < first_retry


def test_timeout_retry_runs_after_all_initial_candidates(tmp_path):
    completed, result_dir = run_retry_sweep(tmp_path)
    assert_retry_order(completed)

    rows = [json.loads(line) for line in (result_dir / "measurements.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    first, second = rows
    assert first["attempt_count"] == 2
    assert first["timeout_retries_used"] == 1
    assert first["initial_timed_out"] is True
    assert first["timed_out"] is False
    assert [attempt["timed_out"] for attempt in first["attempt_history"]] == [True, False]
    assert first["wall_time_s"] >= first["last_attempt_wall_time_s"]
    assert second["attempt_count"] == 1
    assert second["initial_timed_out"] is False

    first_log = (result_dir / "logs/d1-b1-m0.log").read_text()
    assert first_log.index("attempt 1/2") < first_log.index("attempt 2/2")
    assert "RETRY_TEST_INITIAL_TIMEOUT" in first_log
    assert "BENCHMARK operator=retry_test" in first_log


def test_simple_results_keep_one_row_for_retried_candidate(tmp_path):
    completed, result_dir = run_retry_sweep(tmp_path, simple_output=True)
    assert_retry_order(completed)

    with (result_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["尝试次数"] == "2"
    assert rows[0]["首轮是否超时"] == "True"
    assert rows[0]["最终是否超时"] == "False"
    assert rows[1]["尝试次数"] == "1"
    assert "结果=待重试" in completed.stdout
