import subprocess
import sys
import os

def test_benchmark_runs():
    """Check that the benchmark script runs without errors."""

    benchmark_path = os.path.join(
        os.path.dirname(__file__),
        "..", "ex", "benchmarks-deterministic", "PRoTECT-versions", "ex1_dt_DS.py"
    )

    assert os.path.exists(benchmark_path), f"Benchmark script not found: {benchmark_path}"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..")

    result = subprocess.run(
        [sys.executable, benchmark_path],
        capture_output=True,
        text=True,
        env=env
    )

    assert result.returncode == 0, f"Benchmark failed with error:\n{result.stderr}"
