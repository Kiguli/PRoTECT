import subprocess
import sys
import os

def test_benchmark_runs():
    """Check that the benchmark script runs without errors."""

    # Path to the benchmark script (update if needed)
    benchmark_path = os.path.join(
        os.path.dirname(__file__),
        "..", "ex", "benchmarks-deterministic", "PRoTECT-versions", "ex1_dt_DS.py"
    )

    # Verify the file exists before running
    assert os.path.exists(benchmark_path), f"Benchmark script not found: {benchmark_path}"

    result = subprocess.run(
        [sys.executable, benchmark_path],
        capture_output=True,
        text=True
    )

    # Assert successful exit
    assert result.returncode == 0, f"Benchmark failed with error:\n{result.stderr}"
