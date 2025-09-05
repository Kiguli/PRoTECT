import subprocess
import sys
import os

def test_main_runs():
    """Check that main.py runs without errors."""
    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "..", "main.py")],
        capture_output=True,
        text=True
    )
    # Assert that the script exits successfully
    assert result.returncode == 0, f"main.py failed with error:\n{result.stderr}"
