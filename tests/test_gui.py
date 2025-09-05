import subprocess
import sys
import os

def test_main_runs():
    """Check that main.py launches the GUI and exits without errors."""

    main_path = os.path.join(os.path.dirname(__file__), "..", "main.py")
    assert os.path.exists(main_path), f"main.py not found at {main_path}"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..")
    env["AUTO_CLOSE_GUI"] = "1"  # auto-close GUI after 1 second

    result = subprocess.run(
        [sys.executable, main_path],
        capture_output=True,
        text=True,
        env=env
    )

    assert result.returncode == 0, f"main.py failed with error:\n{result.stderr}"