import os
import pytest

# Add the repository root to sys.path so we can import main.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import launch_app

# Ensure Qt runs in offscreen mode (headless)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["AUTO_CLOSE_GUI"] = "1"  # Auto-close GUI after 1 second

def test_main_launches():
    """Test that main.py GUI launches and exits cleanly."""
    exit_code = launch_app(auto_close=True)
    assert exit_code == 0
