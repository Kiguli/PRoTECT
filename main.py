import logging
import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from src.models.barrier_tool_model import BarrierToolModel
from src.presenters.barrier_tool_presenter import BarrierToolPresenter
from src.views.barrier_tool_view import BarrierToolView
from src.utils.logging import setup_logging


def launch_app(auto_close=False):
    """
    Launch the Barrier Tool GUI.

    :param auto_close: If True, closes the GUI automatically after 1 second (for testing).
    :return: exit code of the QApplication
    """
    # Setup logging
    setup_logging(logging.INFO)

    app = QApplication(sys.argv)

    model = BarrierToolModel(parallel=True)
    view = BarrierToolView()
    presenter = BarrierToolPresenter(model, view)

    presenter.show()

    # Auto-close if testing
    if auto_close or os.getenv("AUTO_CLOSE_GUI") == "1":
        QTimer.singleShot(1000, app.quit)  # quit after 1 second

    return app.exec()


def main():
    exit_code = launch_app()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()