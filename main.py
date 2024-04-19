import logging
import sys
from PyQt6.QtWidgets import QApplication

from src.models.barrier_tool_model import BarrierToolModel, BarrierToolParallelModel
from src.presenters.barrier_tool_presenter import BarrierToolPresenter
from src.views.barrier_tool_view import BarrierToolView

from src.utils.logging import setup_logging

if __name__ == "__main__":
    # Setup logging. Possible values: logging.DEBUG, logging.INFO:
    setup_logging(logging.INFO)

    app = QApplication(sys.argv)

    model = BarrierToolParallelModel()
    view = BarrierToolView()
    presenter = BarrierToolPresenter(model, view)

    presenter.show()
    sys.exit(app.exec())
