import logging
import copy

from PyQt6.QtCore import pyqtSignal, QObject, QThread

from src.functions.ct_DS import ct_DS
from src.functions.ct_SS import ct_SS
from src.functions.dt_DS import dt_DS
from src.functions.dt_SS import dt_SS
from src.functions.parallel_ct_DS import parallel_ct_DS
from src.functions.parallel_ct_SS import parallel_ct_SS
from src.functions.parallel_dt_DS import parallel_dt_DS
from src.functions.parallel_dt_SS import parallel_dt_SS
from src.utils.system_mode import SystemMode

logger = logging.getLogger(__name__)


class BarrierToolModel(QObject):
    """
    Barrier tool model is a class for performing necessary
    computations with the provided parameters.

    Attributes:
        _system_mode(SystemMode): Computation mode. Can be either of the enum values.
        _result(dict): Dictionary mapping the result parameters to their actual values.
            The parameters may vary based on the computation mode.
    """

    result_computed = pyqtSignal()

    def __init__(self, parallel: bool = None):
        super().__init__()
        self._system_mode = None
        self._parallel = parallel
        self._gateway_thread = None
        self._result = None

    class GatewayThread(QThread):
        def __init__(self, mode: SystemMode, parameters: dict, parallel: bool, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mode = copy.deepcopy(mode)
            self.parameters = copy.deepcopy(parameters)
            self._parallel = parallel
            self._result = None

        def run(self):
            self.__find_barrier()
            self.finished.emit()

        def __find_barrier(self):
            """
            Invokes the particular function, based on the provided parameters
            and updates the stored result. The function is based on the provided system_mode
            inside the dictionary.

            Note:
                Before calling the appropriate function validates the result and
                raises the Exception (or its subclass) as is intended in
                validate_computation_parameters method.
            """
            mode = self.mode
            parameters = self.parameters

            if self._parallel:
                if mode == SystemMode.DT_DS:
                    self._result = parallel_dt_DS(**parameters)
                elif mode == SystemMode.CT_DS:
                    self._result = parallel_ct_DS(**parameters)
                elif mode == SystemMode.DT_SS:
                    self._result = parallel_dt_SS(**parameters)
                elif mode == SystemMode.CT_SS:
                    self._result = parallel_ct_SS(**parameters)
                else:
                    raise NotImplementedError("Error: this mode is not implemented so far.")
            else:
                if mode == SystemMode.DT_DS:
                    self._result = dt_DS(**parameters)
                elif mode == SystemMode.CT_DS:
                    self._result = ct_DS(**parameters)
                elif mode == SystemMode.DT_SS:
                    self._result = dt_SS(**parameters)
                elif mode == SystemMode.CT_SS:
                    self._result = ct_SS(**parameters)
                else:
                    raise NotImplementedError("Error: unrecognized system mode.")

        def __del__(self):
            try:
                self.disconnect()
            except Exception as e:
                logger.debug(e)

        def retrieve_result(self) -> dict:
            return self._result

    def __update_result(self):
        self._result = self._gateway_thread.retrieve_result()
        self.result_computed.emit()

    def find_barrier(self, mode: SystemMode, parameters: dict, parallel: bool = None):
        if self._gateway_thread is not None:
            self.terminate_computing()

        if parallel is not None:
            self.set_parallel(parallel)

        self._gateway_thread = self.GatewayThread(mode, parameters, self._parallel)
        self._gateway_thread.finished.connect(self.__update_result)
        self._gateway_thread.start()

    def retrieve_result(self) -> dict:
        return self._result

    def terminate_computing(self):
        if self._gateway_thread is not None:
            try:
                self._gateway_thread.terminate()
            except Exception as e:
                logger.debug(e)

    def set_parallel(self, parallel: bool):
        self._parallel = parallel

    def __del__(self):
        self.terminate_computing()
