from src.functions.ct_DS import ct_DS
from src.functions.ct_SS import ct_SS
from src.functions.dt_DS import dt_DS
from src.functions.dt_SS import dt_SS
from src.functions.parallel_ct_DS import parallel_ct_DS
from src.functions.parallel_ct_SS import parallel_ct_SS
from src.functions.parallel_dt_DS import parallel_dt_DS
from src.functions.parallel_dt_SS import parallel_dt_SS
from src.utils.system_mode import SystemMode


class BarrierToolModel:
    """
    Barrier tool model is a class for performing necessary
    computations with the provided parameters.

    Attributes:
        _system_mode(SystemMode): Computation mode. Can be either of the enum values.
        _result(dict): Dictionary mapping the result parameters to their actual values.
            The parameters may vary based on the computation mode.
    """
    def __init__(self):
        self._system_mode = None
        self._result = None

    def find_barrier(self, mode: SystemMode, parameters: dict):
        """
        Invokes the particular function, based on the provided parameters
        and updates the stored result. The function is based on the provided system_mode
        inside the dictionary.

        Note:
            Before calling the appropriate function validates the result and
            raises the Exception (or its subclass) as is intended in
            validate_computation_parameters method.
        """

        self.validate_computation_parameters(parameters)

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

    def retrieve_result(self) -> dict:
        return self._result

    @staticmethod
    def validate_computation_parameters(parameters: dict):
        """
        Validates the provided dictionary of parameters and raises exceptions in case


        Args:
            parameters(dict): dictionary, where the key is the name of the parameter,
            and it is mapped to the actual value.

        Raises:
            TypeError: If the provided value in the dictionary doesn't match the
                intended type of the key.
            ValueError: If value in the dictionary is inappropriate, considering
                the specified key.
        """
        # Currently the functions themselves raise the respective errors.
        pass


class BarrierToolParallelModel(BarrierToolModel):
    def find_barrier(self, mode: SystemMode, parameters: dict):
        """
        Invokes the particular (parallel) function, based on the provided parameters
        and updates the stored result. The function is based on the provided system_mode
        inside the dictionary.

        Note:
            Before calling the appropriate function validates the result and
            raises the Exception (or its subclass) as is intended in
            validate_computation_parameters method.
        """

        self.validate_computation_parameters(parameters)

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
