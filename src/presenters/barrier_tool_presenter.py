import logging
import json

import numpy as np
import sympy as sp

from src.models.barrier_tool_model import BarrierToolModel
from src.utils.common import get_np_array_from_string, get_expression_from_string
from src.utils.exceptions import BarrierNotFoundError, ExpressionFromStringError, RequiredParameterMissingError
from src.utils.noise_type import NoiseType
from src.views.barrier_tool_view import BarrierToolView
from src.utils.system_mode import SystemMode

logger = logging.getLogger(__name__)


class BarrierToolPresenter:
    def __init__(self, model: BarrierToolModel, view: BarrierToolView):
        self.model = model
        self.view = view

        # Connect signals from the view to presenter methods
        self.view.selected_discrete.connect(self.__update_system_mode)
        self.view.selected_continuous.connect(self.__update_system_mode)
        self.view.selected_deterministic.connect(self.__update_system_mode)
        self.view.selected_stochastic.connect(self.__update_system_mode)
        self.view.add_avoid_region_button_clicked.connect(self.view.add_avoid_region)
        self.view.delete_avoid_region_button_clicked.connect(self.__delete_avoid_region)
        self.view.find_barrier_button_clicked.connect(self.__find_barrier)
        self.view.number_of_dimensions_changed.connect(self.__update_dimensionality)
        self.view.parallelization_check_box_state_changed.connect(self.__set_parallel)
        self.view.import_config_button_clicked.connect(self.__import_config)
        self.view.export_config_button_clicked.connect(self.__export_config)

        # Connect signal from the model
        self.model.result_computed.connect(self.__update_results)

    def show(self):
        self.view.show()

    def __set_parallel(self, parallel: bool | int):
        parallel = (parallel is True or parallel > 0)
        self.model.set_parallel(parallel)
        if parallel:
            logger.info("Parallelization activated.")
        else:
            logger.info("Parallelization deactivated.")

    def __update_system_mode(self):
        mode = self.__retrieve_system_mode()
        self.view.update_system_mode(mode)
        logger.info(f"Set {mode.value} system mode.")
        self.__update_dimensionality()

    def __update_dimensionality(self):
        dimensionality = self.__retrieve_dimensionality()
        self.view.update_dimensionality(dimensionality)

    def __retrieve_system_mode(self) -> SystemMode:
        return self.view.retrieve_system_mode()

    def __retrieve_dimensionality(self) -> int:
        return self.view.retrieve_dimensionality()

    def __delete_avoid_region(self):
        deleted_row = self.view.delete_avoid_region()
        if deleted_row != -1:
            self.view.update_region_labels(deleted_row)

    def __import_config(self):
        path = self.view.import_config_dialog()
        if path is not None:
            try:
                config = self.__load_dict_from_json(path)
                self.__set_parameters(config)
            except Exception as e:
                print(e)
                self.view.show_warning("Error while loading the provided file. "
                                       "Make sure that all the parameters are "
                                       "provided correctly.")

    def __export_config(self):
        path = self.view.export_config_dialog()
        if path is not None:
            try:
                config = self.__retrieve_parameters()
                self.__save_dict_to_json(config, path)
            except:
                self.view.show_warning("Error while saving the parameters.")

    def __set_parameters(self, parameters: dict):
        self.view.update_system_mode(parameters['mode'])
        self.view.update_dimensionality(int(parameters['dim']))
        self.view.set_max_poly_degree(int(parameters['b_degree']))
        self.view.set_l_degree(parameters['l_degree'])
        self.view.set_solver(parameters['solver'])
        self.view.set_gamma(parameters['gam'])
        self.view.set_lambda(parameters['lam'])
        self.view.set_l_space(parameters['L_space'])
        self.view.set_u_space(parameters['U_space'])
        self.view.set_l_initial(parameters['L_initial'])
        self.view.set_u_initial(parameters['U_initial'])
        self.view.set_l_unsafe(parameters['L_unsafe'])
        self.view.set_u_unsafe(parameters['U_unsafe'])
        self.view.set_dynamics(parameters['f'])

        mode = self.__retrieve_system_mode()
        if mode.is_stochastic():
            self.view.set_c(parameters['c_val'])
            self.view.set_time_horizon(parameters['t'])
            self.view.set_min_confidence(float(parameters['confidence']))
            self.view.set_optimize(True if parameters['optimize'].strip().lower() == 'true' else False)

            if mode == SystemMode.DT_SS:
                self.view.set_noise_type(parameters['noise_type'])
                noise_type = self.view.retrieve_noise_type()
                if noise_type == NoiseType.normal:
                    self.view.set_mean(parameters['mean'])
                    self.view.set_sigma(parameters['sigma'])
                elif noise_type == NoiseType.exponential:
                    self.view.set_rate(parameters['rate'])
                elif noise_type == NoiseType.uniform:
                    self.view.set_a_and_b(parameters['a'], parameters['b'])
            elif mode == SystemMode.CT_SS:
                self.view.set_delta(parameters['delta'])
                self.view.set_rho(parameters['rho'])
                self.view.set_p_rate(parameters['p_rate'])

    def __retrieve_parameters(self) -> dict:
        """
        Retrieves computation parameters from the user interface.

        Returns:
            dict: computation parameters, retrieved from the view.
        """
        parameters = dict()
        mode = self.view.retrieve_system_mode()
        dimensionality = self.view.retrieve_dimensionality()

        # Generic parameters:
        parameters['mode'] = mode
        parameters['dim'] = str(dimensionality)
        parameters['b_degree'] = str(self.view.retrieve_max_poly_degree())
        parameters['l_degree'] = self.view.retrieve_l_degree()
        parameters['solver'] = self.view.retrieve_solver()
        parameters['gam'] = self.view.retrieve_gamma()
        parameters['lam'] = self.view.retrieve_lambda()

        # Region parameters:
        regions_dict = self.view.retrieve_regions()
        parameters['L_space'] = regions_dict['L_space']
        parameters['U_space'] = regions_dict['U_space']
        parameters['L_initial'] = regions_dict['L_initial']
        parameters['U_initial'] = regions_dict['U_initial']
        parameters['L_unsafe'] = regions_dict['L_unsafe']
        parameters['U_unsafe'] = regions_dict['U_unsafe']

        # Dynamics parameters:
        parameters['f'] = self.view.retrieve_dynamics()

        if mode.is_stochastic():
            # Common stochastic parameters:
            parameters['optimize'] = str(self.view.retrieve_optimize_checkbox_state()).lower().strip()
            parameters['c_val'] = self.view.retrieve_c()
            parameters['t'] = self.view.retrieve_time_horizon()
            parameters['confidence'] = str(self.view.retrieve_min_confidence())

            if mode == SystemMode.DT_SS:
                # Parameters regarding noise:
                noise_type = self.view.retrieve_noise_type()
                parameters['noise_type'] = noise_type

                if noise_type == 'normal':
                    parameters['mean'] = self.view.retrieve_mean()
                    parameters['sigma'] = self.view.retrieve_sigma()

                elif noise_type == 'exponential':
                    parameters['rate'] = self.view.retrieve_rate()

                elif noise_type == 'uniform':
                    a, b = self.view.retrieve_a_and_b()
                    parameters['a'] = a
                    parameters['b'] = b

            elif mode == SystemMode.CT_SS:
                parameters['delta'] = self.view.retrieve_brownian_motion()
                parameters['rho'] = self.view.retrieve_poisson_processes()
                parameters['p_rate'] = self.view.retrieve_p_rate()

        return parameters

    @staticmethod
    def __prepare_parameters_for_computation(mode: SystemMode, parameters: dict) -> dict:

        if parameters.get('mode') is not None:
            del parameters['mode']

        # Generic parameters:
        parameters['dim'] = int(parameters['dim'])
        parameters['b_degree'] = int(parameters['b_degree'])
        parameters['l_degree'] = int(parameters['l_degree']) if parameters['l_degree'] != '' else parameters['b_degree']
        parameters['gam'] = float(parameters['gam']) if parameters['gam'] != '' else None
        parameters['lam'] = float(parameters['lam']) if parameters['lam'] != '' else None

        # Region parameters:
        parameters['L_space'] = get_np_array_from_string(parameters['L_space'])
        parameters['U_space'] = get_np_array_from_string(parameters['U_space'])
        parameters['L_initial'] = get_np_array_from_string(parameters['L_initial'])
        parameters['U_initial'] = get_np_array_from_string(parameters['U_initial'])
        parameters['L_unsafe'] = [get_np_array_from_string(val) for val in parameters['L_unsafe']]
        parameters['U_unsafe'] = [get_np_array_from_string(val) for val in parameters['U_unsafe']]

        # Dynamics parameters:
        parameters['x'] = sp.symbols(f"x1:{parameters['dim'] + 1}")
        sp_vars = parameters['x']

        if mode == SystemMode.DT_SS:
            parameters['varsigma'] = sp.symbols(f"varsigma1:{parameters['dim'] + 1}")
            sp_vars += parameters['varsigma']

        dynamics_expressions = []
        for i in range(len(parameters['f'])):
            s = parameters['f'][i]
            expression = get_expression_from_string(s=s,
                                                    locals=sp_vars,
                                                    err_message=f"Please make sure to enter correct dynamics "
                                                                f"expression in the line {i + 1}!")
            dynamics_expressions.append(expression)
        parameters['f'] = np.array(dynamics_expressions)

        if mode.is_stochastic():
            parameters['optimize'] = True if parameters['optimize'].lower().strip() == 'true' else False
            parameters['confidence'] = float(parameters['confidence'])
            parameters['c_val'] = float(parameters['c_val']) if parameters['c_val'] != '' else None

            if parameters['t'] == '':
                raise RequiredParameterMissingError("Time horizon parameter is required.")
            parameters['t'] = float(parameters['t'])

            if parameters['optimize'] is True and parameters['lam'] is None:
                print("Here")
                raise RequiredParameterMissingError("λ parameter is required for optimization.")

            if mode == SystemMode.DT_SS:
                if parameters['noise_type'] == 'normal':
                    parameters['mean'] = get_np_array_from_string(parameters['mean']) \
                        if parameters['mean'] != "" else np.zeros(parameters['dim'], dtype=np.double)

                    if parameters['sigma'] == "":
                        raise RequiredParameterMissingError("σ parameter is required.")

                    parameters['sigma'] = get_np_array_from_string(parameters['sigma'])

                elif parameters['noise_type'] == 'exponential':
                    if parameters['rate'] == "":
                        raise RequiredParameterMissingError("Rate is a required parameter.")

                    parameters['rate'] = get_np_array_from_string(parameters['rate'])

                elif parameters['noise_type'] == 'uniform':

                    if parameters['a'] == "" or parameters['b'] == "":
                        raise RequiredParameterMissingError("a and b are required parameters.")

                    parameters['a'] = get_np_array_from_string(parameters['a'])
                    parameters['b'] = get_np_array_from_string(parameters['b'])

            elif mode == SystemMode.CT_SS:
                delta_expressions = []
                for i in range(len(parameters['delta'])):
                    s = parameters['delta'][i]
                    delta_expressions.append(get_expression_from_string(s=s,
                                                                        locals=parameters['x'],
                                                                        err_message=f"Please make sure to enter the "
                                                                                    f"correct δ in the line {i + 1}!")
                                             if s != "" else 0)
                parameters['delta'] = np.array(delta_expressions)

                rho_expressions = []
                for i in range(len(parameters['rho'])):
                    s = parameters['rho'][i]
                    rho_expressions.append(get_expression_from_string(s=s,
                                                                      locals=parameters['x'],
                                                                      err_message=f"Please make sure to enter the "
                                                                                  f"correct ρ in the line {i + 1}!")

                                           if s != "" else 0)
                parameters['rho'] = np.array(rho_expressions)

                parameters['p_rate'] = get_np_array_from_string(parameters['p_rate']) \
                    if parameters['p_rate'] != "" else np.zeros(parameters['dim'], dtype=np.double)

        return parameters

    def __find_barrier(self):
        try:
            mode = self.__retrieve_system_mode()
            parameters = self.__retrieve_parameters()
            parameters = self.__prepare_parameters_for_computation(mode, parameters)
            logger.info("Computation parameters:\n" + str(parameters))
            self.model.find_barrier(mode, parameters)

        except BarrierNotFoundError:
            self.view.show_warning("Barrier not found for given parameters.")

        except ExpressionFromStringError as e:
            self.view.show_warning(str(e))

        except RequiredParameterMissingError as e:
            self.view.show_warning(str(e))

        except Exception as e:
            print(e)
            self.view.show_parameters_warning()
            return

    def __update_results(self):
        result: dict = self.model.retrieve_result()
        if result is not None:
            if result.get('barrier') is not None:
                logger.info("Results:\n" + str(result))
                self.view.update_result(result)
            elif result.get('error'):
                self.view.show_warning(f"Barrier not found: {result['error']}")
            else:
                self.view.show_warning("Unexpected error occurred!")
                print(result)

    @staticmethod
    def __save_dict_to_json(dictionary, file_path):
        with open(file_path, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)

    @staticmethod
    def __load_dict_from_json(file_path) -> dict:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
