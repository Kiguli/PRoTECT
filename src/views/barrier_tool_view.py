import logging
import os

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import QMainWindow, QTableWidget, QHeaderView, QLineEdit, QTableWidgetItem, \
    QSizePolicy, QLabel, QFormLayout, QDoubleSpinBox, QMessageBox, QLayout, QComboBox, QCheckBox, QHBoxLayout, \
    QWidget, QFileDialog
from PyQt6 import uic
from brian2.parsing.sympytools import sympy_to_str

from src.utils.noise_type import NoiseType
from src.utils.system_mode import SystemMode

barrier_tool_path = os.path.join("src", "views", "barrier_tool.ui")
logger = logging.getLogger(__name__)


class BarrierToolView(QMainWindow):
    # System mode signals:
    selected_discrete = pyqtSignal()
    selected_continuous = pyqtSignal()
    selected_deterministic = pyqtSignal()
    selected_stochastic = pyqtSignal()

    # Button signals:
    add_avoid_region_button_clicked = pyqtSignal()
    delete_avoid_region_button_clicked = pyqtSignal()
    find_barrier_button_clicked = pyqtSignal()
    import_config_button_clicked = pyqtSignal()
    export_config_button_clicked = pyqtSignal()

    # Line edit signals:
    number_of_dimensions_changed = pyqtSignal()
    max_b_degree_changed = pyqtSignal()

    # Check-box signals:
    optimize_check_box_state_changed = pyqtSignal(int)
    parallelization_check_box_state_changed = pyqtSignal(int)

    def __init__(self, *args, **kwargs):

        super(BarrierToolView, self).__init__(*args, **kwargs)

        # Fixed visual parameters:
        self.__min_table_horizontal_header_width = 55
        self.__min_right_hand_size_label_width = 95
        self.__parameter_config_form_size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.__time_horizon_line_edit_placeholder_text = "(Required)"
        self.__noise_parameters_row_number = 5
        self.__min_confidence_spin_box_configuration = {"maximum": 1.0, "decimals": 2, "singleStep": 0.1}
        self.__uniform_noise_parameters_layout_configuration = {"spacing": 7}
        self.__c_line_edit_configuration = {"minimumWidth": 70, "placeholderText": "(Optional)"}
        self.__generic_min_spinbox_value = 0
        self.__generic_max_spinbox_value = 2147483647
        self.__possible_existing_noise_parameters = ["sigmaTextLabel", "sigmaLineEdit",
                                                     "rateTextLabel", "rateLineEdit",
                                                     "uniformNoiseParametersLayout",
                                                     "meanTextLabel", "meanLineEdit"]
        self.__region_table_placeholder_text = {(0, 0): "a1, a2, ... an",
                                                (0, 1): "b1, b2, ... bn"}

        """Usage: {"form_layout_name": {"component_name" : componentWidget}}"""
        self.optional_components = {"configurationValuesForm": {},
                                    "mainParameterConfigForm": {},
                                    "confidenceLayout": {}}

        self.ui = self.init_ui()

    def init_ui(self):
        ui = uic.loadUi(barrier_tool_path, self)
        self.__format_existing_tables()
        self.__init_signals()

        # Update the ui-components depending on the system-mode:
        self.update_system_mode(self.retrieve_system_mode())

        return ui

    def update_system_mode(self, mode):
        self.__remove_optional_components()

        if mode == SystemMode.DT_DS:
            self.__set_discrete_deterministic_view()
        elif mode == SystemMode.DT_SS:
            self.__set_discrete_stochastic_view()
        elif mode == SystemMode.CT_DS:
            self.__set_continuous_deterministic_view()
        elif mode == SystemMode.CT_SS:
            self.__set_continuous_stochastic_view()
        else:
            raise ValueError(f"System mode {mode} is not supported.")

    """Attempts to add an avoid region to the regionTable. Returns whether the adding result was successful."""

    def add_avoid_region(self):
        row_count = self.regionTable.rowCount()
        self.regionTable.setRowCount(row_count + 1)

        label = f'Xu_{row_count - 1} :'
        item = QTableWidgetItem(label)
        self.regionTable.setVerticalHeaderItem(row_count, item)

    """Attempts to delete an avoid region from the regionTable. Returns whether the attempt was successful."""

    def delete_avoid_region(self) -> int:
        # Consider row count:
        row_num = self.regionTable.rowCount() - 1
        # State space, initial region and at least on avoid region:
        if row_num < 3:
            self.show_region_warning()
            return -1

        # Get selected indexes:
        selected_indexes = self.regionTable.selectedIndexes()

        # Check if anything was selected:
        if len(selected_indexes) == 0:
            self.regionTable.removeRow(row_num)
            return row_num
        else:
            # Get selected row:
            selected_row = self.regionTable.currentRow()

            # Make the rows 0 and 1 immutable:
            if 0 <= selected_row <= 1:
                self.show_region_warning()
                return -1

            # Delete the selected row
            if selected_row != -1:
                self.regionTable.removeRow(selected_row)
                return selected_row

    def retrieve_system_mode(self) -> SystemMode:
        result_mode_str = str()
        result_mode_str += "continuous" if self.radioButtonContinuous.isChecked() else "discrete"
        result_mode_str += '_'
        result_mode_str += "stochastic" if self.radioButtonStochastic.isChecked() else "deterministic"

        for mode in SystemMode:
            if mode.value == result_mode_str:
                return mode

    def retrieve_dimensionality(self) -> int:
        try:
            dimensionality = int(self.numberOfDimensionsSpinBox.value())
            return dimensionality
        except Exception as e:
            logger.debug(e)
            self.show_warning(str(e))
            return 0

    def retrieve_max_poly_degree(self):
        return self.maxPolynomialDegreeSpinBox.value()

    def retrieve_l_degree(self):
        return self.lDegreeLineEdit.text().strip()

    def retrieve_parallelization_checkbox_state(self) -> bool:
        return self.parallelizationCheckBox.isChecked()

    def retrieve_regions(self) -> dict:

        row_count = self.regionTable.rowCount()
        assert (row_count >= 3)

        """Returns text if something was entered and None otherwise."""
        get_cell_value = lambda x, y: \
            self.regionTable.item(x, y).text().strip() if self.regionTable.item(x, y) is not None else ""

        regions = dict()
        regions['L_space'] = "" if get_cell_value(0, 0) == self.__region_table_placeholder_text[(0, 0)] \
            else get_cell_value(0, 0)
        regions['U_space'] = "" if get_cell_value(0, 1) == self.__region_table_placeholder_text[(0, 1)] \
            else get_cell_value(0, 1)
        regions['L_initial'] = get_cell_value(1, 0)
        regions['U_initial'] = get_cell_value(1, 1)
        regions['L_unsafe'] = [get_cell_value(row, 0) for row in range(2, row_count)]
        regions['U_unsafe'] = [get_cell_value(row, 1) for row in range(2, row_count)]

        return regions

    def retrieve_number_of_avoid_regions(self) -> int:
        return self.regionTable.rowCount() - 2

    def retrieve_dynamics(self) -> list[str]:

        """Gets a list of rows' text if it was entered and returns None otherwise."""
        get_row_value = lambda r: \
            self.dynamicsTable.item(r, 0).text().strip().lower() if self.dynamicsTable.item(r, 0) is not None else ""

        return [get_row_value(r) for r in range(0, self.dynamicsTable.rowCount())]

    def retrieve_brownian_motion(self) -> list[str]:

        """Gets a list of rows' text if it was entered and returns None otherwise."""
        get_row_value = lambda r: \
            self.optional_components["configurationValuesForm"]["gTable"].item(r, 0).text().strip() \
                if self.optional_components["configurationValuesForm"]["gTable"] is not None else None

        return [get_row_value(r)
                for r in range(0, self.optional_components["configurationValuesForm"]["gTable"].rowCount())]

    def retrieve_poisson_processes(self) -> list[str]:
        """Gets a list of rows' text if it was entered and returns None otherwise."""
        get_row_value = lambda r: \
            self.optional_components["configurationValuesForm"]["rTable"].item(r, 0).text().strip() \
                if self.optional_components["configurationValuesForm"]["rTable"] is not None else None

        return [get_row_value(r)
                for r in range(0, self.optional_components["configurationValuesForm"]["rTable"].rowCount())]

    def retrieve_solver(self) -> str:
        return self.solverComboBox.currentText().strip().lower()

    def retrieve_gamma(self):
        return self.gammaLineEdit.text().strip()

    def retrieve_lambda(self):
        return self.lambdaLineEdit.text().strip()

    def retrieve_c(self):
        return self.optional_components["mainParameterConfigForm"]["cLineEdit"].text().strip()

    def retrieve_time_horizon(self):
        return self.optional_components["mainParameterConfigForm"]["timeHorizonLineEdit"].text().strip()

    def retrieve_min_confidence(self):
        return self.optional_components["mainParameterConfigForm"]["minConfidenceSpinBox"].value()

    def retrieve_optimize_checkbox_state(self) -> bool:
        checkbox = self.optional_components["mainParameterConfigForm"].get("optimizeCheckBox")

        if checkbox is None:
            return False
        try:
            return checkbox.isChecked()
        except:
            return False

    def retrieve_noise_type(self):
        return self.optional_components["mainParameterConfigForm"]["noiseTypeComboBox"].currentText().strip().lower()

    def retrieve_mean(self):
        return self.optional_components["mainParameterConfigForm"]["meanLineEdit"].text().strip()

    def retrieve_sigma(self):
        return self.optional_components["mainParameterConfigForm"]["sigmaLineEdit"].text().strip()

    def retrieve_rate(self):
        return self.optional_components["mainParameterConfigForm"]["rateLineEdit"].text().strip()

    def retrieve_a_and_b(self):
        layout = self.optional_components["mainParameterConfigForm"]["uniformNoiseParametersLayout"]
        a = layout.itemAt(2).widget().text().strip()
        b = layout.itemAt(4).widget().text().strip()

        return a, b

    def retrieve_p_rate(self):
        return self.optional_components["configurationValuesForm"]["pRateLineEdit"].text().strip()

    def set_max_poly_degree(self, value: int):
        self.maxPolynomialDegreeSpinBox.setValue(value)

    def set_l_degree(self, text: str):
        self.lDegreeLineEdit.setText(text.strip())

    def set_solver(self, text: str):
        solver = text.strip().lower()
        if solver == 'mosek':
            self.solverComboBox.setCurrentIndex(0)
        elif solver == 'cvxopt':
            self.solverComboBox.setCurrentIndex(1)

    def set_gamma(self, text: str):
        gamma = text.strip().lower()
        self.gammaLineEdit.setText(gamma)

    def set_lambda(self, text: str):
        lam = text.strip().lower()
        self.lambdaLineEdit.setText(lam)

    def set_l_space(self, text: str):
        l_space = text.strip().lower()
        self.__set_region_item_text(0, 0, l_space)

    def set_u_space(self, text: str):
        u_space = text.strip().lower()
        self.__set_region_item_text(0, 1, u_space)

    def set_l_initial(self, text: str):
        l_initial = text.strip().lower()
        self.__set_region_item_text(1, 0, l_initial)

    def set_u_initial(self, text: str):
        u_initial = text.strip().lower()
        self.__set_region_item_text(1, 1, u_initial)

    def set_l_unsafe(self, bounds: list[str]):
        length = len(bounds)
        while self.retrieve_number_of_avoid_regions() < length:
            self.add_avoid_region()
        for i in range(length):
            bound = bounds[i].strip().lower()
            row, col = i + 2, 0
            self.__set_region_item_text(row, col, bound)

    def set_u_unsafe(self, bounds: list[str]):
        length = len(bounds)
        while self.retrieve_number_of_avoid_regions() < length:
            self.add_avoid_region()
        for i in range(length):
            bound = bounds[i].strip().lower()
            row, col = i + 2, 1
            self.__set_region_item_text(row, col, bound)

    def set_dynamics(self, dynamics: list[str]):
        for i in range(len(dynamics)):
            dyn = dynamics[i].strip().lower()
            self.__set_dynamics_item_text(i, 0, dyn)

    def set_delta(self, delta: list[str]):
        for i in range(len(delta)):
            d = delta[i].strip().lower()
            self.__set_g_item_text(i, 0, d)

    def set_rho(self, rho: list[str]):
        for i in range(len(rho)):
            r = rho[i].strip().lower()
            self.__set_r_item_text(i, 0, r)

    def set_c(self, text):
        if self.optional_components["mainParameterConfigForm"].get('cLineEdit') is not None:
            self.optional_components["mainParameterConfigForm"]["cLineEdit"].setText(text.strip().lower())

    def set_time_horizon(self, text):
        if self.optional_components["mainParameterConfigForm"].get('timeHorizonLineEdit') is not None:
            self.optional_components["mainParameterConfigForm"]["timeHorizonLineEdit"].setText(text.strip().lower())

    def set_min_confidence(self, value: float):
        if value < 0 or value > 1:
            raise ValueError("Min. confidence value exceeds the boundaries.")
        if self.optional_components["mainParameterConfigForm"].get('minConfidenceSpinBox') is not None:
            self.optional_components["mainParameterConfigForm"]["minConfidenceSpinBox"].setValue(value)

    def set_noise_type(self, noise_type: str):
        noise_type = noise_type.strip().lower()

        if noise_type in [n for n in NoiseType]:
            self.__set_noise_parameters(noise_type)
        else:
            raise ValueError("Unsupported noise type.")

    def set_mean(self, text: str):
        if self.optional_components["mainParameterConfigForm"].get("meanLineEdit"):
            self.optional_components["mainParameterConfigForm"]["meanLineEdit"].setText(text.strip().lower())

    def set_sigma(self, text: str):
        if self.optional_components["mainParameterConfigForm"].get("sigmaLineEdit"):
            self.optional_components["mainParameterConfigForm"]["sigmaLineEdit"].setText(text.strip().lower())

    def set_rate(self, text: str):
        if self.optional_components["mainParameterConfigForm"].get("rateLineEdit"):
            self.optional_components["mainParameterConfigForm"]["rateLineEdit"].setText(text.strip().lower())

    def set_a_and_b(self, a: str, b: str):
        a, b = a.strip().lower(), b.strip().lower()
        layout = self.optional_components["mainParameterConfigForm"]["uniformNoiseParametersLayout"]
        layout.itemAt(2).widget().setText(a)
        layout.itemAt(4).widget().setText(b)

    def set_p_rate(self, text: str):
        if self.optional_components["configurationValuesForm"].get("pRateLineEdit"):
            self.optional_components["configurationValuesForm"]["pRateLineEdit"].setText(text.strip().lower())

    def set_optimize(self, checked: bool):
        checkbox = self.optional_components["mainParameterConfigForm"].get("optimizeCheckBox")
        if checkbox is not None:
            if checked:
                checkbox.setCheckState(Qt.CheckState.Checked)
            else:
                checkbox.setCheckState(Qt.CheckState.Unchecked)

    def update_dimensionality(self, dimensionality: int):
        dimensionality = int(dimensionality)

        # Get current system mode:
        current_system_mode = self.retrieve_system_mode()

        # Update dimensionality spin box:
        if self.retrieve_dimensionality() != dimensionality:
            self.numberOfDimensionsSpinBox.setValue(dimensionality)

        # Update dynamics view:
        self.__update_dynamics_dimensionality(dimensionality)

        # Update continuous stochastic view
        if current_system_mode == SystemMode.CT_SS:
            self.__update_r_table_dimensionality(dimensionality)
            self.__update_g_table_dimensionality(dimensionality)

    def update_result(self, result: dict):
        mode = self.retrieve_system_mode()

        if result.get('error') is None:
            # TODO: add set <parameter> function and call update result from presenter
            self.barrierTextEdit.setText(sympy_to_str(result["barrier"]))
            self.lambdaLineEdit.setText(str(result["lambda"]))
            self.gammaLineEdit.setText(str(result["gamma"]))

            if mode.is_stochastic():
                self.optional_components["mainParameterConfigForm"]["cLineEdit"].setText(str(result["c"]))
                self.optional_components["confidenceLayout"]["confidenceLineEdit"].setText(str(result["confidence"]))

    def update_region_labels(self, last_row):
        for row in range(last_row, self.regionTable.rowCount()):
            label = f'Xu_{row - 1} :'
            item = QTableWidgetItem(label)
            self.regionTable.setVerticalHeaderItem(row, item)

    def export_config_dialog(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Folder to Save Config")

        if file_path:
            logger.info("Selected file: " + str(file_path))
            return file_path

        return None

    def import_config_dialog(self) -> str | None:
        options = QFileDialog.Option.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File to Open", "", "All Files (*)", options=options)

        if file_path:
            logger.info("Selected file: " + str(file_path))
            return file_path

        return None

    def show_region_warning(self):
        self.show_warning("State space, initial region and the first avoid region can not be deleted.")

    def show_parameters_warning(self):
        self.show_warning("Please, make sure that all the parameters are entered correctly.")

    def show_warning(self, warning_text: str):
        QMessageBox.warning(self, "Warning", warning_text)

    def __format_existing_tables(self):
        # Formatting region table resize-ability:
        region_table = self.findChild(QTableWidget, "regionTable")
        region_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        region_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        # Formatting dynamics table resize-ability:
        dynamics_table = self.findChild(QTableWidget, "dynamicsTable")
        dynamics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        dynamics_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        # Resize table headers
        region_table.verticalHeader().setMinimumSize(self.__min_table_horizontal_header_width, 0)
        dynamics_table.verticalHeader().setMinimumSize(self.__min_table_horizontal_header_width, 0)

    def __init_signals(self):
        # Buttons:
        self.add_avoid_region_button_clicked = self.addRegionButton.clicked
        self.delete_avoid_region_button_clicked = self.deleteRegionButton.clicked
        self.find_barrier_button_clicked = self.findBarrierButton.clicked
        self.import_config_button_clicked = self.importConfigButton.clicked
        self.export_config_button_clicked = self.exportConfigButton.clicked

        # System mode radio buttons:
        self.selected_discrete = self.radioButtonDiscrete.clicked
        self.selected_continuous = self.radioButtonContinuous.clicked
        self.selected_deterministic = self.radioButtonDeterministic.clicked
        self.selected_stochastic = self.radioButtonStochastic.clicked

        # Spin boxes edits:
        self.number_of_dimensions_changed = self.numberOfDimensionsSpinBox.textChanged[str]

        # Check-box signals
        self.parallelization_check_box_state_changed = self.parallelizationCheckBox.stateChanged[int]

        # Tables
        self.regionTable.cellChanged.connect(self.__handle_region_table_cell_changed)

    def __init_optimize_text_label(self):
        optimize_text_label = QLabel()
        optimize_text_label.setText("Optimize :")
        self.optional_components["mainParameterConfigForm"]["optimizeTextLabel"] = optimize_text_label

    def __init_optimize_check_box(self):
        optimize_check_box = QCheckBox()
        self.optimize_check_box_state_changed = optimize_check_box.stateChanged[int]
        self.optimize_check_box_state_changed.connect(self.__update_optimization_parameters_state)
        self.optional_components["mainParameterConfigForm"]["optimizeCheckBox"] = optimize_check_box

    def __init_r_table(self):
        # Instantiate the QTableWidget
        rTable = QTableWidget()

        # Set size policy
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        rTable.setSizePolicy(size_policy)

        # Hide horizontal header
        rTable.horizontalHeader().setVisible(False)

        # Set row and column count
        rTable.setRowCount(2)  # 2 rows
        rTable.setColumnCount(1)  # 1 column

        # Set text for cells
        rTable.setItem(0, 0, QTableWidgetItem("0"))
        rTable.setVerticalHeaderLabels(["ρ(x0) ="])

        # Resize headers
        rTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        rTable.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        rTable.verticalHeader().setMinimumSize(self.__min_table_horizontal_header_width, 0)

        self.optional_components["configurationValuesForm"]["rTable"] = rTable

    def __init_r_text_label(self):
        r_text_label = QLabel()
        r_text_label.setText(
            "<font>Poisson ρ :</font><br>"
            "<font color='#808080'>(Default: 0)</font>")
        self.optional_components["configurationValuesForm"]["rTextLabel"] = r_text_label

    def __init_g_table(self):
        # Instantiate the QTableWidget
        gTable = QTableWidget()

        # Set size policy
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        gTable.setSizePolicy(size_policy)

        # Hide horizontal header
        gTable.horizontalHeader().setVisible(False)

        # Set row and column count
        gTable.setRowCount(2)  # 2 rows
        gTable.setColumnCount(1)  # 1 column

        # Set text for cells
        gTable.setItem(0, 0, QTableWidgetItem("0"))
        gTable.setVerticalHeaderLabels(["δ(x0) ="])

        # Resize headers
        gTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        gTable.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        gTable.verticalHeader().setMinimumSize(self.__min_table_horizontal_header_width, 0)

        self.optional_components["configurationValuesForm"]["gTable"] = gTable

    def __init_g_text_label(self):
        g_text_label = QLabel()
        g_text_label.setText(
            "<font>Brownian δ :</font><br>"
            "<font color='#808080'>(Default: 0)</font>")
        self.optional_components["configurationValuesForm"]["gTextLabel"] = g_text_label

    def __init_min_confidence_spin_box(self):
        min_confidence_spin_box = QDoubleSpinBox()
        min_confidence_spin_box.setSizePolicy(self.__parameter_config_form_size_policy)

        params = self.__min_confidence_spin_box_configuration

        min_confidence_spin_box.setMaximum(params["maximum"])
        min_confidence_spin_box.setDecimals(params["decimals"])
        min_confidence_spin_box.setSingleStep(params["singleStep"])

        self.optional_components["mainParameterConfigForm"]["minConfidenceSpinBox"] = min_confidence_spin_box

    def __init_min_confidence_text_label(self):
        min_confidence_text_label = QLabel()
        min_confidence_text_label.setText("Min. confidence:")
        self.optional_components["mainParameterConfigForm"]["minConfidenceTextLabel"] = min_confidence_text_label

    def __init_c_text_label(self):
        c_text_label = QLabel()
        c_text_label.setText("Constant c :")
        self.optional_components["mainParameterConfigForm"]["cTextLabel"] = c_text_label

    def __init_c_line_edit(self):
        params = self.__c_line_edit_configuration

        c_line_edit = QLineEdit()
        c_line_edit.setMinimumSize(params["minimumWidth"], 0)
        c_line_edit.setPlaceholderText(params["placeholderText"])
        c_line_edit.setSizePolicy(self.__parameter_config_form_size_policy)
        c_line_edit.setAlignment(
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeading | Qt.AlignmentFlag.AlignLeft)
        self.optional_components["mainParameterConfigForm"]["cLineEdit"] = c_line_edit

    def __init_confidence_text_label(self):
        confidence_text_label = QLabel()
        confidence_text_label.setText("Confidence φ :")
        confidence_text_label.setMinimumSize(self.__min_right_hand_size_label_width, 0)
        confidence_text_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.optional_components["confidenceLayout"]["confidenceTextLabel"] = confidence_text_label

    def __init_confidence_line_edit(self):
        confidence_line_edit = QLineEdit()
        confidence_line_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        confidence_line_edit.setPlaceholderText("Confidence will appear here ...")
        self.optional_components["confidenceLayout"]["confidenceLineEdit"] = confidence_line_edit

    def __init_time_horizon_text_label(self):
        time_horizon_text_label = QLabel()
        time_horizon_text_label.setText("Time horizon Τ :")
        self.optional_components["mainParameterConfigForm"]["timeHorizonTextLabel"] = time_horizon_text_label

    def __init_time_horizon_line_edit(self):
        time_horizon_line_edit = QLineEdit()
        time_horizon_line_edit.setPlaceholderText(self.__time_horizon_line_edit_placeholder_text)
        time_horizon_line_edit.setSizePolicy(self.__parameter_config_form_size_policy)
        self.optional_components["mainParameterConfigForm"]["timeHorizonLineEdit"] = time_horizon_line_edit

    def __init_noise_type_text_label(self):
        noise_type_text_label = QLabel()
        noise_type_text_label.setText("Noise type :")
        self.optional_components["mainParameterConfigForm"]["noiseTypeTextLabel"] = noise_type_text_label

    def __init_noise_type_combo_box(self):
        noise_type_combo_box = QComboBox()
        noise_type_combo_box.addItems([t.value for t in NoiseType])
        noise_type_combo_box.setSizePolicy(self.__parameter_config_form_size_policy)
        noise_type_combo_box.currentTextChanged.connect(self.__set_noise_parameters)
        self.optional_components["mainParameterConfigForm"]["noiseTypeComboBox"] = noise_type_combo_box

    def __init_p_rate_text_label(self):
        p_rate_text_label = QLabel()
        p_rate_text_label.setText("Poisson process rate ω :")
        self.optional_components["configurationValuesForm"]["pRateTextLabel"] = p_rate_text_label

    def __init_p_rate_line_edit(self):
        p_rate_line_edit = QLineEdit()
        p_rate_line_edit.setPlaceholderText("(Default: 0) ω1, ω2, ... ω3")
        self.optional_components["configurationValuesForm"]["pRateLineEdit"] = p_rate_line_edit

    def __init_mean_text_label(self):
        mean_text_label = QLabel()
        mean_text_label.setText("Mean μ :")
        self.optional_components["mainParameterConfigForm"]["meanTextLabel"] = mean_text_label

    def __init_mean_line_edit(self):
        mean_line_edit = QLineEdit()
        mean_line_edit.setPlaceholderText("(Default: 0) μ1, μ2, ... μn")
        mean_line_edit.setSizePolicy(self.__parameter_config_form_size_policy)
        self.optional_components["mainParameterConfigForm"]["meanLineEdit"] = mean_line_edit

    def __init_sigma_text_label(self):
        sigma_text_label = QLabel()
        sigma_text_label.setText("Sigma σ :")
        self.optional_components["mainParameterConfigForm"]["sigmaTextLabel"] = sigma_text_label

    def __init_sigma_line_edit(self):
        sigma_line_edit = QLineEdit()
        sigma_line_edit.setPlaceholderText("(Required) σ1, σ2, ... σn")
        sigma_line_edit.setSizePolicy(self.__parameter_config_form_size_policy)
        self.optional_components["mainParameterConfigForm"]["sigmaLineEdit"] = sigma_line_edit

    def __init_rate_text_label(self):
        rate_text_label = QLabel()
        rate_text_label.setText("Rate :")
        self.optional_components["mainParameterConfigForm"]["rateTextLabel"] = rate_text_label

    def __init_rate_line_edit(self):
        rate_line_edit = QLineEdit()
        rate_line_edit.setPlaceholderText("(Required) r1, r2, ... rn")
        rate_line_edit.setSizePolicy(self.__parameter_config_form_size_policy)
        self.optional_components["mainParameterConfigForm"]["rateLineEdit"] = rate_line_edit

    def __init_uniform_noise_parameters_layout(self):
        params = self.__uniform_noise_parameters_layout_configuration

        uniform_noise_parameters_layout = QHBoxLayout()
        uniform_noise_parameters_layout.setSpacing(params["spacing"])
        uniform_noise_parameters_layout.setAlignment(
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeading | Qt.AlignmentFlag.AlignLeft)

        horizontal_spacer_label = QLabel()
        horizontal_spacer_label.setMinimumSize(self.__min_right_hand_size_label_width, 0)

        uniform_noise_parameters_layout.addWidget(horizontal_spacer_label)

        a_text_label = QLabel(text="a :")
        a_text_label.setObjectName("aTextLabel")
        uniform_noise_parameters_layout.addWidget(a_text_label)
        a_line_edit = QLineEdit()
        a_line_edit.setPlaceholderText("a1, a2, ... an")
        a_line_edit.setObjectName("aLineEdit")
        uniform_noise_parameters_layout.addWidget(a_line_edit)
        b_text_label = QLabel(text="b :")
        b_text_label.setObjectName("bTextLabel")
        uniform_noise_parameters_layout.addWidget(b_text_label)
        b_line_edit = QLineEdit()
        b_line_edit.setPlaceholderText("b1, b2, ... bn")
        b_line_edit.setObjectName("bLineEdit")
        uniform_noise_parameters_layout.addWidget(b_line_edit)

        self.optional_components["mainParameterConfigForm"]["uniformNoiseParametersLayout"] = \
            uniform_noise_parameters_layout

    def __handle_region_table_cell_changed(self, row, col):
        self.regionTable.blockSignals(True)

        item = self.regionTable.item(row, col)
        if item is not None:

            if not item.text():
                if (row, col) == (0, 0) or (row, col) == (0, 1):
                    self.__set_region_table_item_placeholder_text(row, col,
                                                                  self.__region_table_placeholder_text[(row, col)])
            else:
                self.__set_region_table_cell_color(row, col, QColor())

        self.regionTable.blockSignals(False)

    def __set_region_table_item_placeholder_text(self, row, col, text):
        item = self.regionTable.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            self.regionTable.setItem(row, col, item)
        color = QColor("#808080")
        self.__set_region_table_cell_color(row, col, color)
        item.setText(text)

    def __set_region_table_cell_color(self, row, col, color):
        item = self.regionTable.item(row, col)
        if item is not None:
            brush = QBrush(color)
            brush.setStyle(Qt.BrushStyle.SolidPattern)
            item.setForeground(brush)

    def __remove_optional_components(self):
        for form_name, component_list in self.optional_components.items():
            form = self.findChild(QLayout, form_name)
            for v in component_list.values():
                try:
                    if isinstance(form, QFormLayout):
                        form.removeRow(v)
                    else:
                        # Optional substitution: v.setParent(None)
                        v.deleteLater()
                except Exception as e:
                    logger.debug(e)

    def __update_optimization_parameters_state(self, optimize_state):
        optimize = (optimize_state == Qt.CheckState.Checked.value or optimize_state is True)

        self.gammaLineEdit.setReadOnly(optimize)
        if optimize:
            self.gammaLineEdit.setText("")
            self.gammaLineEdit.setPlaceholderText("γ will appear here ...")

            self.lambdaLineEdit.setPlaceholderText("(Required)")
        else:
            self.gammaLineEdit.setPlaceholderText("(Optional)")

        system_mode = self.retrieve_system_mode()
        if system_mode == SystemMode.DT_SS or system_mode == SystemMode.CT_SS:
            self.optional_components["mainParameterConfigForm"]["cLineEdit"].setReadOnly(optimize)
            if optimize:
                self.optional_components["mainParameterConfigForm"]["cLineEdit"].setText("")
                self.optional_components["mainParameterConfigForm"]["cLineEdit"].\
                    setPlaceholderText("c will appear here ...")
            else:
                self.optional_components["mainParameterConfigForm"]["cLineEdit"].setPlaceholderText(
                    self.__c_line_edit_configuration["placeholderText"])
                self.lambdaLineEdit.setPlaceholderText("(Optional)")

    def __update_dynamics_labels(self, dimensionality):
        for row in range(dimensionality):
            label = f'f(x{row + 1}) ='
            item = QTableWidgetItem(label)
            self.dynamicsTable.setVerticalHeaderItem(row, item)

    def __update_r_table_labels(self, dimensionality):
        for row in range(dimensionality):
            label = f'ρ(x{row + 1}) ='
            item = QTableWidgetItem(label)
            self.optional_components["configurationValuesForm"]["rTable"].setVerticalHeaderItem(row, item)
            self.optional_components["configurationValuesForm"]["rTable"].setItem(row, 0, QTableWidgetItem(""))

    def __update_g_table_labels(self, dimensionality):
        for row in range(dimensionality):
            label = f'δ(x{row + 1}) ='
            item = QTableWidgetItem(label)
            self.optional_components["configurationValuesForm"]["gTable"].setVerticalHeaderItem(row, item)
            self.optional_components["configurationValuesForm"]["gTable"].setItem(row, 0, QTableWidgetItem(""))

    def __update_dynamics_dimensionality(self, dimensionality: int):
        self.dynamicsTable.setRowCount(dimensionality)
        self.__update_dynamics_labels(dimensionality)

    def __update_r_table_dimensionality(self, dimensionality: int):
        if self.optional_components["configurationValuesForm"].get("rTable") is not None:
            self.optional_components["configurationValuesForm"]["rTable"].setRowCount(dimensionality)
            self.__update_r_table_labels(dimensionality)

    def __update_g_table_dimensionality(self, dimensionality: int):
        if self.optional_components["configurationValuesForm"].get("gTable") is not None:
            self.optional_components["configurationValuesForm"]["gTable"].setRowCount(dimensionality)
            self.__update_g_table_labels(dimensionality)

    def __set_noise_parameters(self, noise_type):
        possible_existing = self.__possible_existing_noise_parameters

        for name, item in self.optional_components["mainParameterConfigForm"].items():
            if name in possible_existing:
                if item is not None:
                    try:
                        if isinstance(item, QWidget):
                            item.deleteLater()
                        elif isinstance(item, QHBoxLayout):
                            self.mainParameterConfigForm.removeRow(self.__noise_parameters_row_number)
                    except Exception as e:
                        logger.debug(e)

        if noise_type == NoiseType.normal:
            self.optional_components["mainParameterConfigForm"]["noiseTypeComboBox"].blockSignals(True)
            self.optional_components["mainParameterConfigForm"]["noiseTypeComboBox"].setCurrentIndex(0)
            self.optional_components["mainParameterConfigForm"]["noiseTypeComboBox"].blockSignals(False)

            self.__init_sigma_text_label()
            self.__init_sigma_line_edit()
            self.__init_mean_text_label()
            self.__init_mean_line_edit()

            self.mainParameterConfigForm.insertRow(self.__noise_parameters_row_number,
                                                   self.optional_components["mainParameterConfigForm"][
                                                       "sigmaTextLabel"],
                                                   self.optional_components["mainParameterConfigForm"][
                                                       "sigmaLineEdit"])

            self.mainParameterConfigForm.insertRow(self.__noise_parameters_row_number,
                                                   self.optional_components["mainParameterConfigForm"][
                                                       "meanTextLabel"],
                                                   self.optional_components["mainParameterConfigForm"][
                                                       "meanLineEdit"])
        elif noise_type == NoiseType.exponential:
            self.optional_components["mainParameterConfigForm"]["noiseTypeComboBox"].blockSignals(True)
            self.optional_components["mainParameterConfigForm"]["noiseTypeComboBox"].setCurrentIndex(1)
            self.optional_components["mainParameterConfigForm"]["noiseTypeComboBox"].blockSignals(False)

            self.__init_rate_text_label()
            self.__init_rate_line_edit()
            self.mainParameterConfigForm.insertRow(self.__noise_parameters_row_number,
                                                   self.optional_components["mainParameterConfigForm"]
                                                   ["rateTextLabel"],
                                                   self.optional_components["mainParameterConfigForm"]
                                                   ["rateLineEdit"])

        elif noise_type == NoiseType.uniform:
            self.optional_components["mainParameterConfigForm"]["noiseTypeComboBox"].blockSignals(True)
            self.optional_components["mainParameterConfigForm"]["noiseTypeComboBox"].setCurrentIndex(2)
            self.optional_components["mainParameterConfigForm"]["noiseTypeComboBox"].blockSignals(False)

            self.__init_uniform_noise_parameters_layout()
            self.mainParameterConfigForm.insertRow(self.__noise_parameters_row_number,
                                                   self.optional_components["mainParameterConfigForm"]
                                                   ["uniformNoiseParametersLayout"])
        else:
            raise NotImplementedError("This noise type was not implemented.")

    def __set_discrete_deterministic_view(self):
        self.discreteContinuousGroupBox.blockSignals(True)
        self.deterministicStochasticGroupBox.blockSignals(True)
        self.radioButtonDiscrete.setChecked(True)
        self.radioButtonDeterministic.setChecked(True)
        self.discreteContinuousGroupBox.blockSignals(False)
        self.deterministicStochasticGroupBox.blockSignals(False)

    def __set_discrete_stochastic_view(self):
        self.discreteContinuousGroupBox.blockSignals(True)
        self.deterministicStochasticGroupBox.blockSignals(True)
        self.radioButtonDiscrete.setChecked(True)
        self.radioButtonStochastic.setChecked(True)
        self.discreteContinuousGroupBox.blockSignals(False)
        self.deterministicStochasticGroupBox.blockSignals(False)

        self.__init_noise_type_text_label()
        self.__init_noise_type_combo_box()

        self.__set_common_stochastic_view()

        self.mainParameterConfigForm.insertRow(self.__noise_parameters_row_number - 1,
                                               self.optional_components["mainParameterConfigForm"][
                                                   "noiseTypeTextLabel"],
                                               self.optional_components["mainParameterConfigForm"][
                                                   "noiseTypeComboBox"])
        self.__set_noise_parameters(NoiseType.normal)

    def __set_continuous_deterministic_view(self):
        self.discreteContinuousGroupBox.blockSignals(True)
        self.deterministicStochasticGroupBox.blockSignals(True)
        self.radioButtonContinuous.setChecked(True)
        self.radioButtonDeterministic.setChecked(True)
        self.discreteContinuousGroupBox.blockSignals(False)
        self.deterministicStochasticGroupBox.blockSignals(False)

    def __set_continuous_stochastic_view(self):
        self.discreteContinuousGroupBox.blockSignals(True)
        self.deterministicStochasticGroupBox.blockSignals(True)
        self.radioButtonContinuous.setChecked(True)
        self.radioButtonStochastic.setChecked(True)
        self.discreteContinuousGroupBox.blockSignals(False)
        self.deterministicStochasticGroupBox.blockSignals(False)

        self.__init_r_text_label()
        self.__init_r_table()
        self.__init_g_text_label()
        self.__init_g_table()
        self.__init_p_rate_text_label()
        self.__init_p_rate_line_edit()

        self.configurationValuesForm.addRow(self.optional_components["configurationValuesForm"]["gTextLabel"],
                                            self.optional_components["configurationValuesForm"]["gTable"])

        self.configurationValuesForm.addRow(self.optional_components["configurationValuesForm"]["rTextLabel"],
                                            self.optional_components["configurationValuesForm"]["rTable"])

        self.configurationValuesForm.addRow(self.optional_components["configurationValuesForm"]["pRateTextLabel"],
                                            self.optional_components["configurationValuesForm"]["pRateLineEdit"])

        self.__set_common_stochastic_view()

    def __set_common_stochastic_view(self):
        self.__init_optimize_text_label()
        self.__init_optimize_check_box()
        self.__init_c_text_label()
        self.__init_c_line_edit()
        self.__init_confidence_text_label()
        self.__init_confidence_line_edit()
        self.__init_min_confidence_text_label()
        self.__init_min_confidence_spin_box()
        self.__init_time_horizon_text_label()
        self.__init_time_horizon_line_edit()

        self.mainParameterConfigForm.insertRow(self.mainParameterConfigForm.rowCount() - 2,
                                               self.optional_components["mainParameterConfigForm"]["cTextLabel"],
                                               self.optional_components["mainParameterConfigForm"]["cLineEdit"])

        self.mainParameterConfigForm.insertRow(0,
                                               self.optional_components["mainParameterConfigForm"]["optimizeTextLabel"],
                                               self.optional_components["mainParameterConfigForm"]["optimizeCheckBox"])

        self.mainParameterConfigForm.insertRow(self.mainParameterConfigForm.rowCount() - 2,
                                               self.optional_components["mainParameterConfigForm"][
                                                   "timeHorizonTextLabel"],
                                               self.optional_components["mainParameterConfigForm"][
                                                   "timeHorizonLineEdit"])

        self.mainParameterConfigForm.insertRow(self.mainParameterConfigForm.rowCount() - 2,
                                               self.optional_components["mainParameterConfigForm"][
                                                   "minConfidenceTextLabel"],
                                               self.optional_components["mainParameterConfigForm"][
                                                   "minConfidenceSpinBox"])

        self.confidenceLayout.addWidget(self.optional_components["confidenceLayout"]["confidenceTextLabel"])
        self.confidenceLayout.addWidget(self.optional_components["confidenceLayout"]["confidenceLineEdit"])

    def __set_region_item_text(self, row, col, text):
        item = self.regionTable.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            self.regionTable.setItem(row, col, item)
        item.setText(text.strip().lower())

    def __set_dynamics_item_text(self, row, col, text):
        item = self.dynamicsTable.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            self.dynamicsTable.setItem(row, col, item)
        item.setText(text.strip().lower())

    def __set_r_item_text(self, row, col, text):
        table = self.optional_components["configurationValuesForm"].get("rTable")
        if table is not None:
            item = table.item(row, col)
            if item is None:
                item = QTableWidgetItem()
                table.setItem(row, col, item)
            item.setText(text.strip().lower())

    def __set_g_item_text(self, row, col, text):
        table = self.optional_components["configurationValuesForm"].get("gTable")
        if table is not None:
            item = table.item(row, col)
            if item is None:
                item = QTableWidgetItem()
                table.setItem(row, col, item)
            item.setText(text.strip().lower())
