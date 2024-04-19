# IMPORTS FROM INSTALLS
import time
import sympy as sp
import numpy as np

# IMPORTS FROM TOOL
from src.functions.parallel_dt_DS import parallel_dt_DS
from src.functions.dt_DS import dt_DS

# ========================= Parameters =========================
if __name__ == '__main__':
    dim = 1  # dimension of state space

    # Initial set
    L_initial = np.array([-0.5])
    U_initial = np.array([0.5])

    # Unsafe set1
    L_unsafe1 = np.array([-6])
    U_unsafe1 = np.array([-5])

    # combine unsafe regions
    L_unsafe = np.array([L_unsafe1])
    U_unsafe = np.array([U_unsafe1])

    # State space
    L_space = np.array([-6])
    U_space = np.array([6])

    # ========================= Symbolic Variables =========================
    x = sp.symbols(f'x1:{dim + 1}')  # Create x1, x2, ..., x_degree symbols
    print(f"Sympy variables: {x}")  # print variables
    # ========================= Dynamics =========================

    tau = 5  # discretise param
    alpha_e = 8 * 1e-3  # heat exchange room-external
    temp_e = 15  # external temp
    alpha_h = 3.6 * 1e-3  # heat exchange room-heater
    temp_h = 55  # boiler temp

    f1 = x[0] + tau * ((temp_e - x[0]) + alpha_h * (temp_h - x[0]) * 0.1)

    # Define the vector field
    f = np.array([f1])

    fixed_params = {
        'dim': dim,
        'L_initial': L_initial,
        'U_initial': U_initial,
        'L_unsafe': L_unsafe,
        'U_unsafe': U_unsafe,
        'L_space': L_space,
        'U_space': U_space,
        'x': x,
        'f': f,
        'solver': "mosek",
        'gam': None,
        'lam': None,
        # Add other fixed parameters here
    }
    # List of degree values
    max_degree_value = 6
    single_degree_value = 2
    start = time.time()
    
    ### Uncomment this line to run the parallel implementation
    #result = parallel_dt_DS(max_degree_value, **fixed_params)
    
    ###Uncomment this line to run the serial implementation
    result = dt_DS(single_degree_value, **fixed_params)
    
    end = time.time()
    print("elapsed time:", end-start)
    if result == None:
        print("Results dictionary is empty.")
    else:
        print(result)
