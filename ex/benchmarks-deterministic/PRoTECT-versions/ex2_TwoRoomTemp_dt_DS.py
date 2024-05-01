# IMPORTS FROM INSTALLS
import time
import sympy as sp
import numpy as np

# IMPORTS FROM TOOL
from src.functions.dt_DS import dt_DS
from src.functions.parallel_dt_DS import parallel_dt_DS

# ========================= Parameters =========================
if __name__ == '__main__':
    dim = 2  # dimension of state space

    # Initial set
    L_initial = np.array([18,18])
    U_initial = np.array([19.75,19.75])

    # Unsafe set
    L_unsafe1 = np.array([22,22])
    U_unsafe1 = np.array([23,23])
    
    # combine unsafe regions
    L_unsafe = np.array([L_unsafe1])
    U_unsafe = np.array([U_unsafe1])
    # State space
    L_space = np.array([18,18])
    U_space = np.array([23,23])

    # ========================= Symbolic Variables =========================
    x = sp.symbols(f'x1:{dim + 1}')  # Create x1, x2, ..., x_degree symbols
    print(f"Sympy variables: {x}\t Type: {type(x[0])}")  # print variables
    # ========================= Dynamics =========================
	
    tau = 5  # discretise param
    alpha = 5 * 1e-2  # heat exchange
    alpha_e1 = 5 * 1e-3  # heat exchange 1
    alpha_e2 = 8 * 1e-3  # heat exchange 2
    temp_e = 15  # external temp
    alpha_h = 3.6 * 1e-3  # heat exchange room-heater
    temp_h = 55  # boiler temp	

    f1 = (1 - tau * (alpha + alpha_e1)) * x[0] + tau * alpha * x[1] + tau * alpha_e1 * temp_e
    f2 = (1 - tau * (1.0 * alpha + alpha_e2)) * x[1] + tau * alpha * x[0] + tau * alpha_e2 * temp_e

    # Define the vector field
    f = np.array([f1,f2])

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
    max_b_degree = 6
    single_run_b_degree = 2
    start = time.time()
    
    ### Uncomment this line to run the parallel implementation
    #result = parallel_dt_DS(max_b_degree, **fixed_params)
    
    ### Uncomment this line to run the serial implementation
    result = dt_DS(single_run_b_degree, **fixed_params)
    
    end = time.time()
    print("elapsed time:", end-start)
    if result == None:
        print("Results dictionary is empty.")
    else:
        print(result)
