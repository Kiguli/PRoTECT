# IMPORTS FROM INSTALLS
import time
import sympy as sp
import numpy as np

# IMPORTS FROM TOOL
from src.functions.parallel_ct_SS import parallel_ct_SS
from src.functions.ct_SS import ct_SS
# ========================= Parameters =========================

if __name__ == '__main__':

    dim = 1  # dimension of state space

    # Initial set
    L_initial = np.array([19.5])
    U_initial = np.array([20])

    # Unsafe set
    L_unsafe1 = np.array([1])
    U_unsafe1 = np.array([17])
    L_unsafe2 = np.array([23])
    U_unsafe2 = np.array([50])

    # combine unsafe regions
    L_unsafe = np.array([L_unsafe1, L_unsafe2])
    U_unsafe = np.array([U_unsafe1, U_unsafe2])

    # State space
    L_space = np.array([1])
    U_space = np.array([50])

    #noise terms
    delta = np.array([0.1]);  # diffusion term (for brownian part)
    rho = np.array([0.1]);  # reset term (for poisson distribution)
    p_rate = np.array([0.1]); # poisson rate
    
    # ========================= Symbolic Variables =========================
    x = sp.symbols(f'x1:{dim + 1}')  # Create x1, x2, ..., x_degree symbols
    # ========================= Dynamics =========================
    eta = 0.005  # Conduction factor between the rooms i+1/i-1 and the room i
    beta = 0.06  # Conduction factor between the external environment and the room i
    teta = 0.156  # Conduction factor between the heater and the room i
    Te = -15  # Outside temperature
    Th = 48  # Heater temperature

    f1 = (-2 * eta * x[0] - beta * x[0] - teta * (-0.0120155 * x[0] ** 2 + 0.7 * x[0])) + teta * Th * ( -0.0120155 * x[0] + 0.7) + beta * Te;

    # Define the vector field
    f = np.array([f1])

    #time horizon
    t = 5

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
        'delta': delta,
        'rho': rho,
        'p_rate': p_rate,
        't': t,
        'optimize': True,
        'solver': "mosek",
        'confidence': None,
        'gam': None,
        'lam': 10,
        'c_val': None,
        'l_degree':None,
        # Add other fixed parameters here
    }

    # List of degree values
    max_degree_value = 6
    start = time.time()
    
    ### Run the parallel implementation
    result = parallel_ct_SS(max_degree_value, **fixed_params)
    
    end = time.time()
    print("elapsed time:", end-start)
    if len(result) == 0:
        print("Results dictionary is empty.")
    else:
        print(result)
