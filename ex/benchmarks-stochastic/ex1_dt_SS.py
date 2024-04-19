# IMPORTS FROM INSTALLS
import time
import sympy as sp
import numpy as np

# IMPORTS FROM TOOL
from src.functions.parallel_dt_SS import parallel_dt_SS
from src.functions.dt_SS import dt_SS
# ========================= Parameters =========================

if __name__ == '__main__':

    dim = 1  # dimension of state space
    
    # Initial set
    L_initial = np.array([19.5])
    U_initial = np.array([20])

    # Unsafe set1
    L_unsafe1 = np.array([1])
    U_unsafe1 = np.array([17])

    # Unsafe set1
    L_unsafe2 = np.array([23])
    U_unsafe2 = np.array([50])

    # combine unsafe regions
    L_unsafe = np.array([L_unsafe1, L_unsafe2])
    U_unsafe = np.array([U_unsafe1, U_unsafe2])

    # State space
    L_space = np.array([1])
    U_space = np.array([50])

    # ========================= Symbolic Variables =========================
    x = sp.symbols(f'x0:{dim}')  # Create x1, x2, ..., x_degree symbols
    # ========================= Dynamics =========================
    beta = 0.06
    teta = 0.145
    Te = -15
    Th = 45
    R = 0.1

    nu = -0.0120155 * x[0] + 0.8
    f1 = (1 - beta - teta * nu) * x[0] + teta * Th * nu + beta * Te + R * varsigma[0]

    # Define the vector field
    f = np.array([f1])

    #noise terms
    NoiseType = "exponential"
    rate = [1]
    
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
        'varsigma': varsigma,
        'f': f,
        't': t,
        'noise_type': NoiseType,
        'optimize': False,
        'solver': "mosek",
        'confidence': None,
        'gam': None,
        'lam': None,
        'c_val': None,
        'sigma': None,
        'mean':None,
        'rate': rate,
        'a': None,
        'b': None,
        # Add other fixed parameters here
    }

    # List of degree values
    max_degree_value = 6
    start = time.time()
    
    ### Run the parallel implementation
    result = parallel_dt_SS(max_degree_value, **fixed_params)
    
    end = time.time()
    print("elapsed time:", end-start)
    if len(result) == 0:
        print("Results dictionary is empty.")
    else:
        print(result)
