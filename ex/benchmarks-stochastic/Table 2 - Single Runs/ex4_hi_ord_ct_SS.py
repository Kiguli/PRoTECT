# IMPORTS FROM INSTALLS
import time
import sympy as sp
import numpy as np

# IMPORTS FROM TOOL
from src.functions.parallel_ct_SS import parallel_ct_SS,ct_SS

# ========================= Parameters =========================
if __name__ == '__main__':
    dim = 4  # dimension of state space

    # Initial set
    L_initial = np.array([0.5, 0.5, 0.5, 0.5])
    U_initial = np.array([1.5, 1.5, 1.5, 1.5])

    # Unsafe set
    L_unsafe1 = np.array([-2.4, -2.4, -2.4, -2.4])
    U_unsafe1 = np.array([-1.6, -1.6, -1.6, -1.6])

    # combine unsafe regions
    L_unsafe = np.array([L_unsafe1])
    U_unsafe = np.array([U_unsafe1])

    # State space
    L_space = np.array([-2, -2, -2, -2])
    U_space = np.array([2, 2, 2, 2])

    # ========================= Symbolic Variables =========================
    x = sp.symbols(f'x1:{dim + 1}')  # Create x1, x2, ..., x_degree symbols
    print(x)  # print variables

    # ========================= Dynamics =========================

    f1 = x[1]
    f2 = x[2]
    f3 = x[3]
    f4 = -3980 * x[3] - 4180 * x[2] - 2400 * x[1] - 576 * x[0]
    
    delta = [0.1,0.1,0.1,0.1]
    rho = [0,0,0,0]
    p_rate = [0,0,0,0]
    t = 3
    
    # Define the vector field
    f = np.array([f1,f2,f3,f4])

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
    degree = 4
    start = time.time()
    
    ### Uncomment this line to run the parallel implementation
    result = ct_SS(degree, **fixed_params)
    
    end = time.time()
    print("elapsed time:", end-start)
    if len(result) == 0:
        print("Results dictionary is empty.")
    else:
        print(result)
