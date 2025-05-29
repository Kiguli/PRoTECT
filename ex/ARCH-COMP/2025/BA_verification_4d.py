# IMPORTS FROM INSTALLS
import time
import sympy as sp
import numpy as np

# IMPORTS FROM TOOL
from src.functions.parallel_dt_SS import parallel_dt_SS
from src.functions.dt_SS import dt_SS
# ========================= Parameters =========================

#
#We assume a naive controller that provides input u[k] = 0, meaning we consider a verification case of the BA.
#

if __name__ == '__main__':

    dim = 4  # dimension of state space

    # Initial set
    L_initial = np.array([17,17,32,32])
    U_initial = np.array([18,18,34,34])

    # Unsafe set
    L_unsafe1 = np.array([18,18,29,29])
    U_unsafe1 = np.array([18.9,21,36,36])
    # Unsafe set
    L_unsafe2 = np.array([18,18,29,29])
    U_unsafe2 = np.array([21,18.9,36,36])
    # Unsafe set
    L_unsafe3 = np.array([18,18,29,29])
    U_unsafe3 = np.array([21,21,29.9,36])
    # Unsafe set
    L_unsafe4 = np.array([18,18,29,29])
    U_unsafe4 = np.array([21,21,36,29.9])

    # combine unsafe regions
    L_unsafe = np.array([L_unsafe1])
    U_unsafe = np.array([U_unsafe1])

    # State space
    L_space = np.array([18,18,29,29])
    U_space = np.array([21,21,36,36])

    # ========================= Symbolic Variables =========================
    x = sp.symbols(f'x0:{dim}')  # Create x1, x2, ..., x_degree symbols
    varsigma = sp.symbols(f'varsigma0:{dim}')
    # ========================= Dynamics =========================

    #noise terms
    NoiseType = "normal"
    sigma = np.array([12.9199, 12.9199, 2.5826, 3.2279])
    mean = np.array([0,0,0,0])
    
    f1 = 0.6682*x[0] + 0.02632*x[2] + 3.4378 + varsigma[0]
    f2 = 0.683*x[1] + 0.02096*x[3] + 2.9272 + varsigma[1]
    f3 = 1.0005*x[0] -0.000499*x[2] + 13.0207 + varsigma[2]
    f4 =  0.8004*x[1] + 0.1996*x[3] + 10.4166 + varsigma[3]

    # Define the vector field
    f = np.array([f1,f2,f3,f4])
    
    #time horizon
    t = 10

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
        'confidence': 0.999,
        'gam': None,
        'lam': None,
        'c_val': None,
        'sigma': sigma,
        'mean' : mean,
        'rate': None,
        'a': None,
        'b': None,
        # Add other fixed parameters here
    }

    # List of degree values
    degree = 2
    start = time.time()
    
    ### Uncomment this line to run the parallel implementation
    result = dt_SS(degree, **fixed_params)
    
    end = time.time()
    print("elapsed time:", end-start)
    if len(result) == 0:
        print("Results dictionary is empty.")
    else:
        print(result)
