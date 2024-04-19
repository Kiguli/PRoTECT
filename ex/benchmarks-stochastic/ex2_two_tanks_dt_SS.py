# IMPORTS FROM INSTALLS
import time
import sympy as sp
import numpy as np

# IMPORTS FROM TOOL
from src.functions.parallel_dt_SS import parallel_dt_SS
from src.functions.dt_SS import dt_SS
# ========================= Parameters =========================

if __name__ == '__main__':

    dim = 2  # dimension of state space

    # Initial set
    L_initial = np.array([1.75,1.75])
    U_initial = np.array([2.25,2.25])

    # Unsafe set1
    L_unsafe1 = np.array([9,9])
    U_unsafe1 = np.array([10,10])

    # combine unsafe regions
    L_unsafe = np.array([L_unsafe1])
    U_unsafe = np.array([U_unsafe1])

    # State space
    L_space = np.array([1,1])
    U_space = np.array([10,10])

    # ========================= Symbolic Variables =========================
    x = sp.symbols(f'x0:{dim}')  # Create x1, x2, ..., x_degree symbols
    varsigma = sp.symbols(f'varsigma0:{dim}')
    print(x)
    print(varsigma)
    # ========================= Dynamics =========================
    
    #noise terms
    NoiseType = "normal"
    sigma = np.array([0.01, 0.01])
    mean = np.array([0,0])
    
    tau = 0.1
    
    f1 = (1-tau)*x[0] + 4.5*tau + varsigma[0]
    f2 = tau*x[0] + (1-tau)*x[1] -3*tau + varsigma[1]
    
    # Define the vector field
    f = np.array([f1,f2])
    
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
        'sigma': sigma,
        'mean' : mean,
        'rate': None,
        'a': None,
        'b': None,
        # Add other fixed parameters here
    }

    # List of degree values
    max_degree_value = 6
    start = time.time()
    
    ### Uncomment this line to run the parallel implementation
    result = parallel_dt_SS(max_degree_value, **fixed_params)
    
    end = time.time()
    print("elapsed time:", end-start)
    if len(result) == 0:
        print("Results dictionary is empty.")
    else:
        print(result)
