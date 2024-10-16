# IMPORTS FROM INSTALLS
import time
import sympy as sp
import numpy as np

# IMPORTS FROM TOOL
from src.functions.parallel_dt_SS import parallel_dt_SS
from src.functions.dt_SS import dt_SS
# ========================= Parameters =========================

if __name__ == '__main__':

    dim = 3  # dimension of state space

    # Initial set
    L_initial = np.array([17,17,17])
    U_initial = np.array([18,18,18])

    # Unsafe set
    L_unsafe1 = np.array([29,29,29])
    U_unsafe1 = np.array([30,30,30])

    # combine unsafe regions
    L_unsafe = np.array([L_unsafe1])
    U_unsafe = np.array([U_unsafe1])

    # State space
    L_space = np.array([17,17,17])
    U_space = np.array([30,30,30])

    # ========================= Symbolic Variables =========================
    x = sp.symbols(f'x0:{dim}')  # Create x1, x2, ..., x_degree symbols
    varsigma = sp.symbols(f'varsigma0:{dim}')
    # ========================= Dynamics =========================

    #noise terms
    NoiseType = "normal"
    sigma = np.array([0.01,0.01,0.01])
    mean = np.array([0,0,0])
    
    T_e = 10
    alpha_e = 8e-3
    alpha = 6.2e-3
    tau = 5
    
    f1 = (1-tau*(alpha+alpha_e))*x[0] + tau*alpha*x[1] + tau*alpha_e*T_e + varsigma[0]
    f2 = (1-tau*(2*alpha+alpha_e))*x[1] + tau*alpha*(x[0]+x[2]) + tau*alpha_e*T_e + varsigma[1]
    f3 = (1-tau*(alpha+alpha_e))*x[2] + tau*alpha*x[1] + tau*alpha_e*T_e + varsigma[2]

    # Define the vector field
    f = np.array([f1,f2,f3])
    
    #time horizon
    t = 3

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
        'optimize': True,
        'solver': "mosek",
        'confidence': None,
        'gam': None,
        'lam': 10,
        'c_val': None,
        'sigma': sigma,
        'mean' : mean,
        'rate': None,
        'a': None,
        'b': None,
        # Add other fixed parameters here
    }

    # List of degree values
    degree = 4
    start = time.time()
    
    ### Uncomment this line to run the parallel implementation
    result = dt_SS(degree, **fixed_params)
    
    end = time.time()
    print("elapsed time:", end-start)
    if len(result) == 0:
        print("Results dictionary is empty.")
    else:
        print(result)
