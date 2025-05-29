# IMPORTS FROM INSTALLS
import time
import sympy as sp
import numpy as np

# IMPORTS FROM TOOL
from src.functions.parallel_dt_SS import parallel_dt_SS
from src.functions.dt_SS import dt_SS
# ========================= Parameters =========================

###
# We assume v[k]=0 and \sigma[k]=0, e.g. the case with no control input to create a verification case study.
###


if __name__ == '__main__':

    dim = 3  # dimension of state space

    # Initial set
    L_initial = np.array([4,8,8])
    U_initial = np.array([6,10,10])

    # Unsafe set
    L_unsafe1 = np.array([0,-1,-1])
    U_unsafe1 = np.array([0.9,-0.1,-0.1])
    
    # Unsafe set
    L_unsafe2 = np.array([0,-1,10.1])
    U_unsafe2 = np.array([0.9,-0.1,11])
    
    # Unsafe set
    L_unsafe3 = np.array([0,10.1,-1])
    U_unsafe3 = np.array([0.9,11,-0.1])
    
    # Unsafe set
    L_unsafe4 = np.array([0,10.1,10.1])
    U_unsafe4 = np.array([0.9,11,11])
    
    # Unsafe set
    L_unsafe5 = np.array([6.1,-1,-1])
    U_unsafe5 = np.array([7,-0.1,-0.1])
    
    # Unsafe set
    L_unsafe6 = np.array([6.1,-1,10.1])
    U_unsafe6 = np.array([7,-0.1,11])
    
    # Unsafe set
    L_unsafe7 = np.array([6.1,10.1,-1])
    U_unsafe7 = np.array([7,11,-0.1])
    
    # Unsafe set
    L_unsafe8 = np.array([6.1,10.1,10.1])
    U_unsafe8 = np.array([7,11,11])
    
    L_unsafe = np.array([L_unsafe1,L_unsafe2,L_unsafe3,L_unsafe4,L_unsafe5,L_unsafe6,L_unsafe7,L_unsafe8])
    
    U_unsafe = np.array([U_unsafe1,U_unsafe2,U_unsafe3,U_unsafe4,U_unsafe5,U_unsafe6,U_unsafe7,U_unsafe8])

    # State space
    L_space = np.array([0,-1,-1])
    U_space = np.array([7,11,11])

    # ========================= Symbolic Variables =========================
    x = sp.symbols(f'x0:{dim}')  # Create x1, x2, ..., x_degree symbols
    varsigma = sp.symbols(f'varsigma0:{dim}')
    # ========================= Dynamics =========================

    #noise terms
    NoiseType = "normal"
    sigma = np.array([0.001,0.001,0.001])
    mean = np.array([0,0,0])
    
    #discretized with time step = 20 seconds
    
    f1 = 0.8192*x[0] + 0.03412*x[1] + 0.01265*x[2] + varsigma[0]
    f2 = 0.01646*x[0] + 0.9822*x[1] + 0.0001*x[2] + varsigma[1]
    f3 = 0.0009*x[0] + 0.00002*x[1] + 0.9989*x[2] + varsigma[2]

    # Define the vector field
    f = np.array([f1,f2,f3])
    
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
        'confidence': 0.95,
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
