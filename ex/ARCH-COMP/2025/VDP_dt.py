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
    L_initial = np.array([-4.5,-4.5])
    U_initial = np.array([4.5,4.5])

    # Unsafe set1
    L_unsafe1 = np.array([-6,-5])
    U_unsafe1 = np.array([-5,5])
    
    # Unsafe set2
    L_unsafe2 = np.array([5,-5])
    U_unsafe2 = np.array([6,5])
    
    # Unsafe set3
    L_unsafe3 = np.array([-5,5])
    U_unsafe3 = np.array([5,6])
    
    # Unsafe set4
    L_unsafe4 = np.array([-5,-6])
    U_unsafe4 = np.array([5,-5])

    # combine unsafe regions
    L_unsafe = np.array([L_unsafe1,L_unsafe2,L_unsafe3,L_unsafe4])
    U_unsafe = np.array([U_unsafe1,U_unsafe2,L_unsafe3,L_unsafe4])

    # State space
    L_space = np.array([-6,-6])
    U_space = np.array([6,6])

    # ========================= Symbolic Variables =========================
    x = sp.symbols(f'x0:{dim}')  # Create x1, x2, ..., x_degree symbols
    varsigma = sp.symbols(f'varsigma0:{dim}')
    # ========================= Dynamics =========================

    #noise term
    NoiseType = "uniform"
    a = np.array([-0.02,-0.02])
    b = np.array([0.02, 0.02])
    
    tau = 0.1
    
    f1 = x[0] + tau*x[1] + varsigma[0]
    f2 = x[1] + (-x[0] + (1-x[0]**2)*x[1])*tau + varsigma[1]
    
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
        'confidence': 0.8,
        'gam': None,
        'lam': None,
        'c_val': None,
        'sigma': None,
        'mean' : None,
        'rate': None,
        'a': a,
        'b': b,
        # Add other fixed parameters here
    }

    # List of degree values
    degree = 6
    start = time.time()
    
    ### Uncomment this line to run the parallel implementation
    result = dt_SS(degree, **fixed_params)
    
    end = time.time()
    print("elapsed time:", end-start)
    if len(result) == 0:
        print("Results dictionary is empty.")
    else:
        print(result)
