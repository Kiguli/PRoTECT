import time
import sympy as sp
import numpy as np
from SumOfSquares import *
import multiprocessing
from functools import partial
from pebble import ProcessPool, ThreadPool
from concurrent.futures import as_completed

# IMPORTS FROM TOOL
from src.functions.ct_SS import ct_SS


def parallel_ct_SS(b_degree, dim, L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space, x, f, delta, rho,
                   p_rate, t, optimize=False, solver="mosek", confidence=None, gam=None, lam=None, c_val=None,
                   l_degree=None):
    '''
        =========================================
        Calculate the barrier for the continuous-time Stochastic System (ct_SS)
        =========================================
        b_degree = maximum b_degree of barrier polynomials to run in parallel
        dim = dimension of state space
        L_initial = numpy array of initial set lower bounds
        U_initial = numpy array of initial set upper bounds
        L_unsafe = numpy array of unsafe set(s) lower bounds
        U_unsafe = numpy array of unsafe set(s) upper bounds
        L_space = numpy array of state set lower bounds
        U_space = numpy array of state set upper bounds
        x = list of sympy variables
        f = numpy array of dynamics functions
        G = numpy array of diffusion term for brownian (could be equations)
        R = numpy array of reset term, for poisson (could be equations)
        p_rate = numpy array of poisson rate values
        t = time horizon
        optimize = True for optimize the solution (need gamma and c values) or find any feasible solution
        solver = cvxopt for True, mosek for False
        confidence = minimal confidence of feasibly solution
        gam = chosen value for gamma
        lam = chosen value for lambda
        c_val = chosen c value (multiplied by time horizon in confidence)
        l_degree = degree of lagrangian multipliers
        '''

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
        'optimize': optimize,
        'solver': solver,
        'confidence': confidence,
        'gam': gam,
        'lam': lam,
        'c_val': c_val,
        'l_degree': l_degree
    }

    get_degree_values = lambda b_degree: [x for x in range(2, b_degree+2, 2)] if b_degree % 2 == 0 else \
        [x for x in range(2, b_degree+1, 2)]
    
    degree_values = get_degree_values(b_degree)
    results_list = []

    with ProcessPool() as pool:

        futures = {pool.schedule(ct_SS, args=(degree,), kwargs=fixed_params): degree for degree in degree_values}

        for future in as_completed(futures, timeout=None):
            try:
                result = future.result()
            except Exception as exc:
                print(f'Function raised an exception: {exc}')
            else:
                if result is not None:
                    if "barrier" in result:
                        results_list.append(result)
                    elif "error" in result:
                        print("Error in degree:", result['b_degree'], " -- ", result['error'])
                    else:
                        print("Error!", " -- Unknown error!")

    if results_list:
        highest_confidence_barrier = max(results_list, key=lambda x: x['confidence'])
        return highest_confidence_barrier
    # Do something with highest_confidence_barrier
    else:
        print("No results with a barrier found.")
        return {"error": "No results with a barrier found"}
