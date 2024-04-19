import time
import sympy as sp
import numpy as np
from SumOfSquares import *
import multiprocessing
from functools import partial
from pebble import ProcessPool, ThreadPool
from concurrent.futures import as_completed

# IMPORTS FROM TOOL
from src.functions.dt_SS import dt_SS


def parallel_dt_SS(b_degree, dim, L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space, x, varsigma, f,
                   t, noise_type="normal", optimize=False, solver="mosek", confidence=None, gam=None, lam=None,
                   c_val=None, mean=None, sigma=None, rate=None, a=None, b=None, l_degree=None):
    '''
        =========================================
        Calculate the barrier for the discrete-time Stochastic System (dt_SS)
        =========================================
        b_degree = maximum b_degree of barrier polynomials to run in parallel
        dim = dimension of state space
        avoid_regions = number of avoid regions (or unsafe sets)
        L_initial = numpy array of initial set lower bounds
        U_initial = numpy array of initial set upper bounds
        L_unsafe = numpy array of unsafe set(s) lower bounds
        U_unsafe = numpy array of unsafe set(s) upper bounds
        L_space = numpy array of state set lower bounds
        U_space = numpy array of state set upper bounds
        x = list of sympy variables
        varsigma = list of sympy variables for noise
        f = numpy array of dynamics functions
        t = time horizon
        NoiseType = "normal" or "gaussian", "exponential", "uniform"
        optimize = True for optimize the solution (need gamma and c values) or find any feasible solution
        solver = "mosek" or "cvxopt"
        confidence = minimal confidence of feasible solution
        gam = chosen value for gamma
        lam = chosen value for lambda
        c_val = chosen c value (multiplied by time horizon in confidence)
        sigma = standard deviations of normal/gaussian noise (mean assumed to be zero)
        rate = exponential noise rate parameter
        a = lower bound of integral for uniform noise
        b = upper bound of integral for uniform noise
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
        'varsigma': varsigma,
        'f': f,
        't': t,
        'noise_type': noise_type,
        'optimize': optimize,
        'solver': solver,
        'confidence': confidence,
        'gam': gam,
        'lam': lam,
        'c_val': c_val,
        'sigma': sigma,
        'mean': mean,
        'rate': rate,
        'a': a,
        'b': b,
        'l_degree': l_degree
    }

    get_degree_values = lambda b_degree: [x for x in range(2, b_degree+2, 2)] if b_degree % 2 == 0 else \
        [x for x in range(2, b_degree+1, 2)]
    
    degree_values = get_degree_values(b_degree)
    results_list = []

    with ProcessPool() as pool:
        futures = {pool.schedule(dt_SS, args=(degree,), kwargs=fixed_params): degree for degree in degree_values}

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
