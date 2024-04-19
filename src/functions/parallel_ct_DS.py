import time
from pebble import ProcessPool, ThreadPool
from concurrent.futures import as_completed
from src.functions.ct_DS import ct_DS


def parallel_ct_DS(b_degree, dim, L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space, x, f,
                   solver="mosek", gam=None, lam=None, l_degree=None):
    """
    =========================================
    Calculate the barrier for the continuous-time Deterministic System (ct_DS)
    =========================================
    b_degree = maximum b_degrees of barrier polynomials to run in parallel
    dim = dimension of state space
    L_initial = numpy array of initial set lower bounds
    U_initial = numpy array of initial set upper bounds
    L_unsafe = numpy array of unsafe set(s) lower bounds
    U_unsafe = numpy array of unsafe set(s) upper bounds
    L_space = numpy array of state set lower bounds
    U_space = numpy array of state set upper bounds
    x = list of sympy variables
    f = numpy array of dynamics functions
    solver = mosek for True, cvxopt for False
    gam = choose a value for gamma
    lam = choose a value for lambda_
    l_degree = degree of lagrangian multipliers
    """

    fixed_params = {
        'dim': dim,
        'L_initial': L_initial,
        'U_initial': U_initial,
        'l_degree': l_degree,
        'L_unsafe': L_unsafe,
        'U_unsafe': U_unsafe,
        'L_space': L_space,
        'U_space': U_space,
        'x': x,
        'f': f,
        'solver': solver,
        'gam': gam,
        'lam': lam
    }

    get_degree_values = lambda b_degree: [x for x in range(2, b_degree+2, 2)] if b_degree % 2 == 0 else \
        [x for x in range(2, b_degree+1, 2)]
    
    degree_values = get_degree_values(b_degree)
    print(degree_values)
    
    results_dict = None

    with ProcessPool() as pool:

        futures = {pool.schedule(ct_DS, args=(degree,), kwargs=fixed_params): degree for degree in degree_values}

        for future in as_completed(futures, timeout=None):
            try:
                result = future.result()
            except Exception as exc:
                print(f'Function raised an exception: {exc}')
            else:
                if result is not None:
                    if "barrier" in result:
                        results_dict = result
                        pool.stop()
                        pool.join(timeout=0)
                        break
                    elif "error" in result:
                        print("Error in degree:", result['b_degree'], " -- ", result['error'])
                    else:
                        print("Error!", " -- Unknown error!")

    return results_dict
