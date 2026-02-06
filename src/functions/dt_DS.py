# IMPORTS FROM INSTALLS
import picos
import sympy as sp
from SumOfSquares import *

# IMPORTS FROM TOOL
from .generate_polynomial import generate_polynomial


def dt_DS(b_degree, dim, L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space, x, f,
          solver="mosek", gam=None, lam=None, l_degree=None):
    """
    =========================================
    Calculate the barrier for the discrete-time Deterministic System (dt_DS)
    =========================================
    poly_degree = degree of barrier polynomial
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

    result = {}
    result['b_degree'] = b_degree

    # check array values are correct length
    if not (len(L_initial) == dim == len(U_initial) == len(L_space) == len(U_space) == len(x) == len(f)):
        raise ValueError("length of arrays doesn't match dimensions!")

    # set default l_degree value
    if l_degree is None:
        l_degree = b_degree

    # Get number of avoid regions
    avoid_regions = len(L_unsafe)
    if len(L_unsafe) != len(U_unsafe):
        raise ValueError("Unsafe regions were not defined correctly.")

    # ========================= Computing g =========================

    # Initial set
    g0 = generate_polynomial(x, L_initial, U_initial)

    g1 = []

    # Unsafe set
    for i in range(avoid_regions):
        g1.append(generate_polynomial(x, L_unsafe[i], U_unsafe[i]))

    # State set
    g = generate_polynomial(x, L_space, U_space)

    # ========================= Initialization the sum of squares program =========================

    prob = SOSProblem()

    # ========================= Lagrangian multiplier (polynomial variables) =========================
    try:
        # Lagrangian for the barrier
        Barrier = poly_variable('Barrier', x, b_degree)

        # Lagrangian for initial set
        L0 = [poly_variable('Li' + str(i + 1), x, l_degree) for i in range(len(x))]

        L1 = []
        # Lagrangian for avoid set
        for j in range(avoid_regions):
            L1.append([poly_variable('La' + str(j) + str(i + 1), x, l_degree) for i in range(len(x))])

        # Lagrangian for state space
        L = [poly_variable('Ls' + str(i + 1), x, l_degree) for i in range(len(x))]

        # gamma and lambda checks
        if gam is None:
            gamma = sp.symbols('gamma')
            gv = prob.sym_to_var(gamma)
            gamma_constraint = prob.add_constraint(gv > 0)
        else:
            if gam < 0:
                raise Exception("Gamma is less than zero!")
            gamma = gam

        if lam is None:
            lambda_ = sp.symbols('lambda_')
            lv = prob.sym_to_var(lambda_)
            lambda_constraint = prob.add_constraint(lv > 0)
        else:
            if lam < 0:
                raise Exception("Lambda is less than zero!")
            lambda_ = lam

        if gam is None and lam is None:
            level_set_condition = prob.add_constraint(lv - gv > 0)
        elif gam is None:
            level_set_condition = prob.add_constraint(lambda_ - gv > 0)
        elif lam is None:
            level_set_condition = prob.add_constraint(lv - gamma > 0)
        else:
            if lam <= gam:
                raise Exception("User defined lambda value is less than user defined gamma!")
    except Exception:
        return {"error": "Gamma or Lambda definition issues", "b_degree":b_degree}

    # ========================= Sub Difference Equations =========================

    # substitute the result of the difference equations
    y = [sp.Dummy(f'y{i}') for i in range(len(x))]
    Barrier_f = Barrier.subs([(x[i], y[i]) for i in range(len(x))])
    Barrier_f = Barrier_f.subs([(y[i], f[i]) for i in range(len(y))])

    # ========================= Constraints =========================
    try:
        L0_times_g0 = [L * g for L, g in zip(L0, g0)]
        first_condition = prob.add_sos_constraint(-Barrier - sum(L0_times_g0) + gamma, x)

        for i in range(avoid_regions):
            temp_L_times_g = [L * g for L, g in zip(L1[i], g1[i])]
            second_condition = prob.add_sos_constraint(Barrier - sum(temp_L_times_g) - lambda_, x)

        L_times_g = [L * g for L, g in zip(L, g)]
        last_condition = prob.add_sos_constraint(-Barrier_f + Barrier - sum(L_times_g), x)

        barrier_constraint = prob.add_sos_constraint(Barrier, x)

        # all lagrangians should be positive
        for i in L0:
            prob.add_sos_constraint(i, x)
        for j in range(avoid_regions):
            for i in L1[j]:
                prob.add_sos_constraint(i, x)
        for i in L:
            prob.add_sos_constraint(i, x)
    except AssertionError:
        return {"error": "AssertionError (probably odd b_degree)", "b_degree":b_degree}

    # ========================= Optimize =========================

    # prob.set_objective('max',lv)

    # ========================= Solve =========================
    try:
        prob.solve(solver=solver)
    except picos.modeling.problem.SolutionFailure:
        return {"error": "picos SolutionFailure", "b_degree":b_degree}
    except Exception:
        return {"error": "Solver Exception", "b_degree":b_degree}

    # ========================= Results =========================
    #Check if expression is scalar
    if(len(barrier_constraint.get_sos_decomp().free_symbols) == 0):
    	return {"error": "barrier is scalar!", "b_degree":b_degree}
    
    if (len(barrier_constraint.get_sos_decomp()) > 0 and len(first_condition.get_sos_decomp()) > 0 and len(second_condition.get_sos_decomp()) > 0 and len(last_condition.get_sos_decomp()) > 0):
    	result["barrier"] = sum(barrier_constraint.get_sos_decomp())
    else:
    	return {"error": "constraints are not sum of squares"}
    
    if gam is None:
        result["gamma"] = float(gv)
    else:
        result["gamma"] = gam
    if lam is None:
        result["lambda"] = float(lv)
    else:
        result["lambda"] = lam
    
    if result["lambda"] > result["gamma"] and result["lambda"] > 0 and result["gamma"] > 0:
    	return result
    elif result["lambda"] <= result["gamma"]:
    	return {"error": "lambda not greater than gamma","b_degree":b_degree}
    else:
    	return {"error": "numerical error on level sets e.g. negative value","b_degree":b_degree}
