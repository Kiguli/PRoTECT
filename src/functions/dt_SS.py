# IMPORTS FROM INSTALLS
import sympy as sp
import numpy as np
from SumOfSquares import *
import picos

# IMPORTS FROM TOOL
from .generate_polynomial import generate_polynomial
from .doublefactorial import doublefactorial
from .factorial import factorial


def dt_SS(b_degree, dim, L_initial, U_initial, L_unsafe, U_unsafe, L_space, U_space, x, varsigma, f,
          t, noise_type="normal", optimize=False, solver="mosek", confidence=None, gam=None, lam=None, c_val=None,
          mean=None, sigma=None, rate=None, a=None, b=None, l_degree=None):
    '''
    =========================================
    Calculate the barrier for the discrete-time Stochastic System (dt_SS)
    =========================================
    b_degree = degree of barrier polynomial
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
    NoiseType = "normal", "exponential", "uniform"
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

    try:
        # gamma and lambda checks
        if gam is None:
            gamma = sp.symbols('gamma')
            gv = prob.sym_to_var(gamma)
            gamma_constraint = prob.add_constraint(gv > 0)
        else:
            if gam < 0:
                raise Exception("Gamma is less than zero!")
            gamma = gam

        if c_val is None:
            c = sp.symbols('c')
            cv = prob.sym_to_var(c)
            c_constraint = prob.add_constraint(cv > 0)
        else:
            if c_val < 0:
                raise Exception("c is less than zero!")
            c = c_val

        if lam is None:
            lambda_ = sp.symbols('lambda_')
            lv = prob.sym_to_var(lambda_)
            lambda_constraint = prob.add_constraint(lv > 0)
        else:
            if lam <= 0:
                raise Exception("Lambda is less than or equal to zero!")
            lambda_ = lam
        if confidence is None or confidence == 0:
            if gam is None and lam is None and c_val is None:
                level_set_condition = prob.add_constraint(lv - gv - cv * t > 0)
            elif gam is None and c_val is None:
                level_set_condition = prob.add_constraint(lambda_ - gv - cv * t > 0)
            elif lam is None and c_val is None:
                level_set_condition = prob.add_constraint(lv - gamma - cv * t > 0)
            elif gam is None and lam is None:
                level_set_condition = prob.add_constraint(lv - gv - c * t > 0)
            elif gam is None:
                level_set_condition = prob.add_constraint(lambda_ - gv - c * t > 0)
            elif lam is None:
                level_set_condition = prob.add_constraint(lv - gamma - c * t > 0)
            elif c_val is None:
                level_set_condition = prob.add_constraint(lambda_ - gamma - cv * t > 0)
            else:
                if lam <= gam:
                    raise Exception("User defined lambda value is less than user defined gamma!")
                elif lam <= gam + c_val * t:
                    raise Exception("User defined parameters will give confidence less than 0!")
        else:
            if gam is None and lam is None and c_val is None:
                level_set_condition = prob.add_constraint(lv*confidence - gv - cv * t > 0)
            elif gam is None and c_val is None:
                level_set_condition = prob.add_constraint(lambda_*confidence - gv - cv * t > 0)
            elif lam is None and c_val is None:
                level_set_condition = prob.add_constraint(lv*confidence - gamma - cv * t > 0)
            elif gam is None and lam is None:
                level_set_condition = prob.add_constraint(lv*confidence - gv - c * t > 0)
            elif gam is None:
                level_set_condition = prob.add_constraint(lambda_*confidence - gv - c * t > 0)
            elif lam is None:
                level_set_condition = prob.add_constraint(lv*confidence - gamma - c * t > 0)
            elif c_val is None:
                level_set_condition = prob.add_constraint(lambda_*confidence - gamma - cv * t > 0)
            else:
                if lam <= gam:
                    raise Exception("User defined lambda value is less than user defined gamma!")
                elif lam <= gam + c_val * t:
                    raise Exception("User defined parameters will give confidence less than 0!")

    except Exception:
        return {"error": "Gamma, Lambda, or c value definition issues","b_degree":b_degree}

    # ========================= Sub Difference Equations =========================
    BB = Barrier.subs([(x[i], f[i]) for i in range(len(x))])
    BB = sp.expand(BB)

    for i in range(len(varsigma)):
        # Get coefficients and monomials
        coeff_dict = BB.as_coefficients_dict()
        m, mm = list(coeff_dict.keys()), list(coeff_dict.values())

        if noise_type == "normal" or noise_type == "gaussian":
            # use central moment and standard dev (sigma) and assume mean = 0
            for k in range(len(m)):
                m[k] = str(m[k])
                m[k] = m[k].replace("**", "^")
                s1 = m[k].split('*')
                chk = 0
                for k2 in range(len(s1)):
                    s2 = str(s1[k2]).split('^')

                    if s2[0] == varsigma[i].name and len(s2) == 1:
                        m[k] = 0
                        chk = 1
                        break

                    if s2[0] == varsigma[i].name and int(s2[1]) % 2 == 1:
                        m[k] = 0
                        chk = 1
                        break

                    if s2[0] == varsigma[i].name:
                        dfk = int(s2[1])
                        df = doublefactorial(dfk - 1) * sigma[i] ** dfk
                        s1[k2] = str(df)

                if chk == 0:
                    sy1 = sp.sympify('*'.join(s1))
                    m[k] = sy1

        elif noise_type == "exponential":
            # use non-central moment and rate parameter
            for k in range(len(m)):
                m[k] = str(m[k])
                m[k] = m[k].replace("**", "^")
                s1 = m[k].split('*')
                for k2 in range(len(s1)):
                    s2 = str(s1[k2]).split('^')
                    if s2[0] == varsigma[i].name and len(s2) == 1:
                        s1[k2] = str(1 / rate[i])
                        break

                    if s2[0] == varsigma[i].name:
                        dfk = int(s2[1])
                        df = factorial(dfk) / (rate[i] ** dfk)
                        s1[k2] = str(df)

                sy1 = sp.sympify('*'.join(s1))
                m[k] = sy1

        elif noise_type == "uniform":
            # use raw moment and a and b are ranges for noise per dimension
            for k in range(len(m)):
                m[k] = str(m[k])
                m[k] = m[k].replace("**", "^")
                s1 = m[k].split('*')
                for k2 in range(len(s1)):
                    s2 = str(s1[k2]).split('^')
                    if s2[0] == varsigma[i].name and len(s2) == 1:
                        s1[k2] = str((b[i] ** 2 - a[i] ** 2) / (3 * (b[i] - a[i])))
                        break

                    if s2[0] == varsigma[i].name:
                        dfk = int(s2[1])
                        df = (b[i] ** (dfk + 1) - a[i] ** (dfk + 1)) / ((dfk + 1) * (b[i] - a[i]))
                        s1[k2] = str(df)
                sy1 = sp.sympify('*'.join(s1))
                m[k] = sy1

        else:
            raise Exception(f"Unrecognised NoiseType: {noise_type}")

        # Recombine terms
        BB = sum([mm_val * m_val for mm_val, m_val in zip(mm, m)])
    Barrier_f = BB

    # ========================= Constraints =========================

    try:
        L0_times_g0 = [L * g for L, g in zip(L0, g0)]
        first_condition = prob.add_sos_constraint(-Barrier - sum(L0_times_g0) + gamma, x)

        for i in range(avoid_regions):
            temp_L_times_g = [L * g for L, g in zip(L1[i], g1[i])]
            second_condition = prob.add_sos_constraint(Barrier - sum(temp_L_times_g) - lambda_, x)

        L_times_g = [L * g for L, g in zip(L, g)]
        last_condition = prob.add_sos_constraint(-Barrier_f + Barrier + c - sum(L_times_g), x)

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
        return {"error": "AssertionError (probably odd b_degree)","b_degree":b_degree}

    # ========================= Optimization ==================
    if optimize:
        #lambda_bound = 1e6
        #lambda2_constraint = prob.add_constraint(lambda_bound - lv > 0)
        if lambda_ is None:
            return {"error":"lambda_ is None","b_degree":b_degree}
        prob.set_objective('min', (gv + cv*t)/lambda_)

    # ========================= Solve =========================
    try:
        prob.solve(solver=solver)
    except picos.modeling.problem.SolutionFailure:
        return {"error": "picos SolutionFailure","b_degree":b_degree}
    except Exception:
        return {"error": "Solver Exception","b_degree":b_degree}

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
    if c_val is None:
        result["c"] = float(cv)
    else:
        result["c"] = c_val
    try:
        result["confidence"] = 1 - (result['gamma'] + result['c'] * t) / result['lambda']
    except ZeroDivisionError:
        return {"error": "Divide by zero error","b_degree":b_degree}
        
    if result["lambda"] > result["gamma"] and result["lambda"] > 0 and result["gamma"] > 0 and result["c"]>0:
    	return result
    elif result["lambda"] <= result["gamma"]:
    	return {"error": "lambda not greater than gamma","b_degree":b_degree}
    else:
    	return {"error": "numerical error on level sets e.g. negative value","b_degree":b_degree}
