def generate_polynomial(variables, lower_bounds, upper_bounds):
    '''
    =========================================
    Create the respective polynomials based on the bounds (g in the paper)
    =========================================
    variables = sympy variables
    lower_bounds = numpy array of lower bound values
    upper_bounds = numpy array of upper bound values
    '''
    polynomial = [(var - lower) * (upper - var) for var, lower, upper in zip(variables, lower_bounds, upper_bounds)]
    return polynomial
