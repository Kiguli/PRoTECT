def factorial(n):
    '''
    =========================================
    Iteratively compute factorial of n
    =========================================
    n = positive integer
    '''
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)