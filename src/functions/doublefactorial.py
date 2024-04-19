def doublefactorial(n):
    '''
    =========================================
    Iteratively calculates the doublefactorial of n
    =========================================
    n = positive integer
    '''
    if n == 0 or n == 1:
        return 1
    else:
        return n * doublefactorial(n-2)