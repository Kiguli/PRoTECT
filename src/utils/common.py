import numpy as np
import sympy as sp
from numpy import ndarray

from src.utils.exceptions import ExpressionFromStringError


def get_np_array_from_string(s: str) -> ndarray | None:
    """
    Parses the string and tries to get the numpy array inside.

    Args:
        s(str): String to parse.

    Returns:
        numpy.ndarray: numpy array with the values of double type
        and None if None type is provided.
    """
    if s is None:
        return None
    elif s.strip() == "":
        return np.empty(0, dtype=np.double)

    return np.asarray(s.replace(" ", "").split(','), dtype=np.double)


def get_expression_from_string(s: str, locals: list[sp.Symbol], err_message=None) -> sp.Expr:
    """Parses information from a string representing arbitrary math expressions.

    Args:
        s (str): The input string containing dynamics information.
        locals (list[sympy.symbol]): List of sympy symbols to encounter during parsing.
        err_message (str): The message in case the expression was not provided correctly.

    Returns:
        sp.Expr: A parsed sympy expression.

    Raises:
        ValueError: If the string cannot be parsed or does not contain valid information.
    """
    # Create a dictionary from the list of symbols
    symbol_dict = {symbol.name: symbol for symbol in locals}

    # Parse the expression using sympy
    try:
        expr = sp.sympify(s, locals=symbol_dict)
    except sp.SympifyError:
        if err_message is not None:
            raise ExpressionFromStringError(err_message)
        else:
            raise ExpressionFromStringError("Error in parsing the expression.")

    return expr
