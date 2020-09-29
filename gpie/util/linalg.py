# -*- coding: utf-8 -*-
# robust and efficient matrix computations

import numpy as np                                                # type: ignore
from numpy import ndarray
from numpy.linalg import slogdet, solve, eigvalsh                 # type: ignore
from scipy.linalg import cho_solve, cho_factor                    # type: ignore


# log determinant
def logdet(A: ndarray) -> float:
    _, d = slogdet(A)
    return d


# log determinant for hermitian matrix
def logdeth(A: ndarray) -> float:
    return np.log(eigvalsh(A)).sum()


# invert then multiply
invmul = solve


# inverse then multiply for hermitian matrix
def invmulh(A: ndarray, b: ndarray) -> ndarray:
    return cho_solve(cho_factor(A), b)
