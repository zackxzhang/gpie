# -*- coding: utf-8 -*-
# data utilities

import numpy as np                                                # type: ignore
from collections.abc import Callable, Sequence
from functools import wraps, partial
from numpy import ndarray
from typing import TypeAlias


array32 = partial(np.array, dtype=np.float32)
array16 = partial(np.array, dtype=np.float16)


def is_array(x, ndim=1, dtype=np.number) -> bool:
    if (
        isinstance(x, ndarray)          and
        x.ndim == ndim                  and
        np.issubdtype(x.dtype, dtype)
    ):
        return True
    else:
        return False


def map_array(
    f: Callable,
    x: ndarray | Sequence[ndarray]
) -> ndarray | Sequence[ndarray]:
    return np.vectorize(f, otypes=[float])(x)


V: TypeAlias = float | ndarray
B: TypeAlias = tuple[float, float] | tuple[ndarray, ndarray]


def concat_values(*values: V) -> ndarray:
        vs = list()
        for v in values:
            if isinstance(v, float):
                vs.append(v)
            elif isinstance(v, ndarray) and np.issubdtype(v.dtype, np.number):
                vs.extend(v.flat)
            else:
                raise TypeError('values must be float or array.')
        return np.array(vs)


def concat_bounds(*bounds: B) -> tuple[ndarray, ndarray]:
        ls = list()
        us = list()
        for b in bounds:
            if isinstance(b, tuple) and len(b) == 2:
                if isinstance(b[0], float) and isinstance(b[1], float):
                    ls.append(b[0])
                    us.append(b[1])
                elif (
                        isinstance(b[0], ndarray)
                    and isinstance(b[1], ndarray)
                    and np.issubdtype(b[0].dtype, np.number)
                    and np.issubdtype(b[1].dtype, np.number)
                ):
                    ls.extend(b[0].flat)
                    us.extend(b[1].flat)
                else:
                    raise TypeError('bounds must be tuple of float or array.')
            else:
                 raise TypeError('bounds must be tuple.')
        return np.array(ls), np.array(us)


# check data dimensions (only once when feeding data to model or function)
def check_X(X: ndarray):
    if not is_array(X, 2, np.number):
        raise ValueError('X must be a 2d numeric array.')


def check_X_Z(X: ndarray, Z: ndarray):
    if not is_array(X, 2, np.number):
        raise ValueError('X must be a 2d numeric array.')
    if not is_array(Z, 2, np.number):
        raise ValueError('Z must be a 2d numeric array.')
    if not X.shape[1] == Z.shape[1]:
        raise ValueError('X and Z must agree on 2nd dimension.')


def check_X_y(X: ndarray, y: ndarray):
    if not is_array(X, 2, np.number):
        raise ValueError('X must be a 2d numeric array.')
    if not is_array(y, 1, np.number):
        raise ValueError('y must be a 1d numeric array.')
    if not X.shape[0] == y.shape[0]:
        raise ValueError('X and y must agree on 1st dimension.')


def check_X_update(X: ndarray, Xo: ndarray):
    check_X_Z(X, Xo)


# check if additional data matches with existing data
def check_X_y_update(X: ndarray, y: ndarray, Xo: ndarray, yo: ndarray):
    # assuming Xo and yo are already checked
    check_X_y(X, y)
    check_X_Z(X, Xo)


# check class labels
def check_labels(y: ndarray):
    pass


# make decorators
def fun2dec(outer):        # outer is a check function
    @wraps(outer)
    def decorator(inner):  # inner is a class/instance method
        @wraps(inner)
        def wrapper(self, *args, **kwargs):
            outer(*args, **kwargs)
            return inner(self, *args, **kwargs)
        return wrapper
    return decorator


audit_X = fun2dec(check_X)
audit_X_Z = fun2dec(check_X_Z)
audit_X_y = fun2dec(check_X_y)
audit_X_update = fun2dec(check_X_update)
audit_X_y_update = fun2dec(check_X_y_update)


class GridData:

    def __init__(self, X: ndarray, y: ndarray | None = None):
        raise NotImplementedError
