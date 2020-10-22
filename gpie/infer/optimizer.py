# -*- coding: utf-8 -*-
# numerical optimizer

import numpy as np                                                # type: ignore
import scipy                                                      # type: ignore
import warnings
from functools import partial
from multiprocessing import Pool
from numpy import ndarray
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union
from ..base import Optimizer, Bounds, OPT_BACKENDS


class GradientDescentOptimizer(Optimizer):

    """
    gradient descent optimizer
    wrapper of optimizer backends
    """

    backends = {'scipy'}

    def __init__(self, solver: str, bounds: Bounds, x0: ndarray,
                 fun: Optional[Callable] = None,
                 jac: Optional[Union[Callable, bool]] = None,
                 n_restarts: int = 0, backend='scipy'):

        super().__init__()
        # configuration
        self.backend = backend
        # algorithm
        self.min = solver
        # search space
        self.bounds = bounds
        # initialization
        self.X0 = x0
        self.n_restarts = n_restarts
        # objective
        self.fun = fun
        self.jac = jac

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str({'bounds': self.bounds, 'x0': self.X0})

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend: str):
        if backend == 'scipy':
            self._backend = backend
        else:
            raise ValueError('backend must be one of {}'.format(OPT_BACKENDS))

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, solver: str):
        self._min = partial(scipy.optimize.minimize, method=solver)

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds: Bounds):
        if not isinstance(bounds, Bounds):
            raise TypeError('bounds must be a Bounds object.')
        self._bounds = bounds

    @property
    def fun(self):
        return self._fun

    @fun.setter
    def fun(self, fun: Optional[Callable]):
        if callable(fun) or fun is None:
            self._fun = fun
        else:
            raise TypeError('fun must be either callable or none.')

    @property
    def jac(self):
        return self._jac

    @jac.setter
    def jac(self, jac: Union[Callable, bool]):
        if callable(jac) or isinstance(jac, bool) or jac is None:
            self._jac = jac
        else:
            raise TypeError('jac must be callable, bool or none.')

    @property
    def X0(self):
        return self._X0

    @X0.setter
    def X0(self, x0: ndarray):
        if not (isinstance(x0, ndarray) and x0.ndim in (1, 2) and \
                x0.dtype == np.number and np.all(np.isfinite(x0))):
            raise TypeError('x0 must be a 1d or 2d numeric array.')
        X0 = np.atleast_2d(x0)
        for x in X0:
            if not self.bounds.contains(x):
                raise ValueError('x0 {} is outside bounds.'.format(x))
        self._X0 = X0

    @property
    def n_restarts(self):
        return self._n_restarts

    @n_restarts.setter
    def n_restarts(self, n_restarts: int):
        if not isinstance(n_restarts, int):
            raise TypeError('n_restarts must be an integer.')
        if self.X0 is None:
            if n_restarts <= 0:
                raise ValueError( 'n_restarts must be a positive integer ' \
                                  'when x0 is not provided.' )
        else:
            if n_restarts < 0:
                raise ValueError('n_restarts must be a nonnegative integer.')
        self._n_restarts = n_restarts

    def _restart(self):
        if self.n_restarts == 0:
            return
        X = np.random.uniform(low=self.bounds.lowers, high=self.bounds.uppers,
                              size=(self.n_restarts, len(self.bounds)))
        if self.X0 is None:
            self.X0 = X
        else:
            self.X0 = np.vstack([self.X0, X])

    def _check(self):
        if self.fun is None:
            raise AttributeError('function is not set.')
        if self.jac is None:
            raise AttributeError('jacobian is not set.')

    def minimize(self, verbose: bool = False) -> Tuple[bool, float, ndarray]:

        assert isinstance(verbose, bool)
        self._check()  # other attribute must be set by now
        self._restart()

        # FIXME: parallelize
        minimize = lambda x0: self.min(fun=self.fun, jac=self.jac, x0=x0,
                                       bounds=self.bounds.get(self.backend))
        results = [minimize(x0) for x0 in self.X0]

        if verbose:
            print(results)

        b = np.array([res['success'] for res in results])
        X = np.vstack([res['x'] for res in results])
        y = np.array([res['fun'] for res in results])

        if np.any(b):
            if verbose:
                raise NotImplementedError
                # return best trajectory
            else:
                return True, y[b].min(), X[b][y[b].argmin()]
        else:
            if verbose:
                raise NotImplementedError
                # return all trajectory
            else:
                return False, y.min(), X[y.argmin()]

    def maximize(self, verbose: bool = False) -> Tuple[bool, float, ndarray]:
        """ ..todo:: how to flip maximize into minimize"""
        raise NotImplementedError
