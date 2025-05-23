# -*- coding: utf-8 -*-
# numerical optimization

import numpy as np                                                # type: ignore
import scipy as sp                                                # type: ignore
import warnings
from collections.abc import Callable
from functools import partial
from multiprocessing import Pool
from numpy import ndarray
from ..base import Optimizer, Bounds


class GradientDescentOptimizer(Optimizer):

    def __init__(
        self,
        bounds: Bounds,
        x0: ndarray,
        n_restarts: int = 0,
        solver: str = 'l-bfgs-b',
    ):
        self.bounds = bounds
        self.X0 = x0
        self.n_restarts = n_restarts
        self.solver = solver

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str({'bounds': self.bounds, 'x0': self.X0})

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds: Bounds):
        if not isinstance(bounds, Bounds):
            raise TypeError('bounds must be a Bounds object.')
        self._bounds = bounds

    @property
    def X0(self):
        return self._X0

    @X0.setter
    def X0(self, x0: ndarray):
        if not (
                isinstance(x0, ndarray)
            and x0.ndim in (1, 2)
            and np.issubdtype(x0.dtype, np.number)
            and np.all(np.isfinite(x0))
        ):
            raise TypeError('x0 must be a 1d or 2d numeric array.')
        X0 = np.atleast_2d(x0)
        for x in X0:
            if not self.bounds.contains(x):
                raise ValueError(f'x0 {x} is outside bounds.')
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
                raise ValueError(
                    'n_restarts must be a positive integer '
                    'when x0 is not provided.'
                )
        else:
            if n_restarts < 0:
                raise ValueError('n_restarts must be a nonnegative integer.')
        self._n_restarts = n_restarts

    def _restart(self):
        if self.n_restarts == 0:
            return
        X = np.random.uniform(
            low=self.bounds.lowers,
            high=self.bounds.uppers,
            size=(self.n_restarts, len(self.bounds)),
        )
        if self.X0 is None:
            self.X0 = X
        else:
            self.X0 = np.vstack([self.X0, X])

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver: str):
        """ check if solver is applicable and options are correctly set """
        if solver == 'l-bfgs-b':
            self._solver = solver
        else:
            raise NotImplementedError

    def _min(self, fun: Callable, jac: Callable | bool, x0: ndarray):
        return sp.optimize.minimize(
            fun=fun, jac=jac, x0=x0,
            bounds=sp.optimize.Bounds(self.bounds.lowers, self.bounds.uppers),
            method=self.solver,
        )

    def minimize(
        self,
        fun: Callable,
        jac: Callable | bool,
        verbose: bool = False,
        callback: Callable | None = None,
    ) -> dict:

        if not callable(fun):
            raise TypeError('fun must be callable.')
        if not (callable(jac) or isinstance(jac, bool)):
            raise TypeError('jac must be callable or boolean.')
        if not isinstance(verbose, bool):
            raise TypeError('verbose must be boolean.')

        self._restart()

        # TODO: parallelize
        results = [self._min(fun, jac, x0) for x0 in self.X0]

        b = np.array([res['success'] for res in results])
        X = np.vstack([res['x'] for res in results])
        y = np.array([res['fun'] for res in results])

        if callback:
            callback(results)
        if verbose:
            return {'success': b, 'f': y, 'x': X}
        if np.any(b):
            return {
                'success': True,
                'f': y[b].min(),
                'x': X[b][y[b].argmin()],
            }
        else:
            return {
                'success': False,
                'f': y.min(),
                'x': X[y.argmin()],
            }

    def maximize(
        self,
        fun: Callable,
        jac: Callable | bool,
        verbose: bool = False,
        callback: Callable | None = None,
    ) -> dict:

        if not callable(fun):
            raise TypeError('fun must be callable.')
        if not (callable(jac) or isinstance(jac, bool)):
            raise TypeError('jac must be callable or boolean.')
        if not isinstance(verbose, bool):
            raise TypeError('verbose must be boolean.')

        self._restart()

        if callable(jac):
            _fun = lambda x: -fun(x)
            _jac = lambda x: -jac(x)
        else:
            _fun = lambda x: tuple(-f for f in fun(x))
            _jac = jac  # type: ignore

        results = [self._min(_fun, _jac, x0) for x0 in self.X0]

        b = np.array([res['success'] for res in results])
        X = np.vstack([res['x'] for res in results])
        y = np.array([res['fun'] for res in results])

        if callback:
            callback(results)
        if verbose:
            return {'success': b, 'f': -y, 'x': X}
        if np.any(b):
            return {
                'success': True,
                'f': -y[b].min(),
                'x': X[b][y[b].argmin()],
            }
        else:
            return {
                'success': False,
                'f': -y.min(),
                'x': X[y.argmin()],
            }
