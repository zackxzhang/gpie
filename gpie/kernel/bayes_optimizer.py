# -*- coding: utf-8 -*-
# bayesian optimizer

import numpy as np                                                # type: ignore
import scipy                                                      # type: ignore
import warnings
from time import time
from numpy import ndarray                                         # type: ignore
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union
from .gaussian_process import GaussianProcessRegressor, tProcessRegressor
from ..base import Bounds, Optimizer
from ..infer import GradientDescentOptimizer
from ..metric import dist
from ..util import is_array


Surrogate = Union[GaussianProcessRegressor, tProcessRegressor]
surrogates = (GaussianProcessRegressor, tProcessRegressor)


class BayesianOptimizer(Optimizer):
    """
    surrogate models blackbox primal objective in a probablistic perspective
    acquisition evaluates input space
    """
    solvers = {'l-bfgs-b'}
    acquisitions = {'pi', 'ei', 'lcb'}

    def __init__(self, fun: Callable, bounds: Bounds,
                 x0: ndarray, y0: Optional[ndarray] = None,
                 n_evals: int = 100, timeout: int = 600,
                 surrogate: Surrogate = GaussianProcessRegressor(),
                 acquisition: str = 'ei', solver='l-bfgs-b'):

        super().__init__()

        if isinstance(surrogate, surrogates):
            self._surrogate = surrogate
        else:
            raise TypeError('surrogate must be a Surrogate object.')

        if acquisition in self.acquisitions:
            self._acquisition = acquisition
        else:
            raise ValueError( 'acquisition must be in {}.' \
                              .format(self.acquisitions)  )

        if solver in self.solvers:
            self._optimizer = GradientDescentOptimizer(solver=solver, x0=x0,
                                                       bounds=bounds, jac=False)
        else:
            raise ValueError('solver must be in {}.'.format(self.solvers))

        self._X = self.optimizer.X0

        if y0 is None:
            self._y = None
        elif is_array(y0, 1, np.number) and y0.shape[0] == x0.shape[0]:
            self._y = y0
        else:
            raise ValueError( 'y0 is either None or a 1d numeric array '
                              'that agree with x0 on 1st dimension.'    )

        if callable(fun):
            self._fun = fun
        else:
            raise TypeError('fun must be callable.')

        if isinstance(n_evals, int) and n_evals > self.optimizer.X0.shape[0]:
            self._n_evals = n_evals
        else:
            raise ValueError("n_evals must be greater than number of x0's.")

        if isinstance(timeout, (int, float)) and timeout > 0:
            self._timeout = timeout
        else:
            raise ValueError('timeout must be a positive number.')

    @property
    def surrogate(self):
        return self._surrogate

    @property
    def acquisition(self):
        return self._acquisition

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def bounds(self):
        return self.optimizer.bounds

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def fun(self):
        return self._fun

    @property
    def n_evals(self):
        return self._n_evals

    @property
    def timeout(self):
        return self._timeout

    def _fit(self):
        # query
        if self.y is None:
            self._y = np.array([self.fun(x0) for x0 in self.X])
        # fit surrogate based on queries
        self.surrogate.fit(self.X, self.y)

    def _update(self):
        # redraw acquisition surface based on updated surrogate
        self.optimizer.fun = self.surrogate._acq(self.acquisition)
        # start searches
        # a) from vicinity of queries (exploitation)
        # b) and from uniformly random states (exploration)
        self.optimizer.X0 = self.X
        # acquisition drawn with ampler information deserves more restarts
        self.optimizer.n_restarts = min(self.X.shape[0], 10**(self.X.shape[1]))
        # choose next query to be the minimizer of acquisition
        success = False
        try:
            success, loss, x = self.optimizer.minimize()
            success &= np.all(dist(np.atleast_2d(x), self.X) > 1e-5)
        except ValueError:
            # numerical instability, e.g. minimize calls function with NaN
            warnings.warn('acquisition optimization fails.')
        finally:
            # turn to pure exploration if
            # a) acuisition minimization fails
            # b) aquisition minimizer is a duplicate
            warnings.warn('turn to pure exploration.')
            if not success:
                x = np.random.uniform(low=self.bounds.lowers,
                                      high=self.bounds.uppers,
                                      size=(len(self.bounds,)))
        # query
        y = self.fun(x)
        # reshape
        X = np.atleast_2d(x)
        y = np.atleast_1d(y)
        # update queries
        self._X = np.vstack([self.X, X])
        self._y = np.append(self.y, y)
        # update surrogate
        self.surrogate.fit(self.X, self.y)

    def minimize(self, verbose: bool = False) -> Tuple[float, ndarray]:
        genesis = time()

        if self.optimizer.bounds.clamped():
            warnings.warn( 'no optimization performed '
                           'since parameters are fixed.' )
        else:
            self._fit()
            while len(self.y) < self.n_evals:
                if time() - genesis > self.timeout:
                    break
                self._update()

        if verbose:
            return self.y, self.X
        else:
            return self.y.min(), self.X[self.y.argmin()]
