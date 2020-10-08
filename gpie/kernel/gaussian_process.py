# -*- coding: utf-8 -*-
# gaussian process

import numpy as np                                                # type: ignore
import warnings
from math import pi, exp, log, sqrt
from numpy import ndarray
from scipy.linalg import cho_solve, cholesky                      # type: ignore
from scipy.stats import norm                                      # type: ignore
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union
from .kernels import Kernel, RBFKernel, WhiteKernel
from ..base import BayesianSupervisedModel, Thetas
from ..infer import GradientDescentOptimizer
from ..util import audit_X_Z, audit_X_y, audit_X_y_update, is_array


class GaussianProcessRegressor(BayesianSupervisedModel):

    """
    Gaussian process regressor
    models covariance functions only and not mean functions (e.g. linear trend)
    """

    inferences = {'exact'}
    solvers = {'l-bfgs-b'}
    aquisitions = {'pi', 'ei', 'lcb'}

    def __init__(self, kernel: Kernel = 1. * RBFKernel(1.) + 1. * WhiteKernel(),
                 inference: str = 'exact', solver: str = 'l-bfgs-b'):

        super().__init__()

        if isinstance(kernel, Kernel):
            self._kernel = kernel
        else:
            raise TypeError('kernel must be a Kernel object.')

        if inference in self.inferences:
            self._inference = inference
        else:
            raise ValueError('inference must be in {}.'.format(self.inferences))

        if solver in self.solvers:
            self._optimizer = \
                GradientDescentOptimizer(solver=solver,
                                         bounds=kernel.thetas.bounds,
                                         x0=kernel.thetas.values)
        else:
            raise ValueError('solver must be in {}.'.format(self.solvers))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'GaussianProcessRegressor(kernel={})'.format(self.kernel)

    @property
    def kernel(self):
        return self._kernel

    @property
    def thetas(self):
        return self.kernel.thetas

    @property
    def hyperparameters(self):
        return self.kernel.hyperparameters

    @property
    def inference(self):
        return self._inference

    @property
    def optimizer(self):
        return self._optimizer

    def fitted(self) -> bool:
        return hasattr(self, 'dual_weights')

    def _set(self, kparams: ndarray):
        self.kernel._set(kparams)

    def _obj(self, X: ndarray, y: ndarray) -> Callable:
        Y = y.reshape(-1, 1)
        n = X.shape[0]
        I = np.identity(n)
        kfun = self.kernel._obj(X)
        # log marginal likelihood and its gradient w.r.t. log params
        # ref. GPML algorithm 2.1, equation 5.9
        def fun(kparams: ndarray) -> Tuple[float, ndarray]:
            assert is_array(kparams, 1, np.number)
            K, dK = kfun(kparams)
            K[np.diag_indices_from(K)] += 1e-8  # jitter
            L = cholesky(K, lower=True)
            # dual weights of training points in kernel space
            invK_y = cho_solve((L, True), Y)
            # log marginal likelihood
            log_mll = -.5 * n * log(2. * pi)
            log_mll -= .5 * (Y.T @ invK_y).item()
            log_mll -= np.log(np.diag(L)).sum()
            # gradient w.r.t. kernel log parameters
            invK = cho_solve((L, True), I)
            S = invK_y @ invK_y.T - invK
            grad = np.einsum('ij,ijk->k', S, dK) * .5
            return -log_mll, -np.array(grad)
        return fun

    def _acq(self, acquisition: str) -> Callable:
        if not self.fitted():
            raise AttributeError('surrogate model is not fitted yet.')
        # probability of improvement
        def fun_pi(x: ndarray, xi=.01) -> float:
            assert is_array(x, 1, np.number)
            assert isinstance(xi, float) and xi > 0.
            mu, var = self.predict_prob(x[np.newaxis, :])
            mu, sigma = mu.item(), sqrt(var.item())
            y_min = self.y.min()
            return norm.cdf(y_min, mu + xi, sigma)
        # expected improvement
        def fun_ei(x: ndarray) -> float:
            assert is_array(x, 1, np.number)
            mu, var = self.predict_prob(x[np.newaxis, :])
            mu, sigma = mu.item(), sqrt(var.item())
            y_min = self.y.min()
            return (y_min - mu) * norm.cdf(y_min, mu, sigma) + \
                   sigma**2 * norm.pdf(y_min, mu, sigma)
        # lower confidence bound
        def fun_lcb(x: ndarray, beta: float = 1.) -> float:
            assert is_array(x, 1, np.number)
            assert isinstance(beta, float) and beta > 0.
            mu, var = self.predict_prob(x[np.newaxis, :])
            mu, sigma = mu.item(), sqrt(var.item())
            return mu - beta * sigma
        # entropy search
        def fun_es():
            raise NotImplementedError
        # knowledge gradient
        def fun_kg():
            raise NotImplementedError
        # dispatch
        if acquisition == 'pi':
            return fun_pi
        elif acquisition == 'ei':
            return fun_ei
        elif acquisition == 'lcb':
            return fun_lcb
        else:
            raise ValueError('acquisition must be in {}.'\
                             .format(self.aquisitions))

    def config(self, x0: Optional[ndarray] = None,
                n_restarts: Optional[int] = None):
        if x0 is not None:
            self.optimizer.X0 = x0
        if n_restarts is not None:
            self.optimizer.n_restarts = n_restarts

    def prior(self, *args):
        raise NotImplementedError

    def posterior(self, *args):
        raise NotImplementedError

    def fit(self, X: ndarray, y: ndarray, verbose: bool = False):
        """ MAP estimatem under uniform prior """
        super().fit(X, y)
        self.optimizer.fun = self._obj(self.X, self.y)
        self.optimizer.jac = True
        success, loss, kparams = self.optimizer.minimize(verbose)
        if not success:
            warnings.warn( 'optimzation fails. try changing x0 or bounds, '
                           'or increasing number of restarts.' )
        self._set(kparams)
        self.log_mll = -loss
        # precompute for prediction
        K = self.kernel(self.X, self.X)
        K[np.diag_indices_from(K)] += 1e-8  # jitter
        self.L = cholesky(K, lower=True)
        self.dual_weights = cho_solve((self.L, True), self.y)
        return self

    def predict(self, X: ndarray):
        super().predict(X)
        Kzx = self.kernel(X, self.X)
        mu = np.einsum('ij,j->i', Kzx, self.dual_weights)
        return mu

    def predict_prob(self, X: ndarray):
        super().predict_prob(X)
        Kzx = self.kernel(X, self.X)
        Kzz = self.kernel(X, X)
        mu = np.einsum('ij,j->i', Kzx, self.dual_weights)
        cov = Kzz - Kzx @ cho_solve((self.L, True), Kzx.T)
        if np.any(cov < 0.):
            warnings.warn('posterior covariance matrix has negative elements. '
                          'possibly numerical issues. correcting to 0.')
        cov[cov < 0.] = 0.
        return mu, cov


class tProcessRegressor(BayesianSupervisedModel):

    def __init__(self):
        pass

    def _unpack_params(self, p: ndarray):
        pass

    def _obj(self):
        pass

    def n_params(self):
        pass

    def set_params(self):
        pass

    def fitted(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def update(self):
        pass

    def prior(self):
        pass

    def posterior(self):
        pass

