# -*- coding: utf-8 -*-
# gaussian process

import numpy as np                                                # type: ignore
import warnings
from collections.abc import Callable
from functools import partial
from math import pi, exp, log, sqrt
from numpy import ndarray
from scipy.linalg import cho_solve, cholesky                      # type: ignore
from scipy.stats import norm, t                                   # type: ignore
from scipy.special import gamma                                   # type: ignore
from ..base import BayesianSupervisedModel
from ..infer import (
    Dirac, Gaussian, Student, LogDensity,
    GradientDescentOptimizer, MarkovChainMonteCarloSampler
)
from ..util import audit_X_Z, audit_X_y, audit_X_y_update, is_array
from .kernels import Kernel, RBFKernel, WhiteKernel


class GaussianProcessRegressor(BayesianSupervisedModel):

    """ Gaussian process regressor """

    inferences = {'exact'}
    solvers = {'l-bfgs-b'}
    aquisitions = {'pi', 'ei', 'lcb'}

    def __init__(
        self,
        kernel: Kernel = 1. * RBFKernel(1.) + 1. * WhiteKernel(),
        inference: str = 'exact',
        solver: str = 'l-bfgs-b'
    ):

        super().__init__()

        if isinstance(kernel, Kernel):
            self._kernel = kernel
        else:
            raise TypeError('kernel must be a Kernel object.')

        if inference in self.inferences:
            self._inference = inference
        else:
            raise ValueError(f'inference must be in {self.inferences}.')

        if solver in self.solvers:
            self._optimizer = GradientDescentOptimizer(
                solver=solver,
                bounds=kernel.thetas.bounds,
                x0=kernel.thetas.values
            )
        else:
            raise ValueError(f'solver must be in {self.solvers}.')

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'GaussianProcessRegressor(kernel={self.kernel})'

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
        def fun(
            kparams: ndarray, grad: bool = True
        ) -> tuple[float, ndarray] | float:
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
            if grad:
                # gradient w.r.t. kernel log parameters
                invK = cho_solve((L, True), I)
                S = invK_y @ invK_y.T - invK
                grad = np.einsum('ij,ijk->k', S, dK) * .5
                return log_mll, np.array(grad)
            else:
                return log_mll
        return fun

    def _acq(self, acquisition: str) -> Callable:
        if not self.fitted():
            raise AttributeError('surrogate model is not fitted yet.')
        # probability of improvement
        def fun_pi(x: ndarray, xi=.01) -> float:
            assert is_array(x, 1, np.number)
            assert isinstance(xi, float) and xi > 0.
            p = self.posterior_predictive(x[np.newaxis, :])
            mu, sigma = p.mu.item(), sqrt(p.cov.item())
            y_min = self.y.min()
            return norm.cdf(y_min, mu + xi, sigma)
        # expected improvement
        def fun_ei(x: ndarray) -> float:
            assert is_array(x, 1, np.number)
            p = self.posterior_predictive(x[np.newaxis, :])
            mu, sigma = p.mu.item(), sqrt(p.cov.item())
            y_min = self.y.min()
            return (
                (y_min - mu) * norm.cdf(y_min, mu, sigma) +
                       sigma * norm.pdf(y_min, mu, sigma)
            )
        # lower confidence bound
        def fun_lcb(x: ndarray, beta: float = 1.) -> float:
            assert is_array(x, 1, np.number)
            assert isinstance(beta, float) and beta > 0.
            p = self.posterior_predictive(x[np.newaxis, :])
            mu, sigma = p.mu.item(), sqrt(p.cov.item())
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
            raise ValueError(f'acquisition must be in {self.aquisitions}.')

    def config(self, x0: ndarray | None = None, n_restarts: int | None = None):
        if x0 is not None:
            self.optimizer.X0 = x0
        if n_restarts is not None:
            self.optimizer.n_restarts = n_restarts

    def hyper_prior(self, n_samples: int = 0):
        super().hyper_prior(n_samples)
        raise NotImplementedError
        if n_samples <= 0:
            ...
        else:
            ...

    def hyper_posterior(self, n_samples: int = 0, **kwargs):
        super().hyper_posterior(n_samples)
        log_mll = partial(self._obj(self.X, self.y), grad=False)
        k = len(self.thetas)
        hyper_posterior = LogDensity(log_mll, k)
        if n_samples <= 0:
            return hyper_posterior
        else:
            sampler = MarkovChainMonteCarloSampler(
                hyper_posterior,
                Gaussian(np.zeros(k), np.eye(k)),
                self.thetas.values, n_samples, **kwargs
            )
            return sampler.sample()
        # only support flat hyperprior θ for now

    def fit(self, X: ndarray, y: ndarray):
        """ empirical bayes, a.k.a. type II maximum likelihood """
        super().fit(X, y)
        result = self.optimizer.maximize(self._obj(self.X, self.y), True)
        success, log_mll, kparams = (result[k] for k in ['success', 'f', 'x'])
        if not success:
            warnings.warn(
                'optimzation fails. try changing x0 or bounds, '
                'or increasing number of restarts.'
            )
        self._set(kparams)
        self.log_mll = log_mll
        # precompute for prediction
        K = self.kernel(self.X, self.X)
        K[np.diag_indices_from(K)] += 1e-8  # jitter
        self.L = cholesky(K, lower=True)
        self.dual_weights = cho_solve((self.L, True), self.y)
        return self

    def predict(self, X: ndarray) -> ndarray:
        super().predict(X)
        Kzx = self.kernel(X, self.X)
        mu = np.einsum('ij,j->i', Kzx, self.dual_weights)
        return mu

    def prior_predictive(
        self, X: ndarray, n_samples: int = 0
    ) -> Gaussian | ndarray:
        super().prior_predictive(X, n_samples)
        mu = np.zeros(len(X))
        cov = self.kernel(X, X)
        prior = Gaussian(mu, cov)
        if n_samples <= 0:
            return prior
        else:
            return prior.sample(n_samples)
        # only support Dirac θ for now

    def posterior_predictive(
        self, X: ndarray, n_samples: int = 0
    ) -> Gaussian | ndarray:
        super().posterior_predictive(X, n_samples)
        Kzx = self.kernel(X, self.X)
        Kzz = self.kernel(X, X)
        mu = np.einsum('ij,j->i', Kzx, self.dual_weights)
        cov = Kzz - Kzx @ cho_solve((self.L, True), Kzx.T)
        # if np.any(cov < 0.):
        #     warnings.warn(
        #         'posterior covariance has negative elements. '
        #         'possibly numerical issues - correcting to 0.'
        #     )
        # cov[cov < 0.] = 0.
        posterior = Gaussian(mu, cov, allow_singular=True)
        if n_samples <= 0:
            return posterior
        else:
            return posterior.sample(n_samples)
        # only support Dirac θ for now


class tProcessRegressor(BayesianSupervisedModel):

    """ t process regressor """

    inferences = {'exact'}
    solvers = {'l-bfgs-b'}
    aquisitions = {'pi', 'ei', 'lcb'}

    def __init__(
        self,
        nu: int = 3,
        kernel: Kernel = 1. * RBFKernel(1.) + 1. * WhiteKernel(),
        inference: str = 'exact',
        solver: str = 'l-bfgs-b'
    ):

        super().__init__()

        if isinstance(nu, int) and nu > 0:
            self._nu = nu
        else:
            raise ValueError(
                'only positive integers are supported for nu, '
                'i.e. degree of freedom for t distribution.'
            )

        if isinstance(kernel, Kernel):
            self._kernel = kernel
        else:
            raise TypeError('kernel must be a Kernel object.')

        if inference in self.inferences:
            self._inference = inference
        else:
            raise ValueError(f'inference must be in {self.inferences}.')

        if solver in self.solvers:
            self._optimizer = GradientDescentOptimizer(
                solver=solver,
                bounds=kernel.thetas.bounds,
                x0=kernel.thetas.values
            )
        else:
            raise ValueError(f'solver must be in {self.solvers}.')

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'tProcessRegressor(nu={self.nu}, kernel={self.kernel})'

    @property
    def nu(self):
        return self._nu

    @property
    def kernel(self):
        return self._kernel

    @property
    def thetas(self):
        return self.kernel.thetas

    @property
    def hyperparameters(self):
        return {'nu': self.nu} + self.kernel.hyperparameters

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
        raise NotImplementedError

    def _acq(self, acquisition: str) -> Callable:
        raise NotImplementedError

    def config(self, x0: ndarray | None = None, n_restarts: int | None = None):
        if x0 is not None:
            self.optimizer.X0 = x0
        if n_restarts is not None:
            self.optimizer.n_restarts = n_restarts

    def hyper_prior(self, n_samples: int = 0):
        super().hyper_prior(n_samples)
        raise NotImplementedError
        if n_samples <= 0:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def hyper_posterior(self, n_samples: int = 0, **kwargs):
        super().hyper_posterior(n_samples)
        log_mll = partial(self._obj(self.X, self.y), grad=False)
        k = len(self.thetas)
        hyper_posterior = LogDensity(log_mll, k)
        if n_samples <= 0:
            return hyper_posterior
        else:
            sampler = MarkovChainMonteCarloSampler(
                hyper_posterior,
                Gaussian(np.zeros(k), np.eye(k)),
                self.thetas.values, n_samples, **kwargs
            )
            return sampler.sample()
        # only support flat hyperprior θ for now

    def fit(self, X: ndarray, y: ndarray):
        raise NotImplementedError

    def predict(self, X: ndarray) -> ndarray:
        raise NotImplementedError

    def prior_predictive(
        self, X: ndarray, n_samples: int = 0
    ) -> Gaussian | ndarray:
        raise NotImplementedError

    def posterior_predictive(
        self, X: ndarray, n_samples: int = 0
    ) -> Gaussian | ndarray:
        raise NotImplementedError
