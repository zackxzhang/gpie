# -*- coding: utf-8 -*-
# density functions

import numpy as np                                                # type: ignore
from abc import ABC, abstractmethod
from numpy import ndarray, newaxis
from scipy.stats import multivariate_normal, uniform # type: ignore
from typing import Callable


class Density(ABC):

    @abstractmethod
    def __init__(self):
        """ initialize density object """

    @abstractmethod
    def __call__(self, x):
        """ evaluate density at x """

    def _log_ratio(self, x_star, x):
        """ q(x|x_star) / q(x_star|x) """
        if self.symmetric():
            return 0.
        else:
            raise NotImplementedError


class NormalizedMixedin:
    """ normalized density """

    def normalized(self) -> bool:
        return True


class UnnormalizedMixedin:
    """ unnormalized density """

    def normalized(self) -> bool:
        return False


class SymmetricMixin:
    """ symmetric density """
    def symmetric(self) -> bool:
        return True


class AsymmetricMixin:
    """ asymmetric density """

    def symmetric(self) -> bool:
        return False


class LogDensity(Density, UnnormalizedMixedin, AsymmetricMixin):

    def __init__(self, log_density: Callable):
        super().__init__()
        self.log_dst = log_density

    def __call__(self, x: ndarray):
        return self.log_dst(x)


class Gaussian(Density, NormalizedMixedin, SymmetricMixin):
    """ multivariate Gaussian (normal) density """

    def __init__(self, mu: ndarray = np.zeros((1,)),
                 cov: ndarray = np.ones((1,))):
        super().__init__()
        self.parametrise(mu=mu, cov=cov)

    def parametrise(self, **kwargs):
        if 'mu' in kwargs:
            self.mu = kwargs['mu']
        if 'cov' in kwargs:
            self.cov = kwargs['cov']
        self.dst = multivariate_normal(self.mu, self.cov)

    def __call__(self, x: ndarray, log: bool = False):
        if log:
            return self.dst.logpdf(x)
        else:
            return self.dst.pdf(x)

    def sample(self, size: int = 1):
        return self.dst.rvs(size=size)

    def propose(self, x: ndarray, style: str = 'local'):
        """ sample x_star from q(x_star|x) """
        if style == 'local':
            self.parametrise(mu=x)
        x_star = self.sample(1)
        return x_star, self._log_ratio(x_star, x)


class Uniform(Density, NormalizedMixedin, SymmetricMixin):
    """ independent(!) multivariate uniform density """

    def __init__(self, a: ndarray = np.zeros((1,)),
                b: ndarray = np.ones((1,))):
        super().__init__()
        self.parametrise(a=a, b=b)

    def parametrise(self, **kwargs):
        if 'a' in kwargs:
            self.a = kwargs['a']
        if 'b' in kwargs:
            self.b = kwargs['b']
        self.dst = list()
        for l, s in zip(self.a, self.b):
            self.dst.append(uniform(l, s))

    def __call__(self, x: ndarray, log: bool = False):
        if log:
            log_pdf = 0.
            for xx, dd in zip(x, self.dst):
                log_pdf += dd.logpdf(xx)
            return log_pdf
        else:
            pdf = 1.
            for xx, dd in zip(x, self.dst):
                pdf *= dd.pdf(xx)
            return pdf

    def sample(self, size: int = 1):
        return np.hstack([dd.rvs(size)[:, newaxis] for dd in self.dst])

    def propose(self, x: ndarray, style: str = 'local'):
        raise NotImplementedError
        x_star = self.sample(1)
        return x_star, self._log_ratio(x_star, x)
