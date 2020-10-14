# -*- coding: utf-8 -*-
# density functions

import numpy as np                                                # type: ignore
from abc import ABC, abstractmethod
from numpy import ndarray
from scipy.stats import multivariate_normal                       # type: ignore
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


class Uniform(Density, NormalizedMixedin, SymmetricMixin):

    def __init__(self):
        super().__init__()

    def proposal(self, dst: ndarray, src: ndarray):
        pass

    def sample(self):
        # generate sample
        pass

    def __call__(self, x):
        # evaluate density given input x
        pass


class Gaussian(Density, NormalizedMixedin, SymmetricMixin):

    def __init__(self, mu: ndarray = np.zeros((1,)),
                cov: ndarray = np.ones((1,))):
        super().__init__()
        self.mu = mu
        self.cov =  cov
        self.dst = multivariate_normal(self.mu, self.cov)

    def propose(self, x: ndarray):
        self.mu = x
        self.dst = multivariate_normal(self.mu, self.cov)
        x_star = self.sample(1)
        log_ratio = self._log_ratio(x_star, x)
        return x_star, log_ratio

    def sample(self, size: int = 1):
        return self.dst.rvs(size=size)

    def __call__(self, x: ndarray, log: bool = False):
        if log:
            return self.dst.logpdf(x)
        return self.dst.pdf(x)
