# -*- coding: utf-8 -*-
# density functions

import numpy as np                                                # type: ignore
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from numpy import ndarray, newaxis
from scipy.stats import multivariate_normal, uniform              # type: ignore


def verify_density_operands(density_operator):
    """ decorator for overloading density operators """
    @wraps(density_operator)
    def wrapped_density_operator(self, operand):
        if isinstance(operand, Density):
            if self.n_variates == operand.n_variates:
                return density_operator(self, operand)
            else:
                return ValueError('densities must agree on n_variates')
        else:
            raise ValueError('a density operator only accepts two densities.')
    return wrapped_density_operator


class Density(ABC):

    """ unnormalized probability density """

    @abstractmethod
    def __init__(self):
        """ initialize density object """

    @abstractmethod
    def __call__(self, x: ndarray):
        """ evaluate density at x """

    @abstractmethod
    def symmetric(self) -> bool:
        """ symmetric or asymmetric """

    @property
    @abstractmethod
    def n_variates(self) -> int:
        """ number of variates i.e. len(x) """

    def _log_ratio(self, x_star, x):
        """ q(x|x_star) / q(x_star|x) """
        if self.symmetric():
            return 0.
        else:
            raise NotImplementedError

    @verify_density_operands
    def __mul__(self, other):
        if isinstance(self, Distribution) and isinstance(other, Distribution):
            return ProductDistribution(self, other)
        else:
            return ProductDensity(self, other)

    @verify_density_operands
    def __rmul__(self, other):
        if isinstance(other, Distribution) and isinstance(self, Distribution):
            return ProductDistribution(other, self)
        else:
            return ProductDensity(other, self)


class ProductDensity(Density):

    """ product density """

    def __init__(self, d1: Density, d2: Density):
        self._d1 = d1
        self._d2 = d2

    def __call__(self, x: ndarray):
        return self.d1(x) *  self.d2(x)

    @property
    def d1(self):
        return self._d1

    @property
    def d2(self):
        return self._d2

    def symmetric(self) -> bool:
        """ sufficient but not necessary condition for symmetry """
        return self.d1.symmetric() and self.d2.symmetric()

    @property
    def n_variates(self):
        return self.d1.n_variates


class Distribution(Density):

    """ normalized probability density """


class ProductDistribution(Distribution, ProductDensity):

    """ product distribution """


class SymmetricMixin:

    def symmetric(self) -> bool:
        return True


class AsymmetricMixin:

    def symmetric(self) -> bool:
        return False


class LogDensity(AsymmetricMixin, Density):

    def __init__(self, log_density: Callable, n_variates: int):
        self.log_dst = log_density
        self.n_variates = n_variates

    @property
    def n_variates(self):
        return self._n_variates

    @n_variates.setter
    def n_variates(self, n_variates):
        if isinstance(n_variates, int) and n_variates > 0:
            self._n_variates = n_variates
        else:
            raise AttributeError('n_variates has to be a positive integer.')

    def __call__(self, x: ndarray):
        return self.log_dst(x)


class Dirac(SymmetricMixin, Distribution):

    """
    dirac distribution
    spike prior, i.e., fixated, unchangeable belief
    """

    def __init__(self, mu: ndarray = np.zeros(1)):
        self.mu = mu

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Dirac(mu={self.mu})'

    @property
    def n_variates(self):
        return len(self.mu)

    def pdf(self, x: ndarray):
        return 1. if np.allclose(x, self.mu) else 0.

    def __call__(self, x: ndarray):
        return self.pdf(x)

    def sample(self, size: int = 1):
        return np.squeeze(np.tile(self.mu, (size, 1)))


class Flat(SymmetricMixin, Density):

    """
    improper uniform density
    slab prior, i.e., no belief whatsoever
    """

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'flat prior'

    @property
    def n_variates(self) -> int:
        return 0

    def pdf(self, x: ndarray):
        return 1.

    def logpdf(self, x: ndarray):
        return 0.

    def __call__(self, x: ndarray):
        return self.pdf(x)


class Gaussian(SymmetricMixin, Distribution):

    """ Gaussian distribution """

    def __init__(self, mu: ndarray = np.zeros(1),
                 cov: ndarray = np.eye(1), **kwargs):
        self.parametrise(mu=mu, cov=cov, **kwargs)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Gaussian(mu={self.mu}, cov={self.cov})'

    @property
    def n_variates(self):
        return len(self.mu)

    def parametrise(self, **kwargs):
        if 'mu' in kwargs:
            self.mu = kwargs.pop('mu')
        if 'cov' in kwargs:
            self.cov = kwargs.pop('cov')
        self.dst = multivariate_normal(self.mu, self.cov, **kwargs)

    def pdf(self, x: ndarray):
        return self.dst.pdf(x)

    def logpdf(self, x: ndarray):
        return self.dst.logpdf(x)

    def __call__(self, x: ndarray):
        return self.pdf(x)

    def sample(self, size: int = 1):
        return self.dst.rvs(size=size)

    def propose(self, x: ndarray, style: str = 'local'):
        """ sample x_star from q(x_star|x) """
        if style == 'local':
            self.parametrise(mu=x)
        x_star = self.sample(1)
        return x_star, self._log_ratio(x_star, x)


class Student(SymmetricMixin, Distribution):

    """ Student distribution """


class Uniform(SymmetricMixin, Distribution):

    """ independent multivariate uniform distribution """

    def __init__(self, a: ndarray = np.zeros(1), b: ndarray = np.ones(1)):
        self.parametrise(a=a, b=b)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Uniform(a={self.a}, b={self.b})'

    @property
    def n_variates(self):
        return len(self.a)

    def parametrise(self, **kwargs):
        if 'a' in kwargs:
            self.a = kwargs['a']
        if 'b' in kwargs:
            self.b = kwargs['b']
        self.dst = list()
        for l, s in zip(self.a, self.b):
            self.dst.append(uniform(l, s))

    def pdf(self, x: ndarray):
        pdf = 1.
        for xx, dd in zip(x, self.dst):
            pdf *= dd.pdf(xx)
        return pdf

    def logpdf(self, x: ndarray):
        log_pdf = 0.
        for xx, dd in zip(x, self.dst):
            log_pdf += dd.logpdf(xx)
        return log_pdf

    def __call__(self, x: ndarray):
        return self.pdf(x)

    def sample(self, size: int = 1):
        return np.hstack([dd.rvs(size)[:, newaxis] for dd in self.dst])

    def propose(self, x: ndarray, style: str = 'local'):
        raise NotImplementedError
        x_star = self.sample(1)
        return x_star, self._log_ratio(x_star, x)
