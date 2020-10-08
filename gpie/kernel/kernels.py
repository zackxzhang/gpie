# -*- coding: utf-8 -*-
# kernel functions

from abc import ABC, abstractmethod
from fractions import Fraction
from functools import wraps
import numpy as np                                                # type: ignore
from math import exp, log, sqrt, pi
from numpy import ndarray, newaxis
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union
from ..base import Model, Bounds, Thetas, Hypers
from ..metric import dist
from ..util import audit_X, audit_X_Z, B, V, is_array, concat_bounds


def check_operand(operator):
    @wraps(operator)
    def wrapped_operator(self, operand):
        if isinstance(operand, Kernel):
            return operator(self, operand)
        elif isinstance(operand, float) and operand > 0.:
            return operator(self, ConstantKernel(operand))
        else:
            raise ValueError('a kernel operator accepts either two kernels '
                             'or a positive float and a kernel as operands.')
    return wrapped_operator


class Kernel(Model):
    """ base class for singleton and composite kernels """

    @property
    @abstractmethod
    def thetas(self):
        """
        learnable hyperparameters and their bounds
        used for model internals
        """

    @property
    @abstractmethod
    def hyperparameters(self):
        """
        fixed and learnable hyperparameters (view of thetas)
        used for printing model specifications
        """

    @abstractmethod
    @audit_X
    def _obj(self, X: ndarray) -> Callable:
        """
        make a objective function
        log thetas -> log likelihood
        """

    @abstractmethod
    def _set(self, values: ndarray):
        """
        dual method of _obj
        assign values for log thetas
        """

    @abstractmethod
    @audit_X_Z
    def __call__(self, X: ndarray, Z: ndarray) -> ndarray:
        """
        computer kernel (covariance) between inputs
        """

    @check_operand
    def __add__(self, other):
        return Sum(self, other)

    @check_operand
    def __radd__(self, other):
        return Sum(other, self)

    @check_operand
    def __mul__(self, other):
        return Product(self, other)

    @check_operand
    def __rmul__(self, other):
        return Product(other, self)

    def __pow__(self, exponent):
        if isinstance(exponent, (int, float)):
            return Exponetiation(self, exponent)
        else:
            raise ValueError('kernel exponentiation only accepts '
                             'an integer or a float as exponent')

    @check_operand
    def __or__(self, other):
        return KroneckerSum(self, other)

    @check_operand
    def __ror__(self, other):
        return KroneckerSum(other, self)

    @check_operand
    def __and__(self, other):
        return KroneckerProduct(self, other)

    @check_operand
    def __rand__(self, other):
        return KroneckerProduct(other, self)


class Sum(Kernel):
    """ elementwise sum operator """

    def __init__(self, k1: Kernel, k2: Kernel):
        self._k1 = k1
        self._k2 = k2
        self._b = len(k1.thetas)  # bifurcation point

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{k1} + {k2}'.format(k1=self.k1, k2=self.k2)

    @property
    def k1(self):
        return self._k1

    @property
    def k2(self):
        return self._k2

    @property
    def b(self):
        return self._b

    @property
    def thetas(self):
        return self.k1.thetas + self.k2.thetas

    @property
    def hyperparameters(self):
        return self.k1.hyperparameters + self.k2.hyperparameters

    def stationary(self) -> bool:
        return self.k1.stationary() and self.k2.stationary()

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self.k1._set(log_params[:self.b])
        self.k2._set(log_params[self.b:])

    def _obj(self, X: ndarray):
        super()._obj(X)
        f1 = self.k1._obj(X)
        f2 = self.k2._obj(X)

        def fun(log_params: ndarray):
            assert len(log_params) == len(self.thetas)
            K1, J1 = f1(log_params[:self.b])
            K2, J2 = f2(log_params[self.b:])
            return K1 + K2, np.dstack((J1, J2))

        return fun

    def __call__(self, X: ndarray, Z: ndarray):
        return self.k1(X, Z) + self.k2(X, Z)


class Product(Kernel):
    """ elementwise product operator"""

    def __init__(self, k1: Kernel, k2: Kernel):
        self._k1 = k1
        self._k2 = k2
        self._b = len(k1.thetas)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{k1} * {k2}'.format(k1=self.k1, k2=self.k2)

    @property
    def k1(self):
        return self._k1

    @property
    def k2(self):
        return self._k2

    @property
    def b(self):
        return self._b

    @property
    def thetas(self):
        return self.k1.thetas + self.k2.thetas

    @property
    def hyperparameters(self):
        return self.k1.hyperparameters + self.k2.hyperparameters

    def stationary(self) -> bool:
        return self.k1.stationary() and self.k2.stationary()

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self.k1._set(log_params[:self.b])
        self.k2._set(log_params[self.b:])

    def _obj(self, X: ndarray):
        super()._obj(X)
        f1 = self.k1._obj(X)
        f2 = self.k2._obj(X)

        def fun(log_params: ndarray):
            K1, J1 = f1(log_params[:self.b])
            K2, J2 = f2(log_params[self.b:])
            return K1 * K2, np.dstack(( np.einsum('ij,ijk->ijk', K2, J1),
                                        np.einsum('ij,ijk->ijk', K1, J2) ))

        return fun

    def __call__(self, X: ndarray, Z: ndarray):
        return self.k1(X, Z) * self.k2(X, Z)


class Exponetiation(Kernel):
    """ elementwise exponentiation operator """

    def __init__(self, k: Kernel, exponent: Union[int, float]):
        self._k = k
        self._e = exponent

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '({k} ** {e:.3g})'.format(k=self.k, e=self.e)

    @property
    def k(self):
        return self._k

    @property
    def e(self):
        return self._e

    @property
    def thetas(self):
        return self.k.thetas

    @property
    def hyperparameters(self):
        return self.k.hyperparameters

    def stationary(self) -> bool:
        return self.k.stationary()

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self.k._set(log_params)

    def _obj(self, X: ndarray):
        super()._obj(X)
        f = self.k._obj(X)

        def fun(log_params: ndarray):
            K, J = f(log_params)
            return K ** self.e , self.e * K ** (self.e - 1.) * J

        return fun

    def __call__(self, X: ndarray, Z: ndarray):
        return self.k(X, Z) ** self.e


class KroneckerSum(Kernel):
    """ Kronecker sum operator """

    def __init__(self, k1: Kernel, k2: Kernel):
        self._k1 = k1
        self._k2 = k2
        self._b = len(k1.thetas)
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '({k1} ⊕ {k2})'.format(k1=self.k1, k2=self.k2)

    @property
    def k1(self):
        return self._k1

    @property
    def k2(self):
        return self._k2

    @property
    def b(self):
        return self._b

    @property
    def thetas(self):
        return self.k1.thetas + self.k2.thetas

    @property
    def hyperparameters(self):
        return self.k1.hyperparameters + self.k2.hyperparameters

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self.k1._set(log_params[:self.b])
        self.k2._set(log_params[self.b:])

    def _obj(self, X: ndarray):
        super()._obj(X)
        X1 = X[:, :self.b]
        X2 = X[:, self.b:]
        f1 = self.k1._obj(X1)
        f2 = self.k2._obj(X2)

        def fun(log_params: ndarray):
            K1, J1 = f1(log_params[:self.b])
            K2, J2 = f2(log_params[self.b:])
            O = np.zeros_like(J1)
            return np.add.outer(K1, K2), \
                   np.dstack((np.add.outer(J1, O), np.add.outer(J2, O)))

        return fun


class KroneckerProduct(Kernel):
    """ Kronecker product operator """

    def __init__(self, k1: Kernel, k2: Kernel):
        self._k1 = k1
        self._k2 = k2
        self._b = len(k1.thetas)
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '({k1} ⊗ {k2})'.format(k1=self.k1, k2=self.k2)

    @property
    def k1(self):
        return self._k1

    @property
    def k2(self):
        return self._k2

    @property
    def b(self):
        return self._b

    @property
    def thetas(self):
        return self.k1.thetas + self.k2.thetas

    @property
    def hyperparameters(self):
        return self.k1.hyperparameters + self.k2.hyperparameters

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self.k1._set(log_params[:self.b])
        self.k2._set(log_params[self.b:])

    def _obj(self, X: ndarray):
        super()._obj(X)
        X1 = X[:, :self.b]
        X2 = X[:, self.b:]
        f1 = self.k1._obj(X1)
        f2 = self.k2._obj(X2)

        def fun(log_params: ndarray):
            K1, J1 = f1(log_params[:self.b])
            K2, J2 = f2(log_params[self.b:])
            return np.kron(K1, K2), \
                   np.dstack((np.kron(K2, J1), np.kron(K1, J2)))

        return fun


class StationaryMixin:
    """ stationary kernel """

    def stationary(self) -> bool:
        return True


class NonStationaryMixin:
    """ stationary kernel """

    def stationary(self) -> bool:
        return False


scalar_formatter = '{0:.3g}'.format
array_formatter = lambda x: '[{}]'.format(', '.join(map(scalar_formatter, x)))


def formatter(x: V):
    if isinstance(x, float):
        return scalar_formatter(x)
    elif isinstance(x, ndarray):
        return array_formatter(x)
    else:
        raise TypeError('unrecognized input for formatter: {}'.format(x))


def check_positive_scalar(x: float, name='amplitude'):
    if isinstance(x, float):
        if x <= 0:
            raise ValueError('{} must be positive.'.format(name))
    else:
        raise TypeError('{} must be a float.'.format(name))


def check_positive_scalar_array(x: V, name='length scale'):
    if isinstance(x, float):
        if x <= 0:
            raise ValueError('{} must be positive.'.format(name))
    elif isinstance(x, ndarray):
        if np.any(x <= 0.):
            raise ValueError('{} must be positive.'.format(name))
    else:
        raise TypeError('{} must be a float or an array.'.format(name))


class ConstantKernel(Kernel, StationaryMixin):
    """
    constant kernel

    k(x,z) = a

    a (amplitude): positive float
    """

    def __init__(self, a: float = 1.0, a_bounds: B = (1e-4, 1e+4)):
        check_positive_scalar(a, 'amplitude')
        self._thetas = Thetas.from_seq((a,), (a_bounds,), log)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{}**2'.format(formatter(sqrt(self.a)))

    @property
    def thetas(self):
        return self._thetas

    @property
    def hyperparameters(self):
        return {'amplitude': self.a}

    @property
    def a(self):
        return exp(self.thetas.values[0])

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self._thetas.set(log_params)

    def _obj(self, X: ndarray) -> Callable:
        super()._obj(X)
        U = np.ones((X.shape[0], X.shape[0]))
        def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
            assert is_array(log_params, 1, np.number)
            a = exp(log_params[0])
            K = U * a
            d_K_d_loga = K[:, :, newaxis]
            return K, d_K_d_loga
        return fun

    def __call__(self, X: ndarray, Z: ndarray) -> ndarray:
        super().__call__(X, Z)
        return self.a * np.ones((X.shape[0], Z.shape[0]))


class WhiteKernel(Kernel, StationaryMixin):
    """
    white noise kernel

    k(x,z) = 1 if x == z else 0
    """

    def __init__(self):
        super().__init__()
        self._thetas = Thetas.from_seq((),())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'WhiteKernel'

    @property
    def thetas(self):
        return self._thetas

    @property
    def hyperparameters(self):
        return {}

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self._thetas.set(log_params)

    def _obj(self, X: ndarray) -> Callable:
        super()._obj(X)
        same = np.isclose(dist(X, X, 'sqeuclidean'), 0.)
        K = np.where(same, 1., 0.)
        dK = np.empty((X.shape[0], X.shape[0], 0))
        def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
            assert is_array(log_params, 1, np.number)
            return K, dK
        return fun

    def __call__(self, X: ndarray, Z: ndarray):
        super().__call__(X, Z)
        same = np.isclose(dist(X, Z, 'sqeuclidean'), 0.)
        return np.where(same, 1., 0.)


class RBFKernel(Kernel, StationaryMixin):
    """
    radial basis function kernel (a.k.a. exponential quadraric kernel)

    k(x,z) = exp( - 1/2 * || (x-z)/l ||**2 )

    l (length scale): positive float (isotropic) or positive array (anisotropic)
    """

    def __init__(self, l: V = 1.0, l_bounds: B = (1e-4, 1e+4)):
        super().__init__()
        check_positive_scalar_array(l, 'length scale')
        self._thetas = Thetas.from_seq((l,), (l_bounds,), log)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        l = formatter(self.l)
        return 'RBFKernel(l={})'.format(l)

    @property
    def thetas(self):
        return self._thetas

    @property
    def hyperparameters(self):
        return {'length scale': self.l}

    @property
    def isotropic(self):
        if len(self.thetas) == 1:
            return True
        else:
            return False

    @property
    def l(self):
        if self.isotropic:
            return exp(self.thetas.values)
        else:
            return np.exp(self.thetas.values)

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self._thetas.set(log_params)

    def _obj(self, X: ndarray) -> Callable:
        super()._obj(X)
        if self.isotropic:
            """ precompute reusable result """
            # pairwise l2 distance squared
            R2 = dist(X, X, metric='sqeuclidean')
            def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
                """ kernel and its jacobian w.r.t. log parameters """
                assert is_array(log_params, 1, np.number)
                # length scale
                l = exp(log_params[0])
                # pairwise scaled l2 distance squared
                R2_l2 = R2 / l**2
                # kernel
                K = np.exp(R2_l2 / -2.)
                # jacobian
                d_K_d_logl = (K * R2_l2)[:, :, newaxis]
                return K, d_K_d_logl
        else:
            """ check dimensions """
            if len(self.l) != X.shape[1]:
                raise ValueError( 'number of features must agree '
                                  'with number of length scales.'  )
            def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
                """ kernel and jacobian carefully engineered for efficiency """
                assert is_array(log_params, 1, np.number)
                # length scale
                l = np.exp(log_params)
                X_l = np.einsum('ij,j->ij', X, 1./l)
                # pairwise scaled difference squared (feature being 3rd axis)
                d_K_d_logl = (X_l[:, newaxis, :] - X_l[newaxis, :, :])**2
                # pairwise scaled l2 distance squared (summing over feature)
                R2_l2 = d_K_d_logl.sum(axis=2)
                # kernel
                K = np.exp(R2_l2 / -2.)
                # jacobian
                d_K_d_logl *= K[:, :, newaxis]
                return K, d_K_d_logl
        return fun

    def __call__(self, X: ndarray, Z: ndarray) -> ndarray:
        """ kernel for readability """
        super().__call__(X, Z)
        k = X.shape[1]  # number of features
        if self.isotropic:
            l = self.l * np.ones((k,))
        else:
            if len(self.l) != k:
                raise ValueError( 'number of features must agree '
                                  'with number of length scales.'  )
            l = self.l
        X_l = np.einsum('ij,j->ij', X, 1./l)
        Z_l = np.einsum('ij,j->ij', Z, 1./l)
        R2_l2 = dist(X_l, Z_l, metric='sqeuclidean')
        K = np.exp(R2_l2 / -2.)
        return K


class RationalQuadraticKernel(Kernel, StationaryMixin):

    """
    rational quadratic kernel (scale mixture of RBF kernels)
    cannot have ard version since the mixture coefficient follows a gamma
    distribution with learnable shape and fixed rate

    k(x,z) = ( 1 + 1/(2*m) * || (x-z)/l ||**2 ) ** -m

    m (mixture coefficient): shape parameter of gamma distribution over
                             RBF's inverse squared length scales l**2 (rate
                             parameter is set to be proportioal to l**2);
                             converges to RBF kernel as m goes to infinity
    l (average length scale): positive float
    """

    def __init__(self, m: float =1.0, l: float = 1.0,
                 m_bounds: B = (1e-4, 1e+4), l_bounds: B = (1e-4, 1e+4)):
        if isinstance(m, float):
            if m <= 0:
                raise ValueError('mixture coefficient must be positive.')
        else:
            raise TypeError('mixture coefficient must be a float.')
        check_positive_scalar_array(l, 'length scale')
        self._thetas = Thetas.from_seq((m, l), (m_bounds, l_bounds), log)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        m = formatter(self.m)
        l = formatter(self.l)
        return 'RationalQuadraticKernel(m={}, l={})'.format(m, l)

    @property
    def thetas(self):
        return self._thetas

    @property
    def hyperparameters(self):
        return {'mixture coefficient': self.m,
                'length scale': self.l}

    @property
    def isotropic(self):
        if len(self.thetas) == 2:
            return True
        else:
            return False

    @property
    def m(self):
        return exp(self.thetas.values[0])

    @property
    def l(self):
        if self.isotropic:
            return exp(self.thetas.values[1])
        else:
            return np.exp(self.thetas.values[1:])

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self._thetas.set(log_params)

    def _obj(self, X: ndarray) -> Callable:
        super()._obj(X)
        if self.isotropic:
            R2 = dist(X, X, metric='sqeuclidean')
            def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
                assert is_array(log_params, 1, np.number)
                # mixture coefficient, length scale
                m = exp(log_params[0])
                l = exp(log_params[1])
                # pairwise scaled L2 distance squared
                R2_l2 = R2 / l**2
                # kernel
                B =  1. + R2_l2 / (2. * m)
                K = B**-m
                # jacobian
                d_K_d_logm = m * K * (1. - 1./B - np.log(B))
                d_K_d_logl = K / B * R2_l2
                return K, np.dstack((d_K_d_logm, d_K_d_logl))
        else:
            if len(self.l) != X.shape[1]:
                raise ValueError( 'number of features must agree '
                                  'with number of length scales.'  )
            def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
                assert is_array(log_params, 1, np.number)
                # length scale
                m = exp(log_params[0])
                l = np.exp(log_params[1:])
                X_l = np.einsum('ij,j->ij', X, 1./l)
                # pairwise scaled difference squared (feature being 3rd axis)
                d_K_d_logl = (X_l[:, newaxis, :] - X_l[newaxis, :, :])**2
                # pairwise scaled distance squared (summing over feature)
                R2_l2 = d_K_d_logl.sum(axis=2)
                # kernel
                B =  1. + R2_l2 / (2. * m)
                K = B**-m
                # jacobian
                d_K_d_logm = m * K * (1. - 1./B - np.log(B))
                d_K_d_logl *= (K / B)[:, :, newaxis]
                return K, np.dstack((d_K_d_logm, d_K_d_logl))
        return fun

    def __call__(self, X: ndarray, Z: ndarray) -> ndarray:
        super().__call__(X, Z)
        k = X.shape[1]  # number of features
        if self.isotropic:
            l = self.l * np.ones((k,))
        else:
            if len(self.l) != k:
                raise ValueError( 'number of features must agree '
                                  'with number of length scales.'  )
            l = self.l
        X_sqrt2ml = np.einsum('ij,j->ij', X, 1./(sqrt(2*self.m)*l))
        Z_sqrt2ml = np.einsum('ij,j->ij', Z, 1./(sqrt(2*self.m)*l))
        R2_2ml2 = dist(X_sqrt2ml, Z_sqrt2ml, metric='sqeuclidean')
        K = (1. + R2_2ml2) ** -self.m
        return K


class MaternKernel(Kernel, StationaryMixin):
    """
    Matérn kernel (d = 1 -> Ornstein–Uhlenbeck kernel)

    k(x,z) = f(sqrt(d) * ||(x-z)/l||) * exp(-sqrt(d) * ||(x-z)/l||)
    f(t) = 1                        d = 1
    f(t) = 1 + t                    d = 3
    f(t) = 1 + t + 1/3 * t**2       d = 5

    d (Bessel order): positive integer, only small d's are interesting
                      since Matérn kernel becomes smooth as d grows large,
                      and converges to RBF kernel as d goes to infinity
    l (length scale): positive float (isotropic) or positive array (anisotropic)
    """

    def __init__(self, d: int = 5, l: float = 1.0, l_bounds: B = (1e-4, 1e+4)):
        super().__init__()
        check_positive_scalar_array(l, 'length scale')
        if d == 1:
            self._f = lambda t: 1.
            self._g = lambda t: 1.  # g = f - f'
        elif d == 3:
            self._f = lambda t: 1. + t
            self._g = lambda t: t
        elif d == 5:
            self._f = lambda t: 1. + t + 1./3. * t**2
            self._g = lambda t: 1./3. * t * (t + 1.)
        else:
            raise ValueError('only 1, 3, 5 are supported for Bessel order'
                             ' because others are difficult to evaluate.'  )
        self._d = d
        self._thetas = Thetas.from_seq((l,), (l_bounds,), log)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        d = Fraction(self.d, 2)
        l = formatter(self.l)
        return 'MatérnKernel(d={}, l={})'.format(d, l)

    @property
    def thetas(self):
        return self._thetas

    @property
    def hyperparameters(self):
        return {'Bessel order': Fraction(self.d, 2),
                'length scale': self.l}
    @property
    def isotropic(self):
        if len(self.thetas) == 1:
            return True
        else:
            return False

    @property
    def d(self):
        return self._d

    @property
    def f(self):
        return self._f

    @property
    def g(self):  # g = f - f'
        return self._g

    @property
    def l(self):
        if self.isotropic:
            return exp(self.thetas.values)
        else:
            return np.exp(self.thetas.values)

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self._thetas.set(log_params)

    def _obj(self, X: ndarray) -> Callable:
        super()._obj(X)
        d = self.d
        f = self.f
        g = self.g
        if self.isotropic:
            R = dist(X, X, metric='euclidean')
            def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
                assert is_array(log_params, 1, np.number)
                # length scale
                l = exp(log_params[0])
                # pairwise scaled distance
                B = R / (l/sqrt(d))
                # kernel
                J = np.exp(-B)
                K = J * f(B)
                # jacobian
                if d == 1:
                    d_K_d_logl = (J * B)[:, :, newaxis]  # g(B) = 1
                else:
                    d_K_d_logl = (J * B * g(B))[:, :, newaxis]
                return K, d_K_d_logl
        else:
            def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
                assert is_array(log_params, 1, np.number)
                # length scale
                l = np.exp(log_params)
                X_l = np.einsum('ij,j->ij', X, 1./l)
                # pairwise scaled difference squared (feature being 3rd axis)
                d_K_d_logl = (X_l[:, newaxis, :] - X_l[newaxis, :, :])**2
                # pairwise scaled distance (summing over feature)
                R = np.sqrt(d_K_d_logl.sum(axis=2))
                # deal with singularity
                I = np.eye(R.shape[0])
                RR = np.reciprocal(R + I) - I
                # kernel
                B = sqrt(d) * R
                J = np.exp(-B)
                K = J * f(B)
                # jacobian
                if d == 1:
                    d_K_d_logl *= (J * RR * sqrt(d))[:, :, newaxis]
                else:
                    d_K_d_logl *= (J * RR * sqrt(d) * g(B))[:, :, newaxis]
                return K, d_K_d_logl
        return fun

    def __call__(self, X: ndarray, Z: ndarray) -> ndarray:
        super().__call__(X, Z)
        k = X.shape[1]  # number of features
        if self.isotropic:
            l = self.l * np.ones((k,))
        else:
            if len(self.l) != k:
                raise ValueError( 'number of features must agree '
                                  'with number of length scales.'  )
            l = self.l
        sqrtdX_l = np.einsum('ij,j->ij', X, sqrt(self.d)/l)
        sqrtdZ_l = np.einsum('ij,j->ij', Z, sqrt(self.d)/l)
        sqrtdR_l = dist(sqrtdX_l, sqrtdZ_l, metric='euclidean')
        K = np.exp(-sqrtdR_l) * self.f(sqrtdR_l)
        return K


class PeriodicKernel(Kernel, StationaryMixin):
    """
    periodic kernel (RBF kernel in u-space)

    k(x,z) = exp( -2 * || sin( π/p * (x-z)) / l ||**2 )

    p (period): positive float (isotropic) or positive array (anisotropic)
                where periods define contours' shape and scale
    l (length scale): positive float (isotropic) or positive array (anisotropic)
    """

    def __init__(self, p: float = 1.0, l: float = 1.0,
                 p_bounds: B = (1e-4, 1e+4), l_bounds: B = (1e-4, 1e+4)):
        check_positive_scalar_array(p, 'period')
        check_positive_scalar_array(l, 'length scale')
        if not (type(p) == type(l) == float or
                isinstance(p, ndarray) and isinstance(l, ndarray) and
                len(p) == len(l)):
            raise ValueError('periods and length scales must be of same size.')
        self._thetas = Thetas.from_seq((p, l), (p_bounds, l_bounds), log)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        p = formatter(self.p)
        l = formatter(self.l)
        return 'PeriodicKernel(p={}, l={})'.format(p, l)

    @property
    def thetas(self):
        return self._thetas

    @property
    def hyperparameters(self):
        return {'period': self.p, 'length scale': self.l}

    @property
    def isotropic(self):
        if len(self.thetas) == 2:
            return True
        else:
            return False

    @property
    def dim(self):
        return len(self.thetas) // 2

    @property
    def p(self):
        if self.isotropic:
            return exp(self.thetas.values[0])
        else:
            return np.exp(self.thetas.values[:self.dim])

    @property
    def l(self):
        if self.isotropic:
            return exp(self.thetas.values[-1])
        else:
            return np.exp(self.thetas.values[-self.dim:])

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self._thetas.set(log_params)

    def _obj(self, X: ndarray) -> Callable:
        if self.isotropic:
            def fun(log_params)-> Tuple[ndarray, ndarray]:
                assert is_array(log_params, 1, np.number)
                # period and length scale
                p = exp(log_params[0])
                l = exp(log_params[1])
                # map to u-space
                X_p = X * (pi/p)
                D_p = X_p[:, newaxis, :] - X_p[newaxis, :, :]
                sinDp = np.sin(D_p)
                # pairwise scaled l2 distance squared
                R2_l2 = ((2./l * sinDp) ** 2).sum(axis=2)
                # kernel
                K = np.exp(R2_l2 / -2.)
                # jacobian
                d_K_d_logp = D_p * np.cos(D_p) * sinDp
                d_K_d_logp = d_K_d_logp.sum(axis=2) * K * (4./l**2)
                d_K_d_logl = K * R2_l2
                return K, np.dstack((d_K_d_logp, d_K_d_logl))
        else:
            """ check dimensions """
            if len(self.p) != X.shape[1]:
                raise ValueError( 'number of features must agree with '
                                  'number of periods / length scales.'  )
            dim = self.dim
            def fun(log_params)-> Tuple[ndarray, ndarray]:
                assert is_array(log_params, 1, np.number)
                # period and length scale
                p = np.exp(log_params[:dim])
                l = np.exp(log_params[-dim:])
                # map to u-space
                X_p = np.einsum('ij,j->ij', X, pi/p)
                D_p = X_p[:, newaxis, :] - X_p[newaxis, :, :]
                sinDp = np.sin(D_p)
                sin2Dp4_l2 = np.einsum('ijk,k->ijk', sinDp, 2./l) ** 2
                # pairwise scaled l2 distance squared
                R2_l2 = sin2Dp4_l2.sum(axis=2)
                # kernel
                K = np.exp(R2_l2 / -2.)
                # jacobian
                d_K_d_logp = K[:, :, newaxis] * D_p * np.cos(D_p) * sinDp
                d_K_d_logp = np.einsum('ijk,k->ijk', d_K_d_logp, 4./l**2)
                d_K_d_logl = K[:, :, newaxis] * sin2Dp4_l2
                return K, np.dstack((d_K_d_logp, d_K_d_logl))
        return fun

    def __call__(self, X: ndarray, Z: ndarray) -> ndarray:
        super().__init__(X, Z)
        k = X.shape[1]  # number of features
        if self.isotropic:
            l = self.l * np.ones((k,))
            p = self.p * np.ones((k,))
        else:
            if len(self.p) != k:
                raise ValueError( 'number of features must agree with '
                                  'number of periods / length scales.'  )
            l = self.l
            p = self.p
        X_p = np.einsum('ij,j->ij', X, pi/p)
        Z_p = np.einsum('ij,j->ij', Z, pi/p)
        sinDp = np.sin(X_p[:, newaxis, :] - Z_p[newaxis, :, :])
        R2_l2 = (np.einsum('ijk,k->ijk', sinDp, 2./l) ** 2).sum(axis=2)
        K = np.exp(R2_l2 / -2.)
        return K


class SpectralKernel(Kernel, StationaryMixin):
    """
    additive component of spectral mixture kernel
    whose spectral densitiy is Gaussian(p, l)

    k(x,z) = exp( - 1/2 * || (x-z)/l ||**2 ) * cos(p.T @ (x-z))

    p (period): positive float (isotropic) or positive array (anisotropic)
                defines contours' shape and scale
    l (length scale): positive float (isotropic) or positive array (anisotropic)
    """

    def __init__(self, p: float = 1.0, l: float = 1.0,
                 p_bounds: B = (1e-4, 1e+4), l_bounds: B = (1e-4, 1e+4)):
        check_positive_scalar_array(p, 'period')
        check_positive_scalar_array(l, 'length scale')
        if not (type(p) == type(l) == float or
                isinstance(p, ndarray) and isinstance(l, ndarray) and
                len(p) == len(l)):
            raise ValueError('periods and length scales must be of same size.')
        self._thetas = Thetas.from_seq((p, l), (p_bounds, l_bounds), log)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        p = formatter(self.p)
        l = formatter(self.l)
        return 'SpectralKernel(p={}, l={})'.format(p, l)

    @property
    def thetas(self):
        return self._thetas

    @property
    def hyperparameters(self):
        return {'period': self.p, 'length scale': self.l}

    @property
    def isotropic(self):
        if len(self.thetas) == 2:
            return True
        else:
            return False

    @property
    def dim(self):
        return len(self.thetas) // 2

    @property
    def p(self):
        if self.isotropic:
            return exp(self.thetas.values[0])
        else:
            return np.exp(self.thetas.values[:self.dim])

    @property
    def l(self):
        if self.isotropic:
            return exp(self.thetas.values[-1])
        else:
            return np.exp(self.thetas.values[-self.dim:])

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self._thetas.set(log_params)

    def _obj(self, X: ndarray) -> Callable:
        if self.isotropic:
            R2 = dist(X, X, metric='sqeuclidean')
            def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
                assert is_array(log_params, 1, np.number)
                # period and length scale
                p = exp(log_params[0])
                l = exp(log_params[1])
                # pairwise scaled l2 distance squared
                R2_l2 = R2 / l**2
                # pairwise difference summed
                D = (X[:, newaxis, :] - X[newaxis, :, :]).sum(axis=2)
                # precompute
                Dp = D * p
                rbf = np.exp(R2_l2 / -2.)
                # kernel
                K = rbf * np.cos(Dp)
                # jacobian
                d_K_d_logp = rbf * -np.sin(Dp) * Dp
                d_K_d_logl = K * R2_l2
                return K, np.dstack((d_K_d_logp, d_K_d_logl))
        else:
            if len(self.l) != X.shape[1]:
                raise ValueError( 'number of features must agree '
                                  'with number of length scales.'  )
            dim = self.dim
            def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
                assert is_array(log_params, 1, np.number)
                # period and length scale
                p = np.exp(log_params[:dim])
                l = np.exp(log_params[-dim:])
                X_l = np.einsum('ij,j->ij', X, 1./l)
                # pairwise scaled difference squared (feature being 3rd axis)
                d_K_d_logl = (X_l[:, newaxis, :] - X_l[newaxis, :, :])**2
                # pairwise scaled l2 distance squared (summing over feature)
                R2_l2 = d_K_d_logl.sum(axis=2)
                # pairwise difference
                D = X[:, newaxis, :] - X[newaxis, :, :]
                # pairwise perioded difference
                d_K_d_logp = D * p[newaxis, newaxis, :]
                # precompute
                Dp = d_K_d_logp.sum(axis=2)
                rbf = np.exp(R2_l2 / -2.)
                # kernel
                K = rbf * np.cos(Dp)
                # jacobian
                d_K_d_logp *= (rbf * -np.sin(Dp))[:, :, newaxis]
                d_K_d_logl *= K[:, :, newaxis]
                return K, np.dstack((d_K_d_logp, d_K_d_logl))
        return fun

    def __call__(self, X: ndarray, Z: ndarray) -> ndarray:
        super().__call__(X, Z)
        k = X.shape[1]  # number of features
        if self.isotropic:
            l = self.l * np.ones((k,))
            p = self.p * np.ones((k,))
        else:
            if len(self.l) != k:
                raise ValueError( 'number of features must agree '
                                  'with number of length scales.'  )
            if len(self.p) != k:
                raise ValueError( 'number of features must agree '
                                  'with number of length scales.'  )
            l = self.l
            p = self.p
        X_l = np.einsum('ij,j->ij', X, 1./l)
        Z_l = np.einsum('ij,j->ij', Z, 1./l)
        R2_l2 = dist(X_l, Z_l, metric='sqeuclidean')
        Rp = ((X[:, newaxis, :] - Z[newaxis, :, :]) * \
               p[newaxis, newaxis, :]).sum(axis=2)
        K = np.exp(R2_l2 / -2.) * np.cos(Rp)
        return K


class LinearKernel(Kernel, NonStationaryMixin):
    """
    linear kernel

    k(x,z) = (x/l).T @ (z/l)

    l (length scale): positive float (isotropic) or positive array (anisotropic)
    """

    def __init__(self, l: V = 1.0, l_bounds: B = (1e-4, 1e+4)):
        super().__init__()
        check_positive_scalar_array(l, 'length scale')
        self._thetas = Thetas.from_seq((l,), (l_bounds,), log)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        l = formatter(self.l)
        return 'LinearKernel(l={})'.format(l)

    @property
    def thetas(self):
        return self._thetas

    @property
    def hyperparameters(self):
        return {'length scale': self.l}

    @property
    def isotropic(self):
        if len(self.thetas) == 1:
            return True
        else:
            return False

    @property
    def l(self):
        if self.isotropic:
            return exp(self.thetas.values)
        else:
            return np.exp(self.thetas.values)

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self._thetas.set(log_params)

    def _obj(self, X: ndarray) -> Callable:
        if self.isotropic:
            # pairwise dot product
            X2 = np.einsum('ik,jk->ij', X, X)
            def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
                assert is_array(log_params, 1, np.number)
                # length scale
                l = exp(log_params[0])
                # kernel
                K = 1. / l**2 * X2
                # jacobian
                d_K_d_logl = (K * -2.)[:, :, newaxis]
                return K, d_K_d_logl
        else:
            if len(self.l) != X.shape[1]:
                raise ValueError( 'number of features must agree '
                                  'with number of length scales.'  )
            def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
                assert is_array(log_params, 1, np.number)
                # length scale
                l = np.exp(log_params)
                X_l = np.einsum('ij,j->ij', X, 1./l)
                # pairwise product (feature being 3rd axis)
                d_K_d_logl = np.einsum('ik,jk->ijk', X_l, X_l)
                # kernel
                K = d_K_d_logl.sum(axis=2)
                # jacobian
                d_K_d_logl *= -2.
                return K, d_K_d_logl
        return fun

    def __call__(self, X: ndarray, Z: ndarray) -> ndarray:
        super().__call__(X, Z)
        k = X.shape[1]  # number of features
        if self.isotropic:
            l = self.l * np.ones((k,))
        else:
            if len(self.l) != k:
                raise ValueError( 'number of features must agree '
                                  'with number of length scales.'  )
            l = self.l
        X_l = np.einsum('ij,j->ij', X, 1./l)
        Z_l = np.einsum('ij,j->ij', Z, 1./l)
        K = np.einsum('ik,jk->ij', X_l, Z_l)
        return K


class NeuralKernel(Kernel, NonStationaryMixin):
    """
    neural kernel

    k(x, z) = 2/pi * arcsin( (x/l).T @ (z/l) + c ) /
              sqrt((||x/l||**2 + c + 1) * (||z/l||**2 + c + 1)) )

    c (intercept deviation): positive float
    l (length scale): positive float (isotropic) or positive array (anisotropic)
    """

    def __init__(self, c: float = 1.0, l: V = 1.0,
                 c_bounds: B = (1e-4, 1e+4), l_bounds: B = (1e-4, 1e+4)):
        check_positive_scalar(c, 'intercept deviation')
        check_positive_scalar_array(l, 'length scale')
        self._thetas = Thetas.from_seq((c, l), (c_bounds, l_bounds), log)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        c = formatter(self.c)
        l = formatter(self.l)
        return 'NeuralKernel(c={}, l={})'.format(c, l)

    @property
    def thetas(self):
        return self._thetas

    @property
    def hyperparameters(self):
        return {'intercept deviation': self.c, 'length scale': self.l}

    @property
    def isotropic(self):
        if len(self.thetas) == 2:
            return True
        else:
            return False

    @property
    def c(self):
        return exp(self.thetas.values[0])

    @property
    def l(self):
        if self.isotropic:
            return exp(self.thetas.values[1])
        else:
            return np.exp(self.thetas.values[1:])

    def fitted(self) -> bool:
        return self.thetas.assigned()

    def _set(self, log_params: ndarray):
        self._thetas.set(log_params)

    def _obj(self, X: ndarray) -> Callable:
        super()._obj(X)
        if self.isotropic:
            # pairwise dot product
            X2 = np.einsum('ik,jk->ij', X, X)
            # l2 norm squared
            n = np.diagonal(X2)
            def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
                assert is_array(log_params, 1, np.number)
                # intercept deviation, length scale
                c = exp(log_params[0])
                l = exp(log_params[1])
                # precompute
                n_lc = n / l**2 + (c + 1.)
                N2_lc2 = np.einsum('i,j->ij', n_lc, n_lc)  # Kronecher product
                N_lc = np.sqrt(N2_lc2)
                # kernel
                J = (X2 / l**2 + c) / N_lc
                K = 2./pi * np.arcsin(J)
                # precompute
                D = N2_lc2 * np.sqrt(1 - J**2)  # denominator of jacobians
                # jacobian
                d_K_d_logc = 2.*c / pi / D * \
                            (N_lc - .5*J*(n_lc[:, newaxis]+n_lc[newaxis, :]))
                d_K_d_logl = 2./pi / D * (N_lc * X2 * (-2./ l**2) +
                             J * ((n[:, newaxis]*n[newaxis, :])*(2./l**4)   +
                                  (n[:, newaxis]+n[newaxis, :])*((c+1)/l**2) ) )
                return K, np.dstack((d_K_d_logc, d_K_d_logl))
        else:
            def fun(log_params: ndarray) -> Tuple[ndarray, ndarray]:
                assert is_array(log_params, 1, np.number)
                # intercept deviation, length scale
                c = exp(log_params[0])
                l = np.exp(log_params[1:])
                # precompute
                X_l = np.einsum('ij,j->ij', X, 1./l)
                # scaled pairwise Hadamard product (feature being 3rd axis)
                W = np.einsum('ik,jk->ijk', X_l, X_l)
                # scaled pairwise dot product
                X2_l2 = W.sum(axis=2)
                # scaled l2 norm squared
                n_l = np.diagonal(X2_l2)
                n_lc = n_l + (c + 1.)
                N2_lc2 = np.einsum('i,j->ij', n_lc, n_lc)  # Kronecher product
                N_lc = np.sqrt(N2_lc2)
                # kernel
                J = (X2_l2 + c) / N_lc
                K =  2./pi * np.arcsin(J)
                # precompute
                D = N2_lc2 * np.sqrt(1 - J**2)  # denominator of jacobians
                # jacobian
                d_K_d_logc = 2.*c / pi / D * \
                            (N_lc - .5*J * (n_lc[:, newaxis]+n_lc[newaxis, :]))
                d_K_d_logl = 2./pi / D[:, :, newaxis] * \
                             ( (-2.*N_lc)[:, :, newaxis] * W +
                               J[:, :, newaxis] *
                               (np.einsum('i,jjk->ijk', n_lc, W) +
                                np.einsum('j,iik->ijk', n_lc, W)   ) )
                return K, np.dstack((d_K_d_logc, d_K_d_logl))
        return fun

    def __call__(self, X: ndarray, Z: ndarray) -> ndarray:
        super().__call__(X, Z)
        k = X.shape[1]  # number of features
        if self.isotropic:
            l = self.l * np.ones((k,))
        else:
            if len(self.l) != k:
                raise ValueError( 'number of features must agree '
                                  'with number of length scales.'  )
            l = self.l
        X_l = np.einsum('ij,j->ij', X, 1./l)
        Z_l = np.einsum('ij,j->ij', Z, 1./l)
        K = ((X_l**2).sum(axis=1) + self.c + 1.)[:, newaxis] * \
            ((Z_l**2).sum(axis=1) + self.c + 1.)[newaxis, :]
        K = (np.einsum('ik,jk->ij', X_l, Z_l) + self.c) / np.sqrt(K)
        K = 2./pi * np.arcsin(K)
        return K
