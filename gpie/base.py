# -*- coding: utf-8 -*-
# model skeletons

import numpy as np                                                # type: ignore
import scipy                                                      # type: ignore
import warnings
from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Callable, Optional, Sequence, Tuple, Type, Union, Iterable
from .util import check_X_update, check_X_y, check_X_y_update, OPT_API, \
                  is_array, map_array, concat_values, concat_bounds, V, B


class Bounds:
    """
    Thetas' component
    bounds must be finite
    """
    def __init__(self, lowers: ndarray, uppers: ndarray):
        if not is_array(lowers, 1, np.number):
            raise TypeError('lower bounds must be 1d numeric array.')
        if not is_array(uppers, 1, np.number):
            raise TypeError('upper bounds must be 1d numeric array.')
        if len(lowers) != len(uppers):
            raise ValueError('lower/upper bounds must be of same length.')
        if not (np.all(np.isfinite(lowers)) and np.all(np.isfinite(uppers))):
            raise ValueError('lower/upper bounds must be finite.')
        self._lowers = lowers
        self._uppers = uppers

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Bounds(lowers={}, uppers={}'.format(self.lowers, self.uppers)

    def __len__(self):
        return self.lowers.shape[0]

    def __add__(self, other):
        if not isinstance(other, Bounds):
            raise TypeError('Bounds can only be added to other Bounds.')
        return Bounds( np.concatenate([self.lowers, other.lowers]),
                       np.concatenate([self.uppers, other.uppers]) )

    @property
    def lowers(self):
        return self._lowers

    @property
    def uppers(self):
        return self._uppers

    def contains(self, values: ndarray, tol: float = 1e-6) -> bool:
        if values.shape != self.lowers.shape:
            raise ValueError('dimension of values must agree with bounds.')
        return np.all(self.lowers - tol <= values) and \
               np.all(values <= self.uppers + tol)

    def clamped(self) -> bool:
        if np.allclose(self.lowers, self.uppers):
            return True
        else:
            return False

    def get(self, api: str = 'scipy'):
        if api == 'scipy':
            return scipy.optimize.Bounds(self.lowers, self.uppers)
        else:
            raise ValueError('inteface must be one of {}'.format(OPT_API))

    @classmethod
    def from_seq(cls, bounds: Sequence[B],
                    transform: Callable = lambda x: x):
        return Bounds(*map_array(transform, concat_bounds(*bounds)))


class Thetas:
    """
    parameterization of models
    values can be nan or infinite to indicate uninitialized status
    """

    def __init__(self, values: ndarray, bounds: Bounds):
        if not isinstance(bounds, Bounds):
            raise TypeError('bounds must be Bounds object.')
        self._bounds = bounds
        self.set(values)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Thetas(values={}, bounds={})'.format(self.values, self.bounds)

    def __len__(self):
        return self.values.shape[0]

    def __add__(self, other):
        if not isinstance(other, Thetas):
            raise TypeError('Thetas can only be added to another Thetas.')
        return Thetas( np.concatenate([self.values, other.values]),
                       self.bounds + other.bounds                   )

    @classmethod
    def from_seq(cls, values: Sequence[V], bounds: Sequence[B],
                 transform: Callable):
        return Thetas(values=map_array(transform, concat_values(*values)),
                      bounds=Bounds.from_seq(bounds, transform=transform))

    @property
    def values(self):
        return self._values

    @property
    def bounds(self):
        return self._bounds

    def assigned(self) -> bool:
        return np.all(np.isfinite(self.values))

    def get(self):
        return self.values, self.bounds

    def set(self, values: ndarray):
        if not is_array(values, 1, np.number):
            raise TypeError('values must be 1d numeric array.')
        if not self.bounds.contains(values):
            raise ValueError( 'lower bounds must not exceed upper bounds and ',
                              'each value must obey its lowers/upper bounds.'  )
        self._values = values


class Hypers:
    """ a view of learnable parameters and fixed parameters """

    def __init__(self, names: ndarray, values: ndarray):
        if isinstance(names, (tuple, list)):
            names = np.array(names)
        if not is_array(names, 1, np.str_):
            raise TypeError('names must be 1d unicode array.')
        if not is_array(values, 1, np.number):
            raise TypeError('values must be 1d numeric array.')
        if not len(names) == len(values):
            raise ValueError('names and values must be of same length.')
        self._names = names
        self._values = values

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        out = ['{name}={value:.3g}'.format(name=name, value=value)
               for name, value in zip(self.names.flat, self.values.flat)]
        return ', '.join(out)

    @property
    def names(self):
        return self._names

    @property
    def values(self):
        return self._values

    def __add__(self, other):
        if not isinstance(other, Hypers):
            raise TypeError('Hypers can only be added to other Hypers.')
        names = np.concatenate([self.names, other.names])
        values = np.concatenate([self.values, other.values])
        return Hypers(names, values)


class Model(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """ initialize model object """

    @abstractmethod
    def fitted(self) -> bool:
        """ model parameters are assigned or not """


class UnsupervisedModel(Model):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.X = None

    @abstractmethod
    def fit(self, X: ndarray):
        self.X = X

    @abstractmethod
    def transform(self, X: ndarray):
        if not self.fitted():
            raise AttributeError('model is not fitted yet.')
        check_X_update(self.X, X)

    @abstractmethod
    def _obj(self, X: ndarray) -> Callable:
        """
        loss/likelihood, jacobian (optional), hessian (optional)
        used for optimizing model parameters in model's fit method
        callable by external optimization routines
        """


class SupervisedModel(Model):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.X = None
        self.y = None

    @abstractmethod
    def fit(self, X: ndarray, y: ndarray, verbose: bool = False):
        check_X_y(X, y)
        self.X = X
        self.y =y

    @abstractmethod
    def predict(self, X: ndarray):
        if not self.fitted():
            raise AttributeError('model is not fitted yet.')
        if self.X is not None:
            check_X_update(self.X, X)

    @abstractmethod
    def _obj(self, X: ndarray, y: ndarray) -> Callable:
        """
        loss/likelihood, jacobian (optional), hessian (optional)
        used for optimizing model parameters in model's fit method
        callable by external optimization routines
        """


class BayesianSupervisedModel(SupervisedModel):

    @abstractmethod
    def prior(self, *args):
        """ set prior """

    @abstractmethod
    def posterior(self, *args):
        """ compute posterior """

    @abstractmethod
    def predict_prob(self, X: ndarray):
        if not self.fitted():
            raise AttributeError('model is not fitted yet.')
        check_X_update(self.X, X)


class OnlineSupervisedMixin(SupervisedModel):

    @abstractmethod
    def update(self, X: ndarray, y: ndarray):
        if not self.fitted():
            raise ValueError('update after model is fitted first.')
        check_X_y_update(X, y, self.X, self.y)
        self.X = np.vstack((self.X, X))
        self.y = np.append(self.y, y)


class Optimizer(ABC):

    @abstractmethod
    def __init__(self):
        """ initialize optimizer object """


class Sampler(ABC):

    @abstractmethod
    def __init__(self):
        """ initialize sampler object """

