# -*- coding: utf-8 -*-
# infrastructure

import numpy as np                                                # type: ignore
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from numpy import ndarray
from .util import (
    check_X_update, check_X_y, check_X_y_update,
    is_array, map_array, concat_values, concat_bounds, V, B
)


__all__ = ['Bounds', 'Thetas']


class Bounds:

    """
    parameter bounds
        empty bounds indicate no learnable parameters
        bound values must be finite
    """

    def __init__(
        self,
        lowers: ndarray = np.array([]),
        uppers: ndarray = np.array([]),
    ):
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
        return f'Bounds(lowers={self.lowers}, uppers={self.uppers}'

    def __len__(self):
        return self.lowers.shape[0]

    def __add__(self, other):
        if not isinstance(other, Bounds):
            raise TypeError('Bounds can only be added to other Bounds.')
        return Bounds( np.concatenate([self.lowers, other.lowers]),
                       np.concatenate([self.uppers, other.uppers]) )

    def __radd__(self, other):
        if not isinstance(other, Bounds):
            raise TypeError('Bounds can only be added to other Bounds.')
        return Bounds( np.concatenate([other.lowers, self.lowers]),
                       np.concatenate([other.uppers, self.uppers]) )

    @property
    def lowers(self) -> ndarray:
        return self._lowers

    @property
    def uppers(self) -> ndarray:
        return self._uppers

    def contains(self, values: ndarray) -> bool:
        """ returns true if values are bounded within [lower, upper] """
        if values.shape != self.lowers.shape:
            raise ValueError('dimension of values must agree with bounds.')
        return (
            np.all(self.lowers - 1e-8 <= values) and
            np.all(self.uppers + 1e-8 >= values)
        )

    @classmethod
    def from_seq(cls, bounds: Sequence[B], transform: Callable = lambda x: x):
        return Bounds(*map_array(transform, concat_bounds(*bounds)))


class Thetas:

    """
    parameter container
        empty thetas indicate no learnable parameters
        infinite theta values indicate uninitialized status
    """

    def __init__(
        self,
        values: ndarray = np.array([]),
        bounds: Bounds = Bounds(),
    ):
        if not isinstance(bounds, Bounds):
            raise TypeError('bounds must be Bounds object.')
        self._bounds = bounds
        self.set(values)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Thetas(values={self.values}, bounds={self.bounds})'

    def __len__(self):
        return self.values.shape[0]

    def __add__(self, other):
        if not isinstance(other, Thetas):
            raise TypeError('Thetas can only be added to another Thetas.')
        return Thetas(
            values=np.concatenate([self.values, other.values]),
            bounds=self.bounds + other.bounds,
        )

    def __radd__(self, other):
        if not isinstance(other, Thetas):
            raise TypeError('Thetas can only be added to another Thetas.')
        return Thetas(
            values=np.concatenate([other.values, self.values]),
            bounds=other.bounds + self.bounds,
        )

    @classmethod
    def from_seq(
        cls,
        values: Sequence[V],
        bounds: Sequence[B],
        transform: Callable = lambda x: x
    ):
        """ a convenience method to construct thetas with sequence of values """
        return Thetas(
            values=map_array(transform, concat_values(*values)),
            bounds=Bounds.from_seq(bounds, transform=transform),
        )

    @property
    def values(self) -> ndarray:
        return self._values

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    def set(self, values: ndarray):
        """ set values and check whether bounds are violated """
        if not is_array(values, 1, np.number):
            raise TypeError('values must be 1d numeric array.')
        if not self.bounds.contains(values):
            print('values', values)
            print('bounds', self.bounds)
            raise ValueError(
                'lower bounds must not exceed upper bounds and '
                'each value must obey its lowers/upper bounds.'
            )
        self._values = values

    def assigned(self) -> bool:
        """
        parameters are assigned with values/distribution or not
        special case: zero-size theta returns True
        """
        # TODO: returns True if thetas has None as values but has prior dst
        return np.all(np.isfinite(self.values))


class Hypers:

    """
    a view of learnable and fixed parameters
    """
    # TODO: potentially no longer useful

    def __init__(self, names: ndarray, values: ndarray):
        if isinstance(names, (tuple, list)):
            names = np.array(names)
        if not is_array(names, 1, np.str_):
            raise TypeError('names must be 1d unicode string array.')
        if not is_array(values, 1, np.number):
            raise TypeError('values must be 1d numeric array.')
        if not len(names) == len(values):
            raise ValueError('names and values must be of same length.')
        self._names = names
        self._values = values

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ', '.join([
            f'{name}={value:.3g}'
            for name, value in zip(self.names.flat, self.values.flat)
        ])

    def __add__(self, other):
        if not isinstance(other, Hypers):
            raise TypeError('Hypers can only be added to other Hypers.')
        names  = np.concatenate([self.names, other.names])
        values = np.concatenate([self.values, other.values])
        return Hypers(names, values)

    def __radd__(self, other):
        if not isinstance(other, Hypers):
            raise TypeError('Hypers can only be added to other Hypers.')
        names  = np.concatenate([other.names, self.names])
        values = np.concatenate([other.values, self.values])
        return Hypers(names, values)

    @property
    def names(self):
        return self._names

    @property
    def values(self):
        return self._values


class Model(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """ initialize model object """

    @property
    @abstractmethod
    def thetas(self) -> Thetas:
        """ core parametrisation of the model """

    def parametrised(self) -> bool:
        """ model parameters assigned with values or not """
        return self.thetas.assigned()

    def fitted(self) -> bool:
        """ model fitted on data or not """
        if hasattr(self, 'X'):
            return True
        else:
            return False


class UnsupervisedModel(Model):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def fit(self, X: ndarray):
        self.X = X

    @abstractmethod
    def transform(self, X: ndarray) -> ndarray:
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

    @abstractmethod
    def fit(self, X: ndarray, y: ndarray):
        check_X_y(X, y)
        self.X = X
        self.y =y

    @abstractmethod
    def predict(self, X: ndarray) -> ndarray:
        if not self.parametrised():
            raise AttributeError('model not parametrised yet.')
        if self.fitted():
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
    def hyper_prior(self, n_samples: int = 0):
        """
        prior distribution p(θ)
        if n_samples > 0, returns #n_samples θ's (vertically stacked)
        if n_samples = 0, returns a distribution object
        """
        # TODO: check if thetas has prior distributions or just a point value
        if self.fitted():
            raise AttributeError(
                'model already fitted. please refer to model '
                'init stage when hyper prior is set by user.'
            )
        # compute hyper prior

    @abstractmethod
    def hyper_posterior(self, n_samples: int = 0):
        """
        posterior distribution p(θ|Xo,yo)
        if n_samples > 0, returns #n_samples θ's (vertically stacked)
        if n_samples = 0, returns a distribution object
        """
        if not self.fitted():
            raise AttributeError('model not fitted yet.')
        # compute hyper posterior

    @abstractmethod
    def prior_predictive(self, X: ndarray, n_samples: int = 0):
        """
        prior distribution p(y|X,θ)
        if n_samples > 0, returns #n_samples y's (vertically stacked)
        if n_samples = 0, returns a distribution object
        """
        if not self.parametrised():
            raise AttributeError('model not parametrised yet.')
        if self.fitted():
            check_X_update(self.X, X)
        # compute prior predictive

    @abstractmethod
    def posterior_predictive(self, X: ndarray, n_samples: int = 0):
        """
        posterior distribution p(y|X,Xo,yo,θ)
        if n_samples > 0, returns #n_samples y's (vertically stacked)
        if n_samples = 0, returns a distribution object
        """
        if not self.parametrised():
            raise AttributeError('model not parametrised yet.')
        if self.fitted():
            check_X_update(self.X, X)
        else:
            raise AttributeError('model not fitted yet.')
        # compute posterior predictive


class OnlineSupervisedMixin(SupervisedModel):

    @abstractmethod
    def update(self, X: ndarray, y: ndarray):
        if not self.fitted():
            raise AttributeError('update after model is fitted first.')
        check_X_y_update(X, y, self.X, self.y)
        self.X = np.vstack((self.X, X))
        self.y = np.append(self.y, y)


class Optimizer(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def minimize(self, *args, **kwargs):
        pass


class Sampler(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def sample(self):
        pass
