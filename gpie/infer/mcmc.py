# -*- coding: utf-8 -*-
# markov chain monte carlo

import numpy as np                                                # type: ignore
from abc import ABC, abstractmethod
from multiprocessing import Pool
from numpy import ndarray
from typing import Callable, Optional, Sequence, Tuple, Union, Iterable
from ..base import Sampler
from .densities import Density


class MarkovChainSampler(Sampler):
    """ Markov chain sampler, including MCMC and SA """

    def __init__(self, log_p: Density, q: Density, x0: ndarray,
                 n_samples: int, n_burns: int, n_restarts: int):
        self.log_p = log_p
        self.q = q
        self.X0 = x0
        self.n_samples = n_samples
        self.n_burns = n_burns
        self.n_restarts = n_restarts

    @property
    def log_p(self):
        return self._log_p

    @log_p.setter
    def log_p(self, log_p: Density):
        self._log_p = log_p

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q: Density):
        self._q = q

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n_samples: int):
        if not isinstance(n_samples, int) and n_samples > 0:
            raise TypeError('n_samples must be a positive integer.')
        self._n_samples = n_samples

    @property
    def n_burns(self):
        return self._n_burns

    @n_burns.setter
    def n_burns(self, n_burns: int):
        if not isinstance(n_burns, int) and n_burns > 0:
            raise TypeError('n_burns must be a positive integer.')
        self._n_burns = n_burns

    @property
    def X0(self):
        return self._X0

    @X0.setter
    def X0(self, x0: ndarray):
        if not (isinstance(x0, ndarray) and \
                x0.ndim in (1, 2) and \
                np.issubdtype(x0.dtype, np.number) \
                and np.all(np.isfinite(x0))):
            raise TypeError('x0 must be a 1d or 2d numeric array.')
        X0 = np.atleast_2d(x0)
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
            return False
        X = np.random.normal(0., 1., size=(self.n_restarts, self.X0.shape[1]))
        # TODO: perturb user-provided x0 by multiplication with random noise
        # np.einsum('ij,j->ij', X, self.X0[0])
        # but consider the case with multiple x0's
        self.X0 = np.vstack([self.X0, X])
        return True

    @abstractmethod
    def _sample(self, x0):
        """ sample from one chain """

    def sample(self, verbose: bool = False):
        """ sample from multiple chains """
        if self._restart():
            with Pool(self.n_restarts+1) as pool:
                chains = pool.map(self._sample, self.X0)
            return chains
        else:
            return self._sample(self.X0[0])


class MarkovChainMonteCarloSampler(MarkovChainSampler):
    """ Markov chain Monte Carlo, a.k.a. Metropolis Hastings """
    def __init__(self, log_p: Density, q: Density, x0: ndarray,
                 n_samples: int = 10000, n_burns: int = 2000,
                 n_restarts: int = 0):
        super().__init__(log_p, q, x0, n_samples, n_burns, n_restarts)

    def _sample(self, x0: ndarray):
        # intiailize
        x = x0
        log_u = np.log(np.random.uniform(0, 1, size=(self.n_samples,)))
        chain = np.zeros((self.n_samples, len(x)))
        # Metropolis–Hastings
        for i in range(-self.n_burns, self.n_samples):
            x_star, accept = self.q.propose(x)  # TODO: handle bounds
            accept += self.log_p(x_star) - self.log_p(x)
            if log_u[i] < accept: # accept
                x = x_star
            if i >= 0:
                chain[i] = x
        return chain


class SimulatedAnnealingSampler(MarkovChainSampler):
    """ simulated annealing """

    def __init__(self, log_p: Density, q: Density, x0: ndarray,
                 n_samples: int = 10000, n_burns: int = 2000,
                 n_restarts: int = 0, cooling: Union[str, ndarray] = 'linear'):
        super().__init__(log_p, q, x0, n_samples, n_burns, n_restarts)
        if cooling == 'linear':
            self.cooling = np.linspace(1., 0.1, self.n_burns+self.n_samples)
        else:
            raise NotImplementedError

    def _sample(self, x0: ndarray):
        # intiailize
        x = x0
        log_u = np.log(np.random.uniform(0, 1, size=(self.n_samples,)))
        chain = np.zeros((self.n_samples, len(x)))
        # simulated annealing
        for i in range(-self.n_burns, self.n_samples):
            x_star, accept = self.q.propose(x)
            accept += self.cooling[i] * (self.log_p(x_star) - self.log_p(x))
            if log_u[i] < accept: # accept
                x = x_star
            if i >= 0:
                chain[i] = x
        return chain


class ThermodynamicSASampler(SimulatedAnnealingSampler):
    """ thermodynamic simulated annealing with adaptive cooling schedule """

    def __init__(self):
        super().__init__()


class HamiltonianMCSampler(MarkovChainMonteCarloSampler):
    """ Hamiltonian Monte Carlo with adaptive proposals based on gradient """

    def __init__(self):
        super().__init__()


class NoUTurnHMCSampler(HamiltonianMCSampler):
    """ Hamiltonian Monte Carlo with auto-calibration heuristics """

    def __init__(self):
        super().__init__()
