# -*- coding: utf-8 -*-
# test inference

import numpy as np                                                # type: ignore
import unittest
from math import exp, log
from gpie.base import Bounds
from gpie.infer import LogDensity, Gaussian, GradientDescentOptimizer, \
                       MarkovChainMonteCarloSampler, SimulatedAnnealingSampler


def beale(x1_x2) -> float:
    """
    smooth with edges sticking up
    f_min = 0.
    x_min = (3.0, 0.5)
    """
    x1, x2 = x1_x2[0], x1_x2[1]
    return (1.5 - x1 + x1 * x2) ** 2 +\
           (2.25 - x1 + x1 * x2**2) ** 2 +\
           (2.625 - x1 + x1 * x2**3) ** 2


def log_p(x):
    """ unnormalized, bimodal density """
    return log(0.3 * exp(-0.2 * x**2) + 0.7 * exp(-0.2 * (x-10.) **2))


class InferTestCase(unittest.TestCase):

    def test_grad_opt(self):
        b = Bounds(np.array([-4., -4.]), np.array([4., 4.]))
        x = np.array([2.5, 1.])
        try:
            gdo = GradientDescentOptimizer(solver='l-bfgs-b', bounds=b, x0=x)
            print(gdo.minimize(beale, False))
        except Exception:
            self.fail('gradient descent optimizer fails.')

    def test_mcmc(self):
        try:
            mhs1 = MarkovChainMonteCarloSampler(LogDensity(log_p,1), Gaussian(),
                                                np.zeros((1,)), n_restarts=0)
            chain = mhs1.sample()
            print(chain)
            mhs2 = MarkovChainMonteCarloSampler(LogDensity(log_p,1), Gaussian(),
                                                np.zeros((1,)), n_restarts=2)
            chains = mhs2.sample()
            print(chains)
        except Exception:
            self.fail('Metropolis Hastings sampler fails.')

    def test_sa(self):
        try:
            sa1 = SimulatedAnnealingSampler(LogDensity(log_p,1), Gaussian(),
                                            np.zeros((1,)), n_restarts=0)
            chain = sa1.sample()
            print(chain)
            sa2 = SimulatedAnnealingSampler(LogDensity(log_p,1), Gaussian(),
                                            np.zeros((1,)), n_restarts=2)
            chains = sa2.sample()
            print(chains)
        except Exception:
            self.fail('simulated annealing sampler fails.')


if __name__ == '__main__':

    unittest.main()
