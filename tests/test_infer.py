# -*- coding: utf-8 -*-
# test inference

import numpy as np                                                # type: ignore
import unittest
from math import exp, log
from gpie.base import Bounds
from gpie.infer import GradientDescentOptimizer, \
                       LogDensity, Gaussian, MatropolisHastingsSampler


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
    return log(0.3 * exp(-0.2 * x**2) + 0.7 * exp(-0.2 * (x-10.) **2))


class InferTestCase(unittest.TestCase):

    def test_grad_opt(self):
        b = Bounds(np.array([-4., -4.]), np.array([4., 4.]))
        x = np.array([2.5, 1.])
        try:
            gdo = GradientDescentOptimizer(solver='l-bfgs-b', bounds=b, x0=x)
            gdo.fun = beale
            gdo.jac = False
            print(gdo.minimize())
        except Exception:
            self.fail('gradient optimizer fails.')

    def test_metropolis(self):
        try:
            mhs1 = MatropolisHastingsSampler(LogDensity(log_p), Gaussian(),
                                            np.zeros((1,)), n_restarts=0)
            chain = mhs1.sample()
            print(chain)
            mhs2 = MatropolisHastingsSampler(LogDensity(log_p), Gaussian(),
                                            np.zeros((1,)), n_restarts=2)
            chains = mhs2.sample()
            print(chains)
        except Exception:
            self.fail('Metropolis sampler fails.')


if __name__ == '__main__':
    unittest.main()
