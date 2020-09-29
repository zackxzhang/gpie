# -*- coding: utf-8 -*-
# test inference

import numpy as np                                                # type: ignore
import unittest
from gpie.base import Bounds
from gpie.infer import GradientDescentOptimizer


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


class InferTestCase(unittest.TestCase):

    def test_gradient_descent_optimizer(self):
        b = Bounds(np.array([-4., -4.]), np.array([4., 4.]))
        x = np.array([2.5, 1.])
        try:
            gdo = GradientDescentOptimizer(solver='l-bfgs-b', bounds=b, x0=x)
            gdo.fun = beale
            gdo.jac = False
            print(gdo.minimize())
        except Exception:
            self.fail('gdo fails.')


if __name__ == '__main__':
    unittest.main()
