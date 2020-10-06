# -*- coding: utf-8 -*-
# test base

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

    def test_bounds(self):
        pass

    def test_thetas(self):
        pass


if __name__ == '__main__':
    unittest.main()
