# -*- coding: utf-8 -*-
# test base

import numpy as np                                                # type: ignore
import unittest
from math import pi, exp, log, sqrt
from gpie.base import Bounds, Thetas


class InferTestCase(unittest.TestCase):

    def test_bounds(self):
        pass

    def test_thetas(self):
        a = 1.0
        a_bounds = (1e-4, 1e+4)
        Thetas.from_seq((a,), (a_bounds,), log)


if __name__ == '__main__':
    unittest.main()
