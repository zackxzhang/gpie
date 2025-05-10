# -*- coding: utf-8 -*-
# test utilities

import numpy as np                                                # type: ignore
import time
import os
import unittest
from math import exp
from gpie.base import Bounds
from gpie.util import (
    RandomSeed, TimeOut,
    is_array, map_array,
    concat_values, concat_bounds,
    check_X, check_X_Z, check_X_y
)


class UtilTestCase(unittest.TestCase):

    def test_random_seed(self):
        with RandomSeed(42):
            a = np.random.uniform()
        with RandomSeed(42):
            b = np.random.uniform()
        self.assertEqual(a, b)

    def test_time_out(self):
        def f():
            with TimeOut(0.1):
                time.sleep(0.2)
        def g():
            with TimeOut(0.2):
                time.sleep(0.1)
        if os.name == 'posix':
            with self.assertRaises(TimeoutError):
                f()
            try:
                g()
            except TimeoutError:
                self.fail('timeout fails.')

    def test_is_array(self):
        a = np.array([0., 0.])
        b = np.array(['hello', 'world'])
        c = [0, 0.]
        self.assertTrue(is_array(a))
        self.assertFalse(is_array(b))
        self.assertFalse(is_array(c))

    def test_map_array(self):
        a = np.zeros((2,))
        b = np.ones((2,))
        c = map_array(exp, (a, a))
        self.assertTrue((c[0] == b).all())
        self.assertTrue((c[1] == b).all())

    def test_concat_values(self):
        a = np.zeros((2,))
        b = np.zeros((3,))
        c = np.zeros((4,))
        self.assertTrue((concat_values(0., 0.) == a).all())
        self.assertTrue((concat_values(a, 0.) == b).all())
        self.assertTrue((concat_values(0., a) == b).all())
        self.assertTrue((concat_values(a, a) == c).all())

    def test_concat_bounds(self):
        a = np.zeros((2,))
        b = np.zeros((3,))
        c = np.zeros((4,))
        aa = concat_bounds((0., 0.), (0., 0.))
        bb1 = concat_bounds((0., 0.), (a, a))
        bb2 = concat_bounds((a, a), (0., 0.))
        cc = concat_bounds((a, a), (a, a))
        self.assertTrue((aa[0] == a).all())
        self.assertTrue((aa[1] == a).all())
        self.assertTrue((bb1[0] == b).all())
        self.assertTrue((bb1[1] == b).all())
        self.assertTrue((bb2[0] == b).all())
        self.assertTrue((bb2[1] == b).all())
        self.assertTrue((cc[0] == c).all())
        self.assertTrue((cc[1] == c).all())

    def test_check_X(self):
        A = np.zeros((2, 2))
        b = np.zeros((2,))
        C = np.array([['a', 'b'], ['c', 'd']])
        try:
            check_X(A)
        except Exception:
            self.fail('check_X fails.')
        with self.assertRaises(ValueError):
            check_X(b)
        with self.assertRaises(ValueError):
            check_X(C)

    def test_check_X_Z(self):
        A = np.zeros((2, 2))
        B = np.zeros((3, 2))
        C = np.array([['a', 'b'], ['c', 'd']])
        D = np.zeros((2, 3))
        try:
            check_X_Z(A, B)
        except Exception:
            self.fail('check_X_Z fails.')
        with self.assertRaises(ValueError):
            check_X_Z(A, C)
        with self.assertRaises(ValueError):
            check_X_Z(A, D)
        with self.assertRaises(ValueError):
            check_X_Z(B, D)

    def test_check_X_y(self):
        A = np.zeros((2, 2))
        b = np.zeros((2,))
        c = np.zeros((2, 1))
        try:
            check_X_y(A, b)
        except Exception:
            self.fail('check_X_y fails.')
        with self.assertRaises(ValueError):
            check_X_y(A, c)
        with self.assertRaises(ValueError):
            check_X_y(A, A)


if __name__ == '__main__':

    unittest.main()
