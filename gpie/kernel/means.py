# -*- coding: utf-8 -*-
# mean functions

from abc import ABC, abstractmethod
from functools import wraps
import numpy as np                                                # type: ignore
from math import exp, log, sqrt, pi
from numpy import ndarray
from ..base import Model, Bounds, Thetas, Hypers
from ..metric import dist
from ..util import audit_X, audit_X_Z, B, V, is_array, concat_bounds


class ConstantMean:
    """ constant mean """


class LinearMean:
    """ linear mean """
