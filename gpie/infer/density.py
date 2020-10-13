# -*- coding: utf-8 -*-
# density functions

from ..base import Density


class GenericDensity(Density):

    def __init__(self):
        super().__init__()

    # unnormalized
    # cannot generate samples
    # can evaluate density given input


class Gaussian(Density):

    def __init__(self):
        super().__init__()

    # normalized
    # can generate samples
    # can evaluate density given input
