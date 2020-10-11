# -*- coding: utf-8 -*-
# markov chain monte carlo

from ..base import Density, Sampler


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


class MCMCSampler(Sampler):

    def __init__(self):
        super().__init__()


class MatropolisHastingsSampler(MCMCSampler):

    def __init__(self):
        super().__init__()


class HamiltonianMCSampler(MCMCSampler):

    def __init__(self):
        super().__init__()


class NUTSampler(HamiltonianMCSampler):

    def __init__(self):
        super().__init__()
