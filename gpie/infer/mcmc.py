# -*- coding: utf-8 -*-
# markov chain monte carlo

from ..base import Sampler


class MCMCSampler(Sampler):

    def __init__(self):
        super().__init__()


class MatropolisHastingsSampler(MCMCSampler):

    def __init__(self):
        super().__init__()


class HamiltonianSampler(MCMCSampler):

    def __init__(self):
        super().__init__()


class NoUTurnSampler(HamiltonianSampler):

    def __init__(self):
        super().__init__()
