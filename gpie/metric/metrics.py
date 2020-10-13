# -*- coding: utf-8 -*-
# distance functions

from scipy.spatial.distance import cdist, pdist, squareform       # type: ignore


def dist(X, Z, metric='euclidean'):
    if X is Z:
        return squareform(pdist(X, metric=metric))
    else:
        return cdist(X, Z, metric=metric)
