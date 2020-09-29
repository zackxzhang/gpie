# -*- coding: utf-8 -*-
# feature engineering

import numpy as np                                                # type: ignore
import pandas as pd                                               # type: ignore
# FIXME: remove pandas dependency

class MinMaxScalerNa:

    def __init__(self, min_: float = 0., max_: float = 1.):
        self.min_ = min_
        self.max_ = max_
        self.df_min = None
        self.df_max = None
        assert self.max_ > self.min_

    def fit(self, df: pd.DataFrame):
        self.df_min = df.min(axis=0, skipna=True)
        self.df_max = df.max(axis=0, skipna=True)

    def transform(self, df: pd.DataFrame):
        assert self.df_min is not None and self.df_max is not None
        df_ = (df - self.df_min)/(self.df_max - self.df_min) \
              * (self.max_ - self.min_) + self.min_
        df_.loc[:, df.columns[~(self.df_min < self.df_max)]] = 0.
        return df_

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)

