"""训练折内预处理。所有统计量均由 fit 的输入计算。"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


class FinancialPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, missing_limit=0.5, correlation_limit=0.95, scale=True):
        self.missing_limit = missing_limit
        self.correlation_limit = correlation_limit
        self.scale = scale

    def fit(self, X, y=None):
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        self.input_columns_ = list(X.columns)
        quality = (X.isna().mean() <= self.missing_limit) & (X.nunique() > 1)
        candidates = list(X.columns[quality])
        corr = X[candidates].corr().abs()
        self.columns_ = []
        self.dropped_correlated_ = []
        for c in candidates:
            linked = [k for k in self.columns_ if corr.loc[c, k] > self.correlation_limit]
            if linked:
                self.dropped_correlated_.append({"feature": c, "retained_feature": linked[0]})
            else:
                self.columns_.append(c)
        self.dropped_quality_ = [c for c in X if c not in candidates]
        X = X[self.columns_]
        self.lower_ = X.quantile(0.01)
        self.upper_ = X.quantile(0.99)
        clipped = X.clip(self.lower_, self.upper_, axis=1)
        self.imputer_ = SimpleImputer(strategy="median").fit(clipped)
        self.scaler_ = RobustScaler().fit(self.imputer_.transform(clipped)) if self.scale else None
        return self

    def transform(self, X):
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)[self.columns_]
        values = self.imputer_.transform(X.clip(self.lower_, self.upper_, axis=1))
        if self.scaler_ is not None:
            values = self.scaler_.transform(values)
        assert np.isfinite(values).all()
        return values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def pipeline(model):
    return Pipeline([("prepare", FinancialPreprocessor()), ("model", model)])


def plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "figure.dpi": 130,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.prop_cycle": plt.cycler(color=["#A9276B", "#23836A", "#D29B33", "#536E9C"])})
    return plt
