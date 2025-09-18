import numpy as np
import scipy.stats as stats

def mean(data, axis=None):
    return np.mean(data, axis=axis)

def median(data, axis=None):
    return np.median(data, axis=axis)

def mode(data, axis=None):
    mode_res = stats.mode(data, nan_policy='omit', axis=axis, keepdims=True)
    if axis is None:
        return mode_res.mode[0] if mode_res.count[0] > 0 else None
    return mode_res.mode[0]

def variance(data, ddof=1, axis=None):
    return np.var(data, ddof=ddof, axis=axis)

def std_dev(data, ddof=1, axis=None):
    return np.std(data, ddof=ddof, axis=axis)

def skewness(data, axis=None):
    return stats.skew(data, nan_policy='omit', axis=axis)

def kurtosis(data, axis=None):
    return stats.kurtosis(data, nan_policy='omit', axis=axis)

def min_value(data, axis=None):
    return np.min(data, axis=axis)

def max_value(data, axis=None):
    return np.max(data, axis=axis)

def quantiles(data, q=[0.25, 0.5, 0.75], axis=None):
    return np.quantile(data, q, axis=axis)

def range_value(data, axis=None):
    return np.ptp(data, axis=axis)
