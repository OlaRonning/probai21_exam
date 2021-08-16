import numpy as np
import torch
from sklearn.metrics import mean_squared_error


def to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().numpy()
    return arr


def rmse(y_true, y_pred, **kwargs):
    """ Root Mean Square Error """
    y_true, y_pred = map(to_numpy, (y_true, y_pred))
    return mean_squared_error(y_true, y_pred, squared=False, **kwargs)


def pi(y_pred, interval):
    """ Predictive Interval """
    # TODO: may need to be exact
    assert 0. <= interval <= 1.
    lower_bound = (1. - interval) / 2
    upper_bound = 1. - lower_bound
    lower_quantile, upper_quantile = np.quantile(y_pred, [lower_bound, upper_bound], axis=0)
    return lower_quantile, upper_quantile


def picp(y_true, y_pred, interval=.95):
    r""" Prediction Interval Coverage Probability is $1/n \sum 1_{y_{L_i} \leq y_i \leq y_{U_i}$ where
    $P(y_{L_i} \leq y_i \leq y_{U_i}) \geq \gamma$. """
    y_true, y_pred = map(to_numpy, (y_true, y_pred))
    lower_quantile, upper_quantile = pi(y_pred, interval)
    coverage = np.bitwise_and(lower_quantile <= y_true, y_true <= upper_quantile)
    return np.mean(coverage)


def mpiw(y_pred, interval=.95):
    r""" Mean Prediction Interval Width is $1/n \sum y_{U_i} - y_{L_i}$ """
    y_pred = to_numpy(y_pred)
    lower_quantile, upper_quantile = pi(y_pred, interval)
    return np.mean(upper_quantile - lower_quantile)
