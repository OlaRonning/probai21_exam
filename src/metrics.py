import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred, **kwargs):
    """ Root Mean Square Error """
    return mean_squared_error(y_true, y_pred, squared=False, **kwargs)


def pi(post_pred, interval):
    """ Predictive Interval """
    # TODO: may need to be exact
    assert 0. <= interval <= 1.
    lower_bound = (1. - interval) / 2
    upper_bound = 1. - lower_bound
    lower_quantile, upper_quantile = np.quantile(post_pred, [lower_bound, upper_bound], axis=1)  # TODO: CHECK ME
    return lower_quantile, upper_quantile


def picp(post_pred, interval):
    r""" Prediction Interval Coverage Probability is $1/n \sum 1_{y_{L_i} \leq y_i \leq y_{U_i}$ where
    $P(y_{L_i} \leq y_i \leq y_{U_i}) \geq \gamma$. """
    lower_quantile, upper_quantile = pi(post_pred, interval)
    coverage = lower_quantile <= post_pred <= upper_quantile
    return np.mean(coverage)


def mpiw(post_pred, interval):
    r""" Mean Prediction Interval Width is $1/n \sum y_{U_i} - y_{L_i}$ """
    lower_quantile, upper_quantile = pi(post_pred, interval)
    return np.mean(upper_quantile - lower_quantile)
