import numpy as np
import pandas as pd
from sklearn import preprocessing

def standardize_data(df, columns):
    """Standardize the data using z-score."""

    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(df[columns])

def minmax_scale_data(df, columns):
    """Scale data using MinMax scaling on specified columns."""

    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(df[columns])

def fisher_r_to_z(var):
    """Apply Fisher's r-to-z transform."""

    return np.arctanh(var)

def mean_nonzero(mat, axis=None):
    """Compute the mean of a matrix ignoring zero values."""

    masked_arr = np.ma.mean(np.ma.masked_equal(mat, 0), axis=axis)
    try:
        return masked_arr.filled(0)
    except:
        return masked_arr

def mat2symmetric(mat):
    """Make matrix symmetrical."""
    
    return np.tril(mat) + np.tril(mat).T - np.diag(np.diag(mat))

def zscore(series):
    return (series - series.mean()) / series.std()