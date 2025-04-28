import os
import pickle
import numpy as np
import pandas as pd
from scipy.io import loadmat

def save_to_pickle(data, filename):
    """Save data as a pickle file."""

    filepath = os.path.abspath(filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    return filepath

def read_pickle(filepath):
    """Import data from a pickle file."""

    with open(filepath, 'rb') as f:
        var = pickle.load(f)
    return var

def read_mat(filepath):
    """Import data from a Matlab file."""
    
    var = loadmat(filepath)
    return list(var.values())[3]

def read_csv2npy(filepath):
    """Import data from a CSV file."""

    var = pd.read_csv(filepath, header=None).to_numpy()
    return var