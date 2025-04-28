import os
import numpy as np
import pandas as pd
import bct

from src.connectomics import connectivity
from src.utils import io, transform

def read_connectome(connectome_path, fisher=False, standardise=False, not_negative=False, drop_first_roi=False, 
                    mask=None, replace_nan_with=None, verbose=True):
    """Read in the connectome."""

    # check if the connectome_path is valid and the file exists
    if pd.isna(connectome_path) or not os.path.isfile(connectome_path):
        if verbose: print(f"The file {connectome_path} does not exist or the path is NaN.")
        return None

    # infer the file type from the file extension
    _, file_extension = os.path.splitext(connectome_path)
    file_extension = file_extension.lower()
    
    # read in the connectome based on the file extension
    if file_extension == '.pkl':
        connectome = io.read_pickle(connectome_path)
    elif file_extension == '.mat':
        connectome = io.read_mat(connectome_path)
    elif file_extension == '.csv':
        connectome = io.read_csv2npy(connectome_path)
    else:
        if verbose: print(f"Unsupported file extension: {file_extension}")
        return None

    np.fill_diagonal(connectome, 0)

    # drop first ROI in the matrix
    if drop_first_roi:
        connectome = connectome[1:, 1:]
    
    # set negative correlations to zero
    if not_negative:
        connectome[connectome < 0] = 0 
    
    # replace NaNs with a value
    if isinstance(replace_nan_with, (int, float)) and np.isnan(connectome).any():
        connectome = np.nan_to_num(connectome, nan=replace_nan_with)

    # fisher r-to-z transformation
    if fisher:
        connectome = transform.fisher_r_to_z(connectome)

    if standardise:
        connectome = (connectome - np.mean(connectome)) / np.std(connectome)
    
    # apply a mask
    if mask is not None and mask.any():
        connectome = connectome * mask

    return connectome

def merge_connectomes(df, connectome_path_col, not_negative=False, drop_first_roi=False, 
                      fisher=False, standardise=False, replace_nan_with=None, mask=None, verbose=True):
    """Merge connectomes for subjects in dataframe with column for paths to the connectomes."""

    # Initialize empty lists to hold the correlation matrices and indices of NaN matrices
    connectomes = []
    nan_mids = []

    for i, row in df.iterrows():
        connectome = read_connectome(connectome_path=row[connectome_path_col], 
                                     fisher=fisher, 
                                     standardise=standardise, 
                                     drop_first_roi=drop_first_roi,
                                     not_negative=not_negative,
                                     replace_nan_with=replace_nan_with,
                                     mask=mask,
                                     verbose=verbose)

        # Check for None or NaN values and exclude the subject if such values are present
        if connectome is None:
            if verbose: print(f"Missing connectome path for {row['mid']}")
            nan_mids.append(row['mid'])
        elif np.isnan(connectome).any():
            if verbose: print(f"NaN values detected in the connectome; excluding {row['mid']})")
            nan_mids.append(row['mid'])
        elif (connectome==0).all():
            if verbose: print(f"All zero values detected in the connectome; excluding {row['mid']})")
            nan_mids.append(row['mid'])
        else:
            # Append the processed correlation matrix to our list
            connectomes.append(connectome)

    # Convert the list of correlation matrices to a 3D numpy array
    connectomes = np.stack(connectomes, axis=2)

    # Remove the rows from the DataFrame that correspond to NaN matrices
    df_cl = df[~df['mid'].isin(nan_mids)]

    return df_cl, connectomes

def get_percentile_mask(connectome, percentile):
    """Get a binary mask of connectome's top percentile values."""

    thr = np.percentile(np.triu(connectome, k=1), percentile)
    mask = np.where(connectome >= thr, 1, 0)
    return mask

def get_norm_connectome_mask(connectomes, mask=None, percentile=90):
    """Create normative connectome masks for connections above a given percentile threshold."""

    # Calculate the average correlation matrix and apply the triangular mask
    connectomes_norm = connectomes.mean(axis=2)

    # Apply a mask on the averaged connectome
    if mask is not None and mask.any():
        connectomes_norm = connectomes_norm * mask

    # Determine thresholds based on percentile
    connectomes_norm_mask = get_percentile_mask(connectomes_norm, percentile)

    return connectomes_norm, connectomes_norm_mask

def calculate_nodal_strength(connectome_path, fisher=False, standardise=False, drop_first_roi=False, 
                             not_negative=False, replace_nan_with=None, mask=None, percentile=None,
                             verbose=False):
    """Calculate the nodal strength of nodes within the connectome nodes."""
    
    # Read in the connectome
    connectome = read_connectome(connectome_path=connectome_path, 
                                fisher=fisher, 
                                standardise=standardise, 
                                drop_first_roi=drop_first_roi,
                                not_negative=not_negative,
                                replace_nan_with=replace_nan_with,
                                mask=mask,
                                verbose=verbose)
    if connectome is None:
        return None
    
    # Calculate percentile mask and apply it on the connectome
    if percentile is not None:
        perc_mask = get_percentile_mask(connectome, percentile)
        connectome = connectome * perc_mask

    # Calculate nodal strength for the nodes of the connectome
    nodal_strength = np.sum(connectome, axis=1)

    return nodal_strength
