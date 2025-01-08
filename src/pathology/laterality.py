import numpy as np
import pandas as pd

from pathology import laterality

def laterality_index(tau_left, tau_right):
    """
    Calculate asymmetry index.
    
    Args:
        tau_left (float): Tau value for left hemisphere.
        tau_right (float): Tau value for right hemisphere.
    
    Returns:
        float: Asymmetry index.
    """
    try:
        return ((tau_right - tau_left) / (tau_right + tau_left)) * 100
    except ZeroDivisionError:
        return np.nan

def calculate_regional_laterality(df, cols, prefix='', suffix='_LI'):
    """
    Calculate ROI-based laterality indices for tau-PET regions.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (list): List of column names.
        prefix (str): Prefix for output column names.
        suffix (str): Suffix for output column names.
    
    Returns:
        pd.DataFrame: DataFrame with added laterality index columns.
    """
    left_columns = [col for col in cols if '_lh_' in col or '_Left' in col]
    right_columns = [col for col in cols if '_rh_' in col or '_Right' in col]

    region_pairs = [(left, left.replace('_lh_', '_rh_').replace('_Left', '_Right')) 
                    for left in left_columns if left.replace('_lh_', '_rh_').replace('_Left', '_Right') in right_columns]
    
    for left_col, right_col in region_pairs:
        region_name = left_col.split('_lh_')[1] if '_lh_' in left_col else left_col.split('_Left_')[1]
        li_col = f"{prefix}{region_name}{suffix}"
        df[li_col] = laterality_index(df[left_col], df[right_col])

    return df
