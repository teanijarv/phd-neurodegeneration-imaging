import numpy as np
import pandas as pd
import warnings

def determine_amyloid_status(row, pet_col='fnc_ber_com_composite', 
                             cutoff=1.033, csf_ratio_col='Abnormal_CSF_Ab42_Ab40_Ratio'):
    """
    Set amyloid-positivity status for each row in dataframe based on Ab-PET or CSF Ab42/40 ratio.
    
    Args:
        row (pd.Series): A row from the DataFrame.
        pet_col (str): The column name for Ab-PET data.
        cutoff (float): The cutoff value for determining amyloid positivity from PET data.
        csf_ratio_col (str): The column name for CSF data (binary).
    
    Returns:
        int or float: 1 if amyloid positive, 0 if not, and NaN if no data is available.
    """
    if pd.notna(row[pet_col]):
        return int(row[pet_col] > cutoff)
    elif csf_ratio_col and pd.notna(row[csf_ratio_col]):
        return int(row[csf_ratio_col] == 1)
    return np.nan

def compute_roi_amyloid(df, roi_regions, side_substrs, pet_prefix='fnc_sr_mr_fs_', vol_prefix='fnc_vx_fs_'):
    """
    Compute amyloid burden for all ROIs using PET SUVR values, optionally weighted by regional volumes.
    
    Args:
        df (pd.DataFrame): DataFrame containing amyloid PET and volume data.
        roi_regions (list): List of region names to include in the calculation.
        side_substrs (list): List of substrings indicating hemisphere (e.g., ['lh', 'rh']).
        pet_prefix (str): Prefix for amyloid PET columns in the DataFrame.
        vol_prefix (str): Prefix for volume columns in the DataFrame.
    
    Returns:
        float: Volume-weighted or unweighted average amyloid SUVR across specified regions.
    """
    numerator = 0
    denominator = 0
    # loop through all regions for both hemispheres
    for region in roi_regions:
        for side in side_substrs:
            # identify amyloid uptake and volume columns
            amy_col = next((col for col in df.columns if region in col 
                            and pet_prefix in col and side in col), None)
            vol_col = next((col for col in df.columns if region in col 
                            and vol_prefix in col and side in col), None)
            # if both tau and volume are available, add to nominator and denominator
            if amy_col and vol_col:
                numerator += df[amy_col] * df[vol_col]
                denominator += df[vol_col]
            elif amy_col and not vol_col:
                numerator += df[amy_col]
                denominator += 1
    try:
        roi_amy = numerator / denominator
    except:
        roi_amy = numerator
    return roi_amy

def compute_roi_tau(df, roi_regions, side_substrs, pet_prefix='tnic_sr_mr_fs_', vol_prefix="tnic_vx_fs_"):
    """
    Compute ROI tau using tau-PET values, optionally weighted by regional volume sizes.
    
    Args:
        df (pd.DataFrame): DataFrame containing tau PET and volume data.
        roi_regions (list): List of region names to include in the calculation.
        side_substrs (list): List of substrings indicating hemisphere (e.g., ['lh', 'rh']).
        pet_prefix (str): Prefix for tau PET columns in the DataFrame.
                          'tnic' for non-partial volume corrected, 'tgic' for partial volume corrected.
        vol_prefix (str): Prefix for volume columns in the DataFrame (used only with 'tnic').
    
    Returns:
        float: Volume-weighted or unweighted average tau SUVR across specified regions.
    
    Notes:
        - For non-partial volume corrected values ('tnic'), the function weights tau values by region volumes.
        - For partial volume corrected values ('tgic'), the function calculates a simple average.
    """
    numerator = 0
    denominator = 0
    # loop through all regions for both hemispheres
    for region in roi_regions:
        for side in side_substrs:
            # identify tau uptake columns
            tau_col = next((col for col in df.columns if region in col 
                            and pet_prefix in col and side in col), None)
            # if non partial volume corrected (tnic) values are used
            if 'tnic' in pet_prefix:
                # identify volume columns
                vol_col = next((col for col in df.columns if region in col 
                                and vol_prefix in col and side in col), None)
                # if both tau and volume are available, add to nominator and denominator
                if tau_col and vol_col:
                    numerator += df[tau_col] * df[vol_col]
                    denominator += df[vol_col]
                elif tau_col and not vol_col:
                    numerator += df[tau_col]
                    denominator += 1
            # if partial volume corrected (tgic) values instead
            elif 'tgic' in pet_prefix:
                if tau_col:
                    numerator += df[tau_col]
                    denominator += 1
    # return the ROI SUVR either weighted by volume or not
    if 'tnic' in pet_prefix:
        try:
            return numerator / denominator
        except:
            warnings.warn(f"SUVR values not divided by volumes due to divide by zero in ROIs: {roi_regions}", UserWarning)
            return numerator
    elif 'tgic' in pet_prefix:
        return numerator / denominator

def compute_roi_ct(df, roi_regions, side_substrs, prefix='aparc_ct_avg_'):
    """
    Compute average cortical thickness across specified regions of interest.
    
    Args:
        df (pd.DataFrame): DataFrame containing cortical thickness data.
        roi_regions (list): List of region names to include in the calculation.
        side_substrs (list): List of substrings indicating hemisphere (e.g., ['lh', 'rh']).
        prefix (str): Prefix for cortical thickness columns in the DataFrame.
    
    Returns:
        float: Average cortical thickness across specified regions.
    
    Notes:
        This function calculates a simple average of cortical thickness values
        across all specified regions and hemispheres without volume weighting.
    """
    numerator = 0
    denominator = 0
    for region in roi_regions:
        for side in side_substrs:
            ct_col = next((col for col in df.columns if region in col 
                          and prefix in col and side in col), None)
            if ct_col:
                numerator += df[ct_col]
                denominator += 1
    try:
        roi_ct = numerator / denominator
    except:
        roi_ct = numerator

    return roi_ct