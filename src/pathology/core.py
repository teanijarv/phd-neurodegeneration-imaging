import numpy as np
import pandas as pd
import warnings

### AMYLOID

def determine_amyloid_status(row, pet_col='fnc_ber_com_composite', cutoff=1.033, csf_ratio_col='Abnormal_CSF_Ab42_Ab40_Ratio'):
    """
    Set amyloid-positivity status for each row (subject) based on amyloid-PET or CSF.
    
    Args:
        row (pd.Series): A row from the DataFrame.
        pet_col (str): The column name for amyloid-PET data.
        cutoff (float): The cutoff value for determining amyloid positivity from PET data.
        csf_ratio_col (str): The column name for CSF data.
    
    Returns:
        int or float: 1 if amyloid positive, 0 if not, and NaN if no data is available.
    """
    if pd.notna(row[pet_col]):
        return int(row[pet_col] > cutoff)
    elif csf_ratio_col and pd.notna(row[csf_ratio_col]):
        return int(row[csf_ratio_col] == 1)
    return np.nan

def compute_roi_amyloid(df, roi_regions, side_substrs, pet_prefix='fnc_sr_mr_fs_', vol_prefix='fnc_vx_fs_'):
    """Compute ROI amyloid using amyloid-PET."""
    numerator = 0
    denominator = 0

    # 
    for region in roi_regions:
        for side in side_substrs:
            # 
            amy_col = next((col for col in df.columns if region in col 
                            and pet_prefix in col and side in col), None)
            vol_col = next((col for col in df.columns if region in col 
                            and vol_prefix in col and side in col), None)
            
            # need to add warnings in here in case no volume found etc
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

### TAU

def compute_roi_tau(df, roi_regions, side_substrs, pet_prefix='tnic_sr_mr_fs_', vol_prefix="tnic_vx_fs_"):
    """Compute ROI tau using tau-PET and weighing them with regional volume sizes."""
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
                elif tau_col and not vol_col: # add warning message!!!
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

def get_last_ebm_stage(row):
    """
    Identify the last affected EBM stage.
    
    Args:
        row (pd.Series): A row from the DataFrame.
    
    Returns:
        str or np.nan: The last affected EBM stage or NaN if none affected.
    """
    if row['is_ebm_IV']: return 'IV'
    elif row['is_ebm_III']: return 'III'
    elif row['is_ebm_II']: return 'II'
    elif row['is_ebm_I']: return 'I'
    else: return np.nan

def get_last_ebm_value(row, prefix, suffix):
    """
    Get the value for the last affected EBM stage.
    
    Args:
        row (pd.Series): A row from the DataFrame.
        prefix (str): Prefix for EBM column names.
        suffix (str): Suffix for EBM column names.
    
    Returns:
        float or np.nan: The value for the last affected EBM stage or NaN if none affected.
    """
    if pd.isna(row['last_affected_ebm']):
        return np.nan
    else:
        stage = row['last_affected_ebm']
        column_name = f'{prefix}_ebm_com_EBM_{stage}_{suffix}'
        return row[column_name]

### CORTICAL THICKNESS

def compute_roi_ct(df, roi_regions, side_substrs, prefix='aparc_ct_avg_'):
    """Compute ROI cortical thickness. [WIP]"""
    numerator = 0
    denominator = 0

    # 
    for region in roi_regions:
        for side in side_substrs:
            # 
            
            amy_col = next((col for col in df.columns if region in col 
                            and prefix in col and side in col), None)
            # 
            if amy_col:
                numerator += df[amy_col]
                denominator += 1
    try:
        roi_amy = numerator / denominator
    except:
        roi_amy = numerator

    return roi_amy