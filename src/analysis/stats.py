import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
from scipy.stats import zscore

def ttest_comparison(data_dict, comparisons, posthoc='fdr_bh'):
    """
    Perform independent t-tests for multiple group comparisons and adjust p-values.

    Parameters:
    data_dict (dict): Dictionary where keys are group names and values are arrays of data.
    comparisons (list of tuples): List of tuples where each tuple contains two group names to compare.
    posthoc (str, optional): Method for post-hoc p-value adjustment. Default is 'fdr_bh'.

    Returns:
    dict: T-values for each comparison.
    dict: Raw p-values for each comparison.
    dict: Adjusted p-values for each comparison.

    Usage:
    >>> data = {
    ...     'group1': [1.2, 2.3, 3.1, 4.5],
    ...     'group2': [2.1, 3.4, 1.9, 4.2],
    ...     'group3': [3.2, 2.9, 3.8, 4.6]
    ... }
    >>> comparisons = [('group1', 'group2'), ('group1', 'group3')]
    >>> tvals, pvals, pvals_cor = ttest_comparison(data, comparisons)
    """

    # loop through all comparisons
    tvals, pvals, pvals_cor = {}, {}, {}
    for grps in comparisons:
        # run independent t-test and get tvals and pvals 
        tvals[grps], pvals[grps] = ttest_ind(data_dict[grps[0]], data_dict[grps[1]])
    
    # perform post-hoc test if requested and supply corrected pvals
    if posthoc:
        _, _pvals_cor, _, _ = multipletests(list(pvals.values()), alpha=0.05, method=posthoc)
        for i, grps in enumerate(comparisons):
            pvals_cor[grps] = _pvals_cor[i]

    return tvals, pvals, pvals_cor

def ols_comparison(dfs, data_dict, comparisons, covars=[], standardise=True, posthoc='fdr_bh', verbose=False):
    """
    Perform OLS regression for multiple group comparisons with optional covariates and adjust p-values.

    Parameters:
    dfs (dict): Dictionary where keys are group names and values are DataFrames containing covariate data.
    data_dict (dict): Dictionary where keys are group names and values are arrays of data.
    comparisons (list of tuples): List of tuples where each tuple contains two group names to compare.
    covars (list, optional): List of covariate column names to include in the model. Default is an empty list.
    posthoc (str, optional): Method for post-hoc p-value adjustment. Default is 'fdr_bh'.
    verbose (bool, optional): If True, print model summaries for each comparison. Default is False.

    Returns:
    dict: T-values for each comparison.
    dict: Raw p-values for each comparison.
    dict: Adjusted p-values for each comparison.
    dict: OLS models for each comparison.

    Usage:
    >>> dfs = {
    ...     'group1': pd.DataFrame({'cov1': [1, 2, 3, 4], 'cov2': [5, 6, 7, 8]}),
    ...     'group2': pd.DataFrame({'cov1': [2, 3, 4, 5], 'cov2': [6, 7, 8, 9]}),
    ...     'group3': pd.DataFrame({'cov1': [3, 4, 5, 6], 'cov2': [7, 8, 9, 10]})
    ... }
    >>> data = {
    ...     'group1': [1.2, 2.3, 3.1, 4.5],
    ...     'group2': [2.1, 3.4, 1.9, 4.2],
    ...     'group3': [3.2, 2.9, 3.8, 4.6]
    ... }
    >>> comparisons = [('group1', 'group2'), ('group1', 'group3')]
    >>> tvals, pvals, pvals_cor, models = ols_comparison(dfs, data, comparisons, covars=['cov1', 'cov2'])
    """

    # loop through all comparisons
    tvals, pvals, pvals_cor, models = {}, {}, {}, {}
    for grps in comparisons:

        # prepare data for the compared pair
        data_list = []
        df_covars = pd.DataFrame()
        for grp in grps:
            df_covars = pd.concat([df_covars, dfs[grp][covars]], axis=0).reset_index(drop=True)
            for i in range(len(data_dict[grp])):
                data_list.append({'Group': grp, 'value': data_dict[grp][i]})
        df_pair = pd.concat([pd.DataFrame(data_list), df_covars], axis=1)

        if standardise:
            if df_pair.iloc[:, 1:].isnull().any().any():
                raise ValueError("Cannot standardize data containing NaN values. Please handle NaN values before standardization.")
            df_pair.iloc[:, 1:] = zscore(df_pair.iloc[:, 1:])
        
        # create the binary group variable
        df_pair['group_bin'] = df_pair['Group'].apply(lambda x: 1 if x == grps[0] else 0)
        
        # build the formula and fit the model
        covariate_str = " + ".join(covars)
        formula = f"value ~ {covariate_str} + group_bin" if covariate_str else "value ~ group_bin"
        model = ols(formula, data=df_pair).fit()
        
        # extract results
        tvals[grps] = model.tvalues['group_bin']
        pvals[grps] = model.pvalues['group_bin']
        models[grps] = model
        
        # print summary for each comparison
        if verbose:
            print(f"Comparison: {grps[0]} vs {grps[1]}\n"
                  f"{model.summary()}\n")

    # perform post-hoc test if requested and supply corrected pvals
    if posthoc:
        _, _pvals_cor, _, _ = multipletests(list(pvals.values()), alpha=0.05, method=posthoc)
        for i, grps in enumerate(comparisons):
            pvals_cor[grps] = _pvals_cor[i]

    return tvals, pvals, pvals_cor, models

def ols_matrix_comparison(dfs, data_dict, comparison, covars=[], posthoc='fdr_bh', verbose=False):
    """
    Compute OLS regression for each element in connectivity matrices for a specified group comparison,
    with optional covariates and post-hoc correction.

    Parameters:
    dfs (dict): Dictionary where keys are group names and values are DataFrames containing covariate data.
    data_dict (dict): Dictionary where keys are group names and values are 3D arrays (connectivity matrices).
    comparison (list of str): List containing two group names to compare.
    covars (list of str, optional): List of covariate column names to include in the model. Default is an empty list.
    posthoc (str, optional): Method for post-hoc p-value adjustment. Default is 'fdr_bh'.
    verbose (bool, optional): If True, print model summaries for each ROI pair comparison. Default is False.

    Returns:
    tuple: (tvals, pvals, pvals_cor)
        tvals (numpy.ndarray): T-values for each ROI pair.
        pvals (numpy.ndarray): Raw p-values for each ROI pair.
        pvals_cor (numpy.ndarray): Corrected p-values for each ROI pair.
    
    Usage:
    >>> dfs = {
    ...     'group1': pd.DataFrame({'cov1': [1, 2, 3, 4], 'cov2': [5, 6, 7, 8]}),
    ...     'group2': pd.DataFrame({'cov1': [2, 3, 4, 5], 'cov2': [6, 7, 8, 9]})
    ... }
    >>> data_dict = {
    ...     'group1': np.random.rand(4, 4, 5),
    ...     'group2': np.random.rand(4, 4, 5)
    ... }
    >>> comparison = ['group1', 'group2']
    >>> tvals, pvals, pvals_cor = ols_matrix_comparison(dfs, data_dict, comparison, covars=['cov1', 'cov2'])
    """

    # prepare data for the compared pair
    data_list = []
    df_covars = pd.DataFrame()
    for grp in comparison:
        df_covars = pd.concat([df_covars, dfs[grp][covars]], axis=0).reset_index(drop=True)
        for i in range(data_dict[grp].shape[-1]):
            data_list.append({'Group': grp, 'value': data_dict[grp][:, :, i]})
    df_pair = pd.concat([pd.DataFrame(data_list), df_covars], axis=1)

    # create the binary group variable
    df_pair['group_bin'] = df_pair['Group'].apply(lambda x: 1 if x == comparison[0] else 0)

    # loop through all pairs of ROIs in the matrix
    n_rois = data_dict[comparison[0]].shape[0]
    tvals = np.zeros((n_rois, n_rois))
    pvals = np.zeros((n_rois, n_rois))
    pvals_cor = np.zeros((n_rois, n_rois))
    for i in range(n_rois):
        for j in range(n_rois):
            # prepare a dataframe for the current ROI pair
            roi_values = [
                {'value': row['value'][i, j], 'group_bin': row['group_bin'], **row[covars].to_dict()}
                for index, row in df_pair.iterrows()
                if np.any(row['value'][i, j] != 0) and np.isfinite(row['value'][i, j])
            ]
            
            if roi_values:
                df_roi = pd.DataFrame(roi_values)

                # build the formula and fit the model
                if covars:
                    covariate_str = " + ".join(covars)
                    formula = f"value ~ {covariate_str} + group_bin"
                else:
                    formula = "value ~ group_bin"
                model = ols(formula, data=df_roi).fit()

                # print summary for each comparison if verbose is True
                if verbose:
                    print(f"Comparison of ROI {i} vs ROI {j}\n{model.summary()}\n")

                # extract results
                tvals[i, j] = model.tvalues['group_bin']
                pvals[i, j] = model.pvalues['group_bin']
            else:
                tvals[i, j], pvals[i, j] = np.nan, np.nan
    
    # perform post-hoc test if requested and supply corrected pvals
    if posthoc:
        # flatten the p-values array and filter out np.nan values
        pvals_flat = pvals.flatten()
        valid_mask = ~np.isnan(pvals_flat)
        valid_pvals = pvals_flat[valid_mask]
        
        # apply FDR correction to the valid p-values
        _, _pvals_cor, _, _ = multipletests(valid_pvals, alpha=0.05, method=posthoc)
        
        # map the corrected p-values back to their original positions
        pvals_cor = np.full(pvals.shape, np.nan)
        pvals_cor[valid_mask.reshape(pvals.shape)] = _pvals_cor

    return tvals, pvals, pvals_cor