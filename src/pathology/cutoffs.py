import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

def is_above_cutoff(series, cutoff):
    """Check if values in series are above cut-off."""
    return series > cutoff

def find_combined_cutoff(scores_gmm, scores_2std, roi_name='roi', plot_gmm=False, verbose=False):
    """
    Find combined cutoff using GMM and 2SD methods.
    
    Args:
        scores_gmm (np.array): Scores for GMM method.
        scores_2std (np.array): Scores for 2SD method.
        roi_name (str): Name of the ROI.
        plot_gmm (bool): Whether to plot the GMM results.
        verbose (bool): Whether to print verbose output.
    
    Returns:
        float: Combined cutoff value.
    """

    # fit GMM
    gmm = GaussianMixture(n_components=2, random_state=0).fit(scores_gmm.reshape(-1, 1))
    means = gmm.means_.flatten()
    vars = gmm.covariances_.flatten()
    m1, m2 = np.sort(means)
    v1, v2 = vars[np.argsort(means)]
    
    # find intersection point
    def intersection(m1, v1, m2, v2):
        a = 1 / (2 * v1) - 1 / (2 * v2)
        b = m2 / v2 - m1 / v1
        c = m1**2 / (2 * v1) - m2**2 / (2 * v2) - np.log(np.sqrt(v2 / v1))
        roots = np.roots([a, b, c])
        return roots[np.isreal(roots)].real
    
    inter_points = intersection(m1, v1, m2, v2)
    gmm_cutoff = inter_points[0] if inter_points.size > 0 else None
    
    # calculate 2SD cutoff
    mean_neg = np.mean(scores_2std)
    std_neg = np.std(scores_2std)
    sd_cutoff = mean_neg + 2 * std_neg
    
    # combine cutoffs
    combined_cutoff = (gmm_cutoff + sd_cutoff) / 2 if gmm_cutoff is not None else sd_cutoff

    # output results
    if verbose:
        print(f'{roi_name} combined cut-off = {combined_cutoff:.3f}')
    
    # plot if requested
    if plot_gmm:
        plt.figure(figsize=(6, 4), dpi=100)
        plt.hist(scores_gmm, bins=30, density=True, alpha=0.6, color='g')
        x = np.linspace(min(scores_gmm), max(scores_gmm), 1000)
        plt.plot(x, norm.pdf(x, m1, np.sqrt(v1)), '-k', alpha=0.5)
        plt.plot(x, norm.pdf(x, m2, np.sqrt(v2)), '-k', alpha=0.5)
        if gmm_cutoff is not None:
            plt.axvline(gmm_cutoff, color='r', linestyle='--', label=f'GMM cutoff ({gmm_cutoff:.3f})')
        plt.axvline(sd_cutoff, color='b', linestyle='--', label=f'2SD cutoff ({sd_cutoff:.3f})')
        plt.axvline(combined_cutoff, color='m', linestyle='-', label=f'Combined cutoff ({combined_cutoff:.3f})')
        plt.xlabel('Scores')
        plt.ylabel('Density')
        plt.title(f'GMM and 2SD Cut-off for {roi_name}')
        plt.legend(loc='upper right')
        plt.show()
    
    return combined_cutoff

def find_gmm_cutoff(scores_gmm, roi_name='roi', plot_gmm=False, verbose=False):
    """
    Find cutoff using GMM method.
    
    Args:
        scores_gmm (np.array): Scores for GMM method.
        roi_name (str): Name of the ROI.
        plot_gmm (bool): Whether to plot the GMM results.
        verbose (bool): Whether to print verbose output.
    
    Returns:
        float: GMM cutoff value.
    """
    
    # fit GMM
    gmm = GaussianMixture(n_components=2, random_state=0).fit(scores_gmm.reshape(-1, 1))
    means = gmm.means_.flatten()
    vars = gmm.covariances_.flatten()
    m1, m2 = np.sort(means)
    v1, v2 = vars[np.argsort(means)]
    
    # find intersection point
    def intersection(m1, v1, m2, v2):
        a = 1 / (2 * v1) - 1 / (2 * v2)
        b = m2 / v2 - m1 / v1
        c = m1**2 / (2 * v1) - m2**2 / (2 * v2) - np.log(np.sqrt(v2 / v1))
        roots = np.roots([a, b, c])
        return roots[np.isreal(roots)].real
    
    inter_points = intersection(m1, v1, m2, v2)
    gmm_cutoff = inter_points[0] if inter_points.size > 0 else None

    # output results
    if verbose:
        print(f'{roi_name} GMM cut-off = {gmm_cutoff:.3f}')
    
    # plot if requested
    if plot_gmm:
        plt.figure(figsize=(6, 4), dpi=100)
        plt.hist(scores_gmm, bins=30, density=True, alpha=0.6, color='g')
        x = np.linspace(min(scores_gmm), max(scores_gmm), 1000)
        plt.plot(x, norm.pdf(x, m1, np.sqrt(v1)), '-k', alpha=0.5)
        plt.plot(x, norm.pdf(x, m2, np.sqrt(v2)), '-k', alpha=0.5)
        plt.axvline(gmm_cutoff, color='r', linestyle='--', label=f'GMM cutoff ({gmm_cutoff:.3f})')
        plt.xlabel('Scores')
        plt.ylabel('Density')
        plt.title(f'GMM Cut-off for {roi_name}')
        plt.legend(loc='upper right')
        plt.show()
    
    return gmm_cutoff