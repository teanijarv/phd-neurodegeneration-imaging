import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import zscore

from src.analysis.misc import add_significance

def plot_pairwise_comparison(df, x, y, order, comparisons=None, pvals=None, hide_ns=False, figsize=(10, 6), dpi=100, palette=None):
    """
    Plot boxplots with stripplots for specified groups and optionally annotate with significance indicators.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data to plot.
    x (str): Column name for the x-axis (group variable).
    y (str): Column name for the y-axis (response variable).
    order (list): The order of the groups to be plotted.
    comparisons (list of tuples, optional): List of tuples where each tuple contains two group names to compare.
    pvals (dict, optional): Dictionary where keys are group comparisons and values are p-values.
    hide_ns (bool, optional): If True, do not show non-significant comparisons (p >= 0.05). Default is False.
    figsize (tuple, optional): Figure size. Default is (10, 6).
    palette (list, optional): List of colors for the plot. Default is seaborn "deep" palette.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    matplotlib.axes.Axes: The axes object containing the plot.

    Usage:
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt

    >>> df = pd.DataFrame({
    ...     'Group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    ...     'Value': [1.2, 2.3, 3.1, 2.1, 3.4, 1.9, 3.2, 2.9, 3.8]
    ... })

    >>> order = ['A', 'B', 'C']
    >>> comparisons = [('A', 'B'), ('A', 'C')]
    >>> pvals = {('A', 'B'): 0.03, ('A', 'C'): 0.01}

    >>> fig, ax = plot_pairwise_comparison(df, x='Group', y='Value', order=order, comparisons=comparisons, pvals=pvals, hide_ns=False)
    >>> plt.show()
    """
    if palette is None:
        palette = sns.color_palette("deep")

    # plot boxplot and striplot on top
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    sns.boxplot(ax=ax, x=x, y=y, data=df, order=order, palette=palette, showfliers=False)
    sns.stripplot(ax=ax, x=x, y=y, data=df, order=order, palette=palette, linewidth=0.8, size=3, alpha=1)

    # add significance bars with asterices
    if comparisons is not None and pvals is not None:
        add_significance(ax, order, comparisons, pvals, hide_ns=hide_ns)
    
    return fig, ax

def plot_regression(df, x, y, covars=[], standardise=True, vars2std='all', ci=95,
                    ax=None, xlabel=None, ylabel=None, hue=None, title=None, palette=None,
                    fig_args=None, scatter_kwargs=None, line_kwargs=None, text_kwargs=None, legend_kwargs=None):

    # prepare data and optionally z-score
    df_v = df[[y, x] + covars]
    if standardise:
        if vars2std == 'all':
            df_v = zscore(df_v)
        elif type(vars2std) == list:
            df_v[vars2std] = zscore(df_v[vars2std])
    
    # OLS regression and extract the parameters for the main predictor
    model = sm.OLS(df_v[y], sm.add_constant(df_v[[x] + covars])).fit()
    b, pval = model.params[x], model.pvalues[x]

    # create figure and axis if not provided
    if ax is None:
        if fig_args is None:
            fig_args = dict(figsize=(2.5, 2.5), dpi=100)
        fig, ax = plt.subplots(1, 1, **fig_args)
    else:
        fig = ax.figure
    
    # set style
    if scatter_kwargs is None:
        scatter_kwargs = dict(edgecolor='#494949', color='#9f9f9f', s=30, lw=1, alpha=0.8)
    if line_kwargs is None:
        line_kwargs = dict(color='#494949')
    if text_kwargs is None:
        text_kwargs = dict(fontsize=11, color='#494949')
    if legend_kwargs is None:
        legend_kwargs = dict(loc='center left', bbox_to_anchor=(1, 0.5))

    # plot scatter and regression line
    ax_args = dict(ax=ax, data=df, x=x, y=y)
    sns.scatterplot(**ax_args, **scatter_kwargs, zorder=1, hue=hue, palette=palette)
    sns.regplot(**ax_args, **line_kwargs, scatter=False, ci=ci)

    # stats text
    p_text = '< 0.001' if pval < 0.001 else f'= {pval:.3f}'
    ax.text(0.0225, 0.975, f'β = {b:.3f}\np {p_text}', transform=ax.transAxes, va='top', **text_kwargs)

    if hue: ax.legend(**legend_kwargs)
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel(xlabel)
    if title: ax.set_title(title, y=1)

    return fig, ax, model