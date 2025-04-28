import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import ScalarFormatter

def set_plot_style(style='whitegrid', context='paper', font='Helvetica', 
                   fontsize=dict(main=7, title=7, label=7, tick=7, legend=7),
                   grid=dict(line='-', alpha=0.3), dpi=300):
    sns.set_theme(font=font, style=style, context=context)
    plt.rc('grid', linestyle=grid['line'], alpha=grid['alpha'])
    plt.rcParams.update({
        'font.size': fontsize['main'],
        'axes.titlesize': fontsize['title'],
        'axes.labelsize': fontsize['label'],
        'xtick.labelsize': fontsize['tick'],
        'ytick.labelsize': fontsize['tick'],
        'legend.fontsize': fontsize['legend'],
        'figure.dpi': dpi
    })

def set_scientific_formatter(ax, axis='y'):
    format = ScalarFormatter(useMathText=True)
    format.set_scientific(True)
    format.set_powerlimits((-1, 1))
    if axis == 'y': ax.yaxis.set_major_formatter(format)
    elif axis == 'x': ax.xaxis.set_major_formatter(format)
    elif axis == 'xy': 
        ax.xaxis.set_major_formatter(format)
        ax.yaxis.set_major_formatter(format)
    return ax

def add_significance(ax, order, comparisons, pvals, hide_ns=False, h_coef=0.02, y_coef=0.175, lw=1.25, fontsize=10, col='k'):
    """
    Annotate a plot with significance indicators based on p-values for specified group comparisons.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object of the plot to annotate.
    order (list): The order of the groups in the plot.
    comparisons (list of tuples): List of tuples where each tuple contains two group names to compare.
    pvals (dict): Dictionary where keys are group comparisons and values are p-values.
    hide_ns (bool, optional): If True, do not show non-significant comparisons (p >= 0.05). Default is False.
    h_coef (float, optional): Coefficient to determine the height of the significance lines relative to y_max. Default is 0.02.
    y_coef (float, optional): Coefficient to determine the y-offset for significance lines relative to y_max. Default is 0.175.
    lw (float, optional): Line width for the significance lines. Default is 1.25.
    fontsize (int, optional): Font size for the significance asterisks. Default is 10.
    col (str, optional): Color for the significance lines and asterisks. Default is 'k'.
    """
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    comparisons_locs = {comp: (order.index(comp[0]), order.index(comp[1])) for comp in comparisons}
    pvals_comp = {comp: pvals.get(comp, pvals.get((comp[1], comp[0]), None)) for comp in comparisons}

    # calculate the new y-axis limit needed to fit the significance bars
    num_comparisons = sum(1 for p_val in pvals_comp.values() if not hide_ns or p_val < 0.05)
    max_y = y_max + y_range * y_coef * max(num_comparisons, 2)  # ensure there is enough space for at least 2 comparisons

    # set the new y-axis limit
    ax.set_ylim(y_min, max_y)

    # annotate the plot with significance bars and asterisks
    for i, ((group1, group2), p_val) in enumerate(pvals_comp.items()):
        if hide_ns and p_val >= 0.05:
            continue
        
        # get the x-coordinates for the groups
        x1, x2 = comparisons_locs[(group1, group2)]
        y = y_max + y_range * y_coef * (i + 0.25)  # start from the top of the plot

        # draw the lines with tails
        ax.plot([x1, x1, x2, x2], [y, y + y_range * h_coef, y + y_range * h_coef, y], lw=lw, c=col)

        # determine the number of asterisks based on the p-value
        if p_val < 0.001:
            stars = '***'
        elif p_val < 0.01:
            stars = '**'
        elif p_val < 0.05:
            stars = '*'
        else:
            stars = 'ns'

        # annotate the plot with the stars
        ax.text((x1 + x2) * .5, y + y_range * h_coef, stars, ha='center', va='bottom', color=col, fontsize=fontsize)