import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from enigmatoolbox.utils.parcellation import parcel_to_surface
from brainspace.utils.parcellation import map_to_labels
from brainspace.datasets import load_parcellation, load_conte69
from brainspace.plotting import plot_hemispheres

def shorten_dk_names(original_dict):
    new_dict = {}
    for key, value in original_dict.items():
        if 'ctx_lh_' in key:
            region = key.split('ctx_lh_')[1]
            new_key = f'L_{region}'
        elif 'ctx_rh_' in key:
            region = key.split('ctx_rh_')[1]
            new_key = f'R_{region}'
        else:
            new_key = key
        new_dict[new_key] = value
    
    return new_dict

def assign_val2roi(regions, value, side_prefixes=['L', 'R']):
    # loop through all regions
    name_variations = {'Hippocampus': 'hippo', 'Amygdala': 'amyg'}
    roi_dict = {}
    for region in regions:
        # modify the names of hippocampus and amygdala
        if region in name_variations:
            region_key = name_variations[region]
        else:
            region_key = region
        
        # assign the value to regions in both hemispheres
        roi_dict[f'{side_prefixes[0]}_{region_key}'] = value
        roi_dict[ f'{side_prefixes[1]}_{region_key}'] = value
    
    return roi_dict

def plot_surface(data, atlas, figsize=None, layout_style='row', cmap='viridis', cmap_range=None, cbar=True, 
                 output='notebook', plot_kwargs=None):
    """
    Plot brain surface data using either Desikan-Killiany (aparc) or Schaefer 400 parcellation.

    Parameters
    ----------
    data : dict or pandas.Series
        Data to plot on the brain surface. Must be a dictionary or pandas Series 
        with region names as keys/index and values to plot.
    
    atlas : str
        Parcellation atlas to use. Must be either:
        - 'aparc': Desikan-Killiany atlas (68 regions)
        - 'schaefer400': Schaefer atlas (400 regions)
    
    figsize : tuple, optional
        Figure size in pixels as (width, height). Default is None.
    
    layout_style : str, optional
        Layout of the brain views. Default is 'row'.
        Options from brainspace.plotting.plot_hemispheres.
    
    cmap : str, optional
        Colormap to use for the plot. Default is 'viridis'.
        Any matplotlib colormap name can be used.
    
    cmap_range : tuple or None, optional
        Range for the colormap as (min, max). Default is None, 
        which automatically determines the range from the data.
    
    cbar : bool, optional
        Whether to display the colorbar. Default is True.
    
    output : str, optional
        Type of output to generate. Options are:
        - 'notebook': Display in Jupyter notebook (default)
        - 'interactive': Return interactive plot
        - filename: Save plot to file if string is provided
    
    plot_kwargs : dict or None, optional
        Additional keyword arguments to pass to plot_hemispheres. Default values are:
        - nan_color: (0.8, 0.8, 0.8, 1) [gray color for NaN values]
        - zoom: 1.35
        - transparent_bg: False
        - scale: 5

    Returns
    -------
    figure : brainspace plot object
        The generated brain plot. Type depends on output parameter:
        - Notebook display object if output='notebook'
        - Interactive plot if output='interactive'
        - None if output is a filename (saves to file instead)

    Raises
    ------
    TypeError
        If atlas is not 'aparc' or 'schaefer400'
        If data is not a dictionary or pandas Series
        If output is not 'notebook', 'interactive', or a string

    Examples
    --------
    >>> # Plot using Desikan-Killiany atlas
    >>> fig1 = plot_surface(data=dk_data, atlas='aparc')
    >>> display(fig1)
    
    >>> # Plot using Schaefer atlas and directly save to file
    >>> plot_surface(data=schaefer_data, atlas='schaefer400', 
    ...             output='brain_plot.png', cmap='RdBu_r')
    """

    # fetch labels of the atlas
    if atlas == 'aparc':
        _labels = [
            'bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus',
            'entorhinal', 'fusiform', 'inferiorparietal', 'inferiortemporal',
            'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual',
            'medialorbitofrontal', 'middletemporal', 'parahippocampal', 'paracentral',
            'parsopercularis', 'parsorbitalis', 'parstriangularis', 'pericalcarine',
            'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
            'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal',
            'superiorparietal', 'superiortemporal', 'supramarginal', 'frontalpole',
            'temporalpole', 'transversetemporal', 'insula'
        ]
        labels = [f'L_{r}' for r in _labels] + [f'R_{r}' for r in _labels]
    elif atlas == 'schaefer400':
        labels = load_parcellation('schaefer', scale=400, join=True)
    else:
        raise TypeError(f"Atlas must be either 'aparc' or 'schaefer400', not '{atlas}'.")

    # check input data
    if type(data) == dict or type(data) == pd.Series:
        if atlas == 'aparc':
            s_values = pd.Series(data, index=labels).to_numpy()
        elif atlas == 'schaefer400':
            s_values = pd.Series(data).to_numpy()
    else:
        raise TypeError(f"Data must be either dictionary or pandas Series, not {type(data).__name__}.")

    # map the values to vertices using the parcellation
    if atlas == 'aparc':
        vertex_values = parcel_to_surface(s_values, 'aparc_conte69')
        vertex_values[vertex_values == 0] = np.nan
    elif atlas == 'schaefer400':
        vertex_values = map_to_labels(s_values, labels, mask=labels!=0)
        vertex_values[labels == 0] = np.nan
    
    # set output type
    if output == 'notebook':
        embed_nb = True
        interactive = False
        screenshot = False
        filename = None
    elif output == 'interactive':
        embed_nb = False
        interactive = True
        screenshot = False
        filename = None
    elif type(output) == str:
        embed_nb = False
        interactive = False
        screenshot = True
        filename = output
    else:
        raise TypeError(f"Output must be either 'notebook', 'interactive', or filename as a string.")

    if figsize is None:
        if layout_style == 'row':
            figsize = (1000, 200)
        elif layout_style == 'grid':
            figsize = (600, 400)

    # additional plotting parameters to define
    if plot_kwargs is None:
        plot_kwargs = dict(nan_color=(0.8, 0.8, 0.8, 1), zoom=1.35, transparent_bg=False, scale=5)
    
    # load conte69 surfaces
    surf_lh, surf_rh = load_conte69()

    # return plot
    return plot_hemispheres(surf_lh, surf_rh, array_name=vertex_values, size=figsize, layout_style=layout_style,
                            cmap=cmap, color_range=cmap_range, color_bar=cbar, 
                            embed_nb=embed_nb, interactive=interactive, filename=filename, screenshot=screenshot,
                            **plot_kwargs)