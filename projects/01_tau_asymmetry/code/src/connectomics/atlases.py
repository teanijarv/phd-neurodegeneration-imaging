import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting, datasets
import matplotlib.pyplot as plt

from src.utils import misc

def fetch_dk_atlas(coords=False, lut_idx=False):
    """Load Desikan-Killiany atlas' image and labels (and optionally coordinates)."""
    
    # src_dir = os.path.abspath(os.path.join(os.getcwd(), '../../../src'))
    src_dir = misc.find_src_directory()
    file_path = os.path.join(src_dir, 'resources', 'atlases', 'fs_aparcaseg_dk', 'aparc+aseg_mni_relabel.nii.gz')
    
    dk_img = nib.load(file_path)
    
    # get labels as dict with values as either LUT IDs or just iterations
    if lut_idx:
        dk_labels = get_dk_fs_lut()
    else:
        dk_labels = fetch_fs_mrtrix_labels()

    # get coordinates of the labels
    if coords: 
        dk_coords = plotting.find_parcellation_cut_coords(dk_img)
        return dk_img, dk_labels, dk_coords

    return dk_img, dk_labels

def fetch_schaefer_atlas(n_rois=400, coords=False):
    """Load Schaefer 2018 atlas' image and labels (and optionally coordinates)."""

    schaefer_dict = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois) 
    schaefer_img = nib.load(schaefer_dict['maps'])
    schaefer_labels = {label.decode('utf-8'): float(idx+1) for idx, label in enumerate(schaefer_dict['labels'])}
    if coords: 
        schaefer_coords = plotting.find_parcellation_cut_coords(schaefer_img)
        return schaefer_img, schaefer_labels, schaefer_coords

    return schaefer_img, schaefer_labels

def fetch_fs_mrtrix_labels(original_labels=False, drop_unknown=True, drop_duplicate_thalamus=True):
    """
    Reads the fs_default.txt file containing FreeSurfer labels and returns a dictionary mapping label names to indexes.
    Optionally modifies the labels to a specified format.
    """

    # src_dir = os.path.abspath(os.path.join(os.getcwd(), '../../../src'))
    src_dir = misc.find_src_directory()
    file_path = os.path.join(src_dir, 'resources', 'fs_labels', 'fs_default.txt')
    
    replacements = {
        'ctx-lh-': 'L_', 'ctx-rh-': 'R_', 'Left-': 'L_', 'Right-': 'R_',
        '-Cortex': '', '-Proper': 'proper', '-area': 'area',
    }

    # Open and read the file
    label_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                continue
            # Split the line into components
            parts = line.split()
            if len(parts) >= 3:
                # Extract the index and label name
                index = int(parts[0])
                label_name = parts[2]
                if drop_unknown:
                    if label_name in ('Unknown'):
                        continue
                if drop_duplicate_thalamus:
                    if label_name in ('Left-Thalamus', 'Right-Thalamus'):
                        continue
                # Modify the label name if requested
                if not original_labels:
                    for old, new in replacements.items():
                        label_name = label_name.replace(old, new)
                    if label_name.startswith('L_'):
                        label_name = label_name[2:].lower() + '_L'
                    elif label_name.startswith('R_'):
                        label_name = label_name[2:].lower() + '_R'
                # Add to the dictionary
                label_dict[label_name] = index

    return label_dict

def fetch_fs_lut_labels(original_labels=False):
    """
    Reads the FreeSurferColorLUT.txt file containing FreeSurfer labels and returns a dictionary mapping label names to indexes.
    Optionally modifies the labels to a specified format.
    """

    # src_dir = os.path.abspath(os.path.join(os.getcwd(), '../../../src'))
    src_dir = misc.find_src_directory()

    replacements = {
        'ctx-lh-': 'L_', 'ctx-rh-': 'R_', 'Left-': 'L_', 'Right-': 'R_',
        '-Cortex': '', '-Proper*': 'proper', '-area': 'area',
    }

    # Construct the path to the FreeSurferColorLUT.txt file
    file_path = os.path.join(src_dir, 'resources', 'fs_labels', 'FreeSurferColorLUT.txt')

    # Open and read the file
    label_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                continue
            
            # Split the line into components
            parts = line.split()
            if len(parts) >= 3:
                # Extract the index and label name
                index = int(parts[0])
                label_name = parts[1]

                # Modify the label name if requested
                if not original_labels:
                    for old, new in replacements.items():
                        label_name = label_name.replace(old, new)
                    if label_name.startswith('L_'):
                        label_name = label_name[2:].lower() + '_L'
                    elif label_name.startswith('R_'):
                        label_name = label_name[2:].lower() + '_R'
                
                # Add to the dictionary
                label_dict[label_name] = index
    
    return label_dict

def sortby_ordered_dict(item, dict_ordered):
    key, _ = item
    return dict_ordered.get(key, float('inf'))

def get_dk_fs_lut(sort=True):
    
    fs_lut_dict = fetch_fs_lut_labels()
    # _, dk_labels = fetch_dk_atlas()
    dk_labels = fetch_fs_mrtrix_labels()

    for label in list(fs_lut_dict):
        if label not in list(dk_labels):
            del fs_lut_dict[label]
    
    if sort:
        # fs_lut_dict = dict(sorted(fs_lut_dict.items(), key=sort_left_first))
        fs_lut_dict = dict(sorted(fs_lut_dict.items(), key=lambda item: sortby_ordered_dict(item, dk_labels)))
    
    return fs_lut_dict

def get_dk_hemisphere_mask(dk_labels, type='interhemi', plot=False):
    """
    Generate a mask for inter- or intra-hemispheric connections for Desikan-Killiany atlas.
    Note: brainstem/cerebellum has been excluded in either mask type.
    """
    # initialize the mask with False
    mask = np.zeros((len(dk_labels), len(dk_labels)), dtype=bool)

    # identify the indices for left and right hemispheres
    left_indices = {label: index for label, index in dk_labels.items() if '_L' in label}
    right_indices = {label: index for label, index in dk_labels.items() if '_R' in label}

    # exclude cerebellum and brainstem regions
    exclude_indices = [index for label, index in dk_labels.items() if 'cerebellum' in label.lower() or 'brainstem' in label.lower()]

    if type == 'interhemi':
        for i in left_indices.values():
            for j in right_indices.values():
                if i not in exclude_indices and j not in exclude_indices:
                    mask[i-1, j-1] = True
                    mask[j-1, i-1] = True
    elif type == 'intrahemi':
        for i in left_indices.values():
            for j in left_indices.values():
                if i != j and i not in exclude_indices and j not in exclude_indices:
                    mask[i-1, j-1] = True
                    mask[j-1, i-1] = True
        for i in right_indices.values():
            for j in right_indices.values():
                if i != j and i not in exclude_indices and j not in exclude_indices:
                    mask[i-1, j-1] = True
                    mask[j-1, i-1] = True 
    elif type == 'full':
        for i in range(1, len(dk_labels)+1):
            for j in range(1, len(dk_labels)+1):
                if i not in exclude_indices and j not in exclude_indices:
                    mask[i-1, j-1] = True
    elif type == 'interhemi_pairs':
        for left_label, left_index in left_indices.items():
            right_label = left_label.replace('_L', '_R')
            if right_label in right_indices:
                right_index = right_indices[right_label]
                if left_index not in exclude_indices and right_index not in exclude_indices:
                    mask[left_index-1, right_index-1] = True
                    mask[right_index-1, left_index-1] = True
    
    if plot:
        fig = plt.figure(figsize=(20, 20))
        fig = plotting.plot_matrix(mask, labels=dk_labels, figure=fig, grid=True)
        plt.title(f'{type} mask')
        plt.show()

    return mask

def roidict2hemispheric(roi_dict):
    """
    Append '_L' and '_R' to region names to indicate left and right hemispheres, 
    and combine them into a single dictionary with left hemisphere regions first.

    Parameters:
    roi_dict (dict): Dictionary where keys are ROI names and values are lists of region names.

    Returns:
    dict: Dictionary with ROI names suffixed with '_L' and '_R' and corresponding regions suffixed similarly.
    
    Usage:
    >>> anatomical_rois = {
    ...     'frontal': ['superiorfrontal', 'rostralmiddlefrontal'],
    ...     'parietal': ['superiorparietal', 'inferiorparietal']
    ... }
    >>> roi_dict_hemi = roidict2hemispheric(anatomical_rois)
    """
    roi_dict_hemi = {}
    for roi, regions in roi_dict.items():
        roi_dict_hemi[f"{roi}_L"] = [f"{reg}_L" for reg in regions]
    for roi, regions in roi_dict.items():
        roi_dict_hemi[f"{roi}_R"] = [f"{reg}_R" for reg in regions]
    return roi_dict_hemi

def dk_connectome_reg2roi(c_dk_arr, roi_dict, dk_labels, dk_coords=None):
    """
    Convert connectivity matrices from regional to ROI level, optionally calculating ROI coordinates.

    Parameters:
    c_dk_arr (numpy.ndarray): 3D array of connectivity data (regions x regions x subjects).
    roi_dict (dict): Dictionary where keys are ROI names and values are lists of region names.
    dk_labels (dict): Dictionary mapping region names to label indices.
    dk_coords (numpy.ndarray, optional): Array of coordinates for the regions. Default is None.

    Returns:
    tuple: (fc_arrs_roi, roi_coords_arr) if dk_coords is provided, else fc_arrs_roi
        fc_arrs_roi (numpy.ndarray): 3D array of connectivity data at ROI level.
        roi_coords_arr (numpy.ndarray, optional): Array of coordinates for the ROIs if dk_coords is provided.
    
    Usage:
    >>> c_dk_arr = np.random.rand(68, 68, 10)  # Example connectivity data
    >>> dk_labels = {f'region{i}': i for i in range(1, 69)}
    >>> anatomical_rois = {
    ...     'frontal': ['region1', 'region2'],
    ...     'parietal': ['region3', 'region4']
    ... }
    >>> roi_dict_hemi = roidict2hemispheric(anatomical_rois)
    >>> c_arrs_roi, roi_coords_arr = dk_connectome_reg2roi(c_dk_arr, roi_dict_hemi, dk_labels, dk_coords=np.random.rand(68, 3))
    """

    rois = list(roi_dict.keys())
    n_rois = len(rois)
    n_subjects = c_dk_arr.shape[-1]

    c_arrs_roi = np.empty((n_rois, n_rois, n_subjects))
    roi_coords = {roi: [] for roi in rois}

    # loop through all subjects
    for i in range(n_subjects):
        c_arr = np.triu(c_dk_arr[:, :, i])
        df_c_arr = pd.DataFrame(c_arr, columns=dk_labels.keys(), index=dk_labels.keys())
        df_c_arr.replace(0, np.nan, inplace=True)

        # create a mapping from each region to its corresponding ROI
        region_to_roi = {region: roi for roi, regions in roi_dict.items() for region in regions}
        if dk_coords is not None:
            for roi, regions in roi_dict.items():
                for region in regions:
                    roi_coords[roi].append(dk_coords[dk_labels[region] - 1]) # adjust index by -1 if needed

        # replace regions in the DataFrame with their ROI names and aggregate together
        df_c_arr_roi = df_c_arr.rename(index=region_to_roi, columns=region_to_roi)
        df_c_arr_roi = df_c_arr_roi.loc[rois, rois]
        df_c_arr_roi = df_c_arr_roi.groupby(level=0, sort=False).mean().T.groupby(level=0, sort=False).mean()

        c_arrs_roi[:, :, i] = df_c_arr_roi.to_numpy()

    # calculate the center coordinates for each ROI
    if dk_coords is not None:
        roi_center_coords = {roi: np.median(coords_list, axis=0) for roi, coords_list in roi_coords.items()}
        roi_coords_arr = np.array(list(roi_center_coords.values()))
        return c_arrs_roi, roi_coords_arr
    
    return c_arrs_roi

def get_dk_rois():
    _, dk_labels = fetch_dk_atlas()
    all_regions = list(set(
        dk_label[:-2].capitalize() if dk_label[:-2] in ['hippocampus', 'amygdala'] else dk_label[:-2]
        for dk_label in dk_labels
    ))

    ROIs = {
        # all regions
        'global': all_regions,
        # temporal meta-ROI
        'temporal_meta': ['entorhinal', 'parahippocampal', 'fusiform', 'Amygdala', 'inferiortemporal', 'middletemporal'],
        # tau Braak stages
        'cho_com_I_II': ['entorhinal'],
        'cho_com_III_IV': ['parahippocampal', 'parahippocampal', 'fusiform', 'Amygdala', 'inferiortemporal', 'middletemporal'],
        'cho_com_V_VI': ['caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'inferiorparietal', 'isthmuscingulate',
                         'lateraloccipital', 'lateralorbitofrontal', 'lingual', 'medialorbitofrontal', 'paracentral',
                         'parsopercularis', 'parstriangularis', 'parsorbitalis', 'pericalcarine', 'postcentral', 'posteriorcingulate',
                         'precentral', 'precuneus', 'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal',
                         'superiortemporal', 'supramarginal', 'frontalpole', 'temporalpole', 'transversetemporal', 'insula'],
        # tau EBM stages
        'ebm_com_EBM_I': ['entorhinal', 'Amygdala', 'Hippocampus'],
        'ebm_com_EBM_II': ['bankssts', 'fusiform', 'inferiortemporal', 'middletemporal', 'parahippocampal', 'superiortemporal', 'temporalpole'],
        'ebm_com_EBM_III': ['caudalmiddlefrontal', 'inferiorparietal', 'isthmuscingulate', 'lateraloccipital', 'posteriorcingulate', 'precuneus', 
                            'superiorparietal', 'supramarginal'],
        'ebm_com_EBM_IV': ['caudalanteriorcingulate', 'frontalpole', 'insula', 'lateralorbitofrontal', 'medialorbitofrontal', 'parsopercularis', 
                'parsorbitalis', 'parstriangularis', 'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal'],
        # anatomical ROIs (as suggested in Freesurfer)
        'frontal': ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 'parsopercularis', 'parstriangularis', 
                    'parsorbitalis', 'lateralorbitofrontal', 'medialorbitofrontal', 'precentral', 'paracentral', 'frontalpole', 
                    'rostralanteriorcingulate', 'caudalanteriorcingulate'],
        'parietal': ['superiorparietal', 'inferiorparietal', 'supramarginal', 'postcentral', 'precuneus', 'posteriorcingulate', 
                    'isthmuscingulate'],
        'temporal': ['superiortemporal', 'middletemporal', 'inferiortemporal', 'bankssts', 'fusiform', 'transversetemporal', 
                    'entorhinal', 'temporalpole', 'parahippocampal'],
        'occipital': ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine'],
        'subcortical': ['Hippocampus', 'Amygdala'],
        # amyloid stages
        'early_amyloid': ['precuneus', 'posteriorcingulate', 'isthmuscingulate', 'insula', 'medialorbitofrontal', 'lateralorbitofrontal'],
        'intermediate_amyloid': ['bankssts', 'caudalmiddlefrontal', 'cuneus', 'frontalpole', 'fusiform', 'inferiorparietal', 
                        'inferiortemporal', 'lateraloccipital', 'middletemporal', 'parahippocampal', 'parsopercularis', 
                        'parsorbitalis', 'parstriangularis', 'Putamen', 'rostralanteriorcingulate', 'rostralmiddlefrontal', 
                        'superiorfrontal', 'superiorparietal', 'superiortemporal', 'supramarginal'],
        'late_amyloid': ['lingual', 'pericalcarine', 'paracentral', 'precentral', 'postcentral'],
    }
    return ROIs

def get_sch_rois():

    _, sch_labels = fetch_schaefer_atlas(400)

    ROIs = {
        'global': list(sch_labels.keys()),
        'temporal_meta': ['LH_Limbic_TempPole_3', 'LH_Vis_1', 'LH_Vis_2', 'LH_Vis_3', 'LH_Vis_5', 'LH_Vis_7', 'LH_DorsAttn_Post_1', 
                          'LH_DorsAttn_Post_2', 'LH_DorsAttn_Post_3', 'LH_Limbic_TempPole_1', 'LH_Limbic_TempPole_4', 'LH_Limbic_TempPole_5', 
                          'LH_Limbic_TempPole_8', 'LH_Cont_Temp_1', 'LH_Default_Temp_1', 'LH_Default_Temp_2', 'LH_Default_Temp_3', 'LH_Default_Temp_4', 
                          'LH_Default_Temp_6', 'RH_Vis_1', 'RH_Vis_2', 'RH_Vis_3', 'RH_Vis_4', 'RH_Vis_5', 'RH_Vis_7', 'RH_DorsAttn_Post_1', 
                          'RH_DorsAttn_Post_2', 'RH_Limbic_TempPole_1', 'RH_Limbic_TempPole_2', 'RH_Limbic_TempPole_4', 'RH_Limbic_TempPole_6', 
                          'RH_Limbic_TempPole_7', 'RH_Cont_Temp_1', 'RH_Cont_Temp_2', 'RH_Default_Temp_1', 'RH_Default_Temp_2', 'RH_Default_Temp_5']
    }

    return ROIs