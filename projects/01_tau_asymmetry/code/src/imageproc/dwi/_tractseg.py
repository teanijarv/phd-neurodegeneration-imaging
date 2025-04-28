# to-do: generalise QC function

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.transform import zscore

def QC(df_tseg:pd.DataFrame, tracts:list, metric='tract_vol', zscore_thr=-2, plot=True):

    # calculate zscores for each tract and whether exceeding threshold
    df_tseg_QC = df_tseg.copy()
    for tract in tracts:
        df_tseg_QC[f'QC_zs_{metric}_{tract}'] = df_tseg_QC[f'{metric}_{tract}'].transform(zscore)
        df_tseg_QC[f'QC_zs_{metric}_{tract}_thr'] = (df_tseg_QC[f'QC_zs_{metric}_{tract}'] < zscore_thr).astype(int)

    # plot the summary
    if plot:
        metric_cols = [f'{metric}_{tract}' for tract in tracts]
        QC_cols = [f'QC_zs_{metric}_{tract}_thr' for tract in tracts]
        
        df_metrics = df_tseg_QC.melt(id_vars=['mid', 'date', 'mri_date__index'], 
                                value_vars=metric_cols,
                                var_name='tract', value_name=metric)
        df_QC = df_tseg_QC.melt(id_vars=['mid', 'date', 'mri_date__index'], 
                                value_vars=QC_cols,
                                var_name='tract', value_name='QC')

        df_metrics['tract'] = df_metrics['tract'].str.replace(f'{metric}_', '')
        df_QC['tract'] = df_QC['tract'].str.replace(f'QC_zs_{metric}_', '').str.replace('_thr', '')

        df_melt = pd.merge(df_metrics, df_QC, on=['mid', 'date', 'mri_date__index', 'tract'])

        _, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax = sns.stripplot(ax=ax, data=df_melt, x='tract', y=metric, hue='QC', alpha=0.5,
                           palette={0: 'blue', 1: 'red'})
        
        for tract in tracts:
            count_0 = df_melt[(df_melt['tract'] == tract) & (df_melt['QC'] == 0)].shape[0]
            count_1 = df_melt[(df_melt['tract'] == tract) & (df_melt['QC'] == 1)].shape[0]
            ax.text(x=tracts.index(tract), y=df_melt[metric].max()+8, s=f'> {zscore_thr} SD: {count_0}', color='blue', ha='center')
            ax.text(x=tracts.index(tract), y=df_melt[metric].max()+5, s=f'< {zscore_thr} SD: {count_1}', color='red', ha='center')

    return df_tseg_QC