# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:43:16 2023

@author: mast527
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#import custom modules from working directory
from batch_correction_example import combat_with_missing_v3
from PCA_funcs import plot_pca, category_prep

def plot_feature_hist(cdf:pd.DataFrame, rdf:pd.DataFrame, feature_col:str=None, batch:int=None):
    i=1
    feature = 'feature3'
    if batch is not None:
        i=batch
    if feature_col is not None:
        feature=feature_col
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))    
    rdf[rdf['batch']==i][feature].hist(ax=ax[0])
    cdf[cdf['batch']==i][feature].hist(color='gray', alpha=0.5, hatch='/',ax=ax[1])
    ax[0].set_title(f'{feature} histogram (R remove_batch_effect)')
    ax[1].set_title(f'{feature} histogram (python remove_batch_effect)')
    

def main_v3():
    # Create a random dataset with five batches and some missing values
    X, batch = make_blobs(n_samples=[50, 50, 50, 50, 50], centers=None, n_features=20, cluster_std=1, random_state=0)
    X[25, 1] = np.nan
    X[30, 0] = np.nan
    X[70, 0] = np.nan
    X[75, 1] = np.nan
    X[26, 1] = np.nan
    X[37, 0] = np.nan
    X[72, 0] = np.nan
    X[77, 1] = np.nan
    X[25, 1] = np.nan
    X[30, 2] = np.nan
    X[70, 3] = np.nan
    X[75, 4] = np.nan
    X[26, 1] = np.nan
    X[37, 4] = np.nan
    X[72, 2] = np.nan
    X[77, 1] = np.nan
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(X, columns=['feature{}'.format(i) for i in range(1, 21)])
    df['batch'] = batch
    
    # Perform batch correction with missing values
    X_corrected = combat_with_missing_v3(df.iloc[:, :-1].values, df['batch'].values, missing_indicator=np.isnan(df.iloc[:, :-1].values))
    
    # Convert the corrected data back to a pandas DataFrame
    df_corrected = pd.DataFrame(X_corrected, columns=['feature{}'.format(i) for i in range(1, 21)])
    df_corrected['batch'] = batch
    
    # Plot the original and corrected data
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(5):
        axs[0].scatter(df[df['batch'] == i]['feature1'], df[df['batch'] == i]['feature2'], label='Batch {}'.format(i+1))
    axs[0].legend()
    axs[0].set_title('Original Data')
    for i in range(5):
        axs[1].scatter(df_corrected[df_corrected['batch'] == i]['feature1'], df_corrected[df_corrected['batch'] == i]['feature2'], label='Batch {}'.format(i+1))
    axs[1].legend()
    axs[1].set_title('Corrected Data')
    plt.show()
    
    
    
    #PCA on uncorrected data  
    categories = category_prep(df['batch'].to_list())
    x = df.drop(columns='batch').to_numpy()
    plot_pca(x,categories)
    
    #make copy of df_corrected
    cdf = df_corrected.copy()
    #pca on corrected_df
    categories_c =category_prep(cdf['batch'].to_list())
    x_c = cdf.drop(columns='batch').to_numpy()
    plot_pca(x_c,categories_c)
    
    #load the data that was corrected by R MSnSet.utils remove_batch_effect() function
    rdf = pd.read_csv("R_corrected_data.csv")
    #add a batch column R
    rdf['batch'] = cdf['batch']
    
    #plot pca on R corrected data
    categories_r = category_prep(rdf['batch'].to_list())
    x_r = rdf.drop(columns='batch')
    x_r = x_r.to_numpy()
    plot_pca(x_r,categories_r)
    
    plot_feature_hist(cdf, rdf)
    
    return df, df_corrected




if __name__ == '__main__':
    
    
    #generate a simulated data set, and perform batch correction
    df, cdf = main_v3()
    
    #load the data that was corrected by R MSnSet.utils remove_batch_effect() function
    rdf = pd.read_csv("R_corrected_data.csv")
    #add a batch column R
    rdf['batch'] = cdf['batch']
    
    
    
    
    