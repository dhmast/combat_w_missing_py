# -*- coding: utf-8 -*-

"""
Created on Tue Apr 11 12:32:00 2023

@author: mast527

1. Compute the mean and variance of each batch:

    -Identify the unique batches in the batch variable using np.unique.
    -For each batch, extract the corresponding rows of X.
    -If missing_indicator is not None, mask the missing values in the batch using np.ma.masked_array.
    -Compute the batch mean and variance using np.mean and np.var with ddof=0 (delta degrees of freedom)
        to calculate the unbiased estimator of the variance.
    -Store the batch mean and variance in separate lists batch_means and batch_vars.

2. Compute the global mean and variance of X:

    -If missing_indicator is not None, mask the missing values in X.
    -Compute the global mean and variance using np.mean and np.var with ddof=1 
        (delta degrees of freedom) to calculate the biased estimator of the variance.
   
3. Compute the scaling and shift parameters for each batch:

    -For each batch, compute the scaling parameter sval using the formula sval = sqrt(batch_var / global_var),
        where batch_var is the variance of the batch and global_var is the global variance of X.
    
    -Compute the shift parameter shift using the formula shift = global_mean - sval * batch_mean.
    -Store the scaling and shift parameters in separate lists svals and shift.

4. Apply the scaling and shift parameters to X:

    -Initialize an empty matrix X_corrected with the same shape as X.
    -For each batch, extract the corresponding rows of X and the corresponding indices in X_corrected.
    -If missing_indicator is not None, mask the missing values in the batch using np.ma.masked_array.
    -Apply the scaling and shift parameters to the batch using svals[i] * X_batch + shift[i].
    -Fill the masked values with np.nan using np.ma.filled.
    -Assign the corrected batch values to the corresponding indices in X_corrected.
    -If missing_indicator is not None, assign np.nan to the missing values in X_corrected.
    
5. Return X_corrected.

"""

import numpy as np
import pandas as pd


def combat_with_missing_v3(X, batch, missing_indicator=None, center=True, scale_to_global_var=False):
    """
    Perform batch correction on a matrix X with respect to a batch variable.
    This implementation also supports missing values, which can be indicated by
    a binary indicator matrix with the same shape as X.
    
    center: whether to center the data after batch correction
    
    """
    # Compute the mean and variance of each batch
    batches = np.unique(batch)
    batch_means = []
    batch_vars = []
    for i, b in enumerate(batches):
        X_batch = X[batch == b]
        if missing_indicator is not None:
            X_batch = np.ma.masked_array(X_batch, missing_indicator[batch == b])
        batch_mean = np.mean(X_batch, axis=0)
        batch_var = np.var(X_batch, axis=0, ddof=0)
        batch_means.append(batch_mean)
        batch_vars.append(batch_var)

    # Compute the global mean and variance of X
    if missing_indicator is not None:
        X = np.ma.masked_array(X, missing_indicator)
    global_mean = np.mean(X, axis=0)
    global_var = np.var(X, axis=0, ddof=1)

    # Compute the scaling and shift parameters for each batch
    svals = []
    shift = []
    
    for i, b in enumerate(batches):
        batch_mean = batch_means[i]
        batch_var = batch_vars[i]
        
        if scale_to_global_var:
            sval = np.sqrt(batch_var / global_var) # optional line
        else:
            sval = np.sqrt(batch_var)
        svals.append(sval)
        shift.append(global_mean - sval * batch_mean)

    # Apply the scaling and shift parameters to X
    X_corrected = np.zeros(X.shape)
    for i, b in enumerate(batches):
        batch_indices = np.where(batch == b)[0]
        if missing_indicator is not None:
            missing_batch = missing_indicator[batch == b]
            X_batch = np.ma.masked_array(X[batch == b], missing_batch)
            X_batch_corrected = svals[i] * X_batch + shift[i]
            X_batch_corrected = np.ma.filled(X_batch_corrected, np.nan)
            X_corrected[batch_indices] = X_batch_corrected
            missing_indices = np.where(missing_batch)
            X_corrected[batch_indices[missing_indices[0]], missing_indices[1]] = np.nan
        else:
            X_batch_corrected = svals[i] * X[batch == b] + shift[i]
            X_corrected[batch_indices] = X_batch_corrected

    # Center the data after batch correction
    if center:
        X_corrected -= global_mean

    return X_corrected








