# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:38:56 2023

@author: mast527
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA

class SubsetDataFrame:
    """
    A class that creates a subset of a pandas DataFrame based on specified feature and category columns.

    Attributes:
        feature_columns (list): a list of feature columns to include in the subset.
        category_columns (list): a list of category columns to include in the subset.

    Methods:
        subset(df): takes in a pandas DataFrame and returns a subset of it based on the specified feature and category columns.

"""
    def __init__(self, feature_columns:list, category_columns:list):
        if isinstance(feature_columns, str):
            self.feature_columns = [feature_columns]
        elif isinstance(feature_columns, list):
            self.feature_columns = feature_columns
        else:
            pass
            
        if isinstance(category_columns, str):
            self.category_columns = [category_columns]
        elif isinstance(category_columns, list):
            self.category_columns=category_columns
        else:
            pass
        
    def subset(self, df):
        
        try:
            subset = df.copy()[self.category_columns+self.feature_columns]
        except Exception:
            print('could not subset dataframe')
            return None
        else:
            return subset

class PlotPCA:
    def __init__(self, df, feature_columns:list, categorical_columns:list):
        
        feature_test = [True if feature in df.columns else False for feature in feature_columns]
        category_test = [True if cat in df.columns else False for cat in categorical_columns]
            
        self.categorical_columns = categorical_columns
        self.feature_columns = feature_columns
        if all(feature_test):
            self.X_df = df[feature_columns]
            self.X = df[feature_columns].to_numpy()
        else:
            self.X_df = None
            self.X = None
            
        if all(category_test):
            self.categorical_data = df[categorical_columns]
        else:
            self.categorical_data = None
        

        
            
        
        pass

    def category_prep(self, categorical_column, nan=False, color_list=None):
        """
        Given an array-like object of categorical data, returns a dictionary mapping unique categories to colors,
        as well as a new array of the input data converted to strings.
    
        Args:
            categorical_data (array-like): The input categorical data. Must be an instance of pandas.Categorical,
                pandas.Series, numpy.ndarray, or list.
            nan (bool, optional): If True, retains NaN values in the unique categories set. Default is False.
            color_list (list of str, optional): A list of color names to use for the category-color mapping. If not
                provided, defaults to ['red', 'blue', 'green', 'purple', 'orange'].
    
        Returns:
            A tuple of two objects:
            - A dictionary mapping unique categories to colors from the provided color_list.
            - A new array of the input data converted to strings.
    
        Examples:
            >>> category_prep([1, 2, 2, 3, np.nan])
            ({'1': 'red', '2': 'blue', '3': 'green'}, array(['1', '2', '2', '3', 'nan'], dtype='<U3'))
        """
        if categorical_column in self.categorical_columns:
            
            cat_data = self.categorical_data[categorical_column]
        else:
            print('categorical column enetered not in categorical columns')
            return
        
        if color_list is None:
            color_list = ['red', 'blue', 'green', 'purple', 'orange']
        
        elif color_list is not None and len(color_list) < len(set(np.array(cat_data, dtype=str))):
            print('list of colors must be longer than the number of categories in the categorical data')
            return 
    
        if isinstance(cat_data, (pd.Categorical, pd.Series, np.ndarray, list)):
            new = np.array(cat_data, dtype=str)
            the_set = set(new)
            if not nan and 'nan' in the_set:
                the_set.remove('nan')
            return dict(zip(the_set, color_list)), new
    
    def plot_pca(self, categorical_column, select_categories:list=None, circle=False, components=5, **kwargs):
        """
        Perform PCA on the given data and plot the results.

        Parameters:
        -----------
        X : numpy.ndarray
            The data to perform PCA on.

        categorical_data : tuple, optional
            A tuple of two elements. The first element is a dictionary that maps category names
            to colors. The second element is a numpy.ndarray that indicates the category of each
            data point in X.

        **kwargs : dict, optional
            Additional keyword arguments to customize the plot. Supported options include:

            - xlim : list or tuple
                The limits of the x-axis.
            - ylim : list or tuple
                The limits of the y-axis.
            - title : str
                The title of the plot.
            - xlabel : str
                The label of the x-axis.
            - ylabel : str
                The label of the y-axis.

        Returns:
        --------
        numpy.ndarray
            The result of performing PCA on X.
        """
        try:
            
            categorical_data = self.category_prep(categorical_column)
        except Exception:
            print('plot PCA failed')
            return None
        
        # Create an instance of the iterative imputer
        imputer = IterativeImputer(max_iter=50, random_state=1)

        # Impute the missing values in the data
        data_imputed = imputer.fit_transform(self.X)

         # Perform PCA on the imputed data
        pca = PCA(n_components=components)
        pca.fit(data_imputed)

         # Return the PCA transformed data
        X_pca = pca.transform(data_imputed)

        fig, ax = plt.subplots()
        
        
        
        
        #Plot the PCA results with colored data points
        if categorical_data is not None:
            colors, y = categorical_data

            for category, color in colors.items():
                mask = (y == category)
                X_cat = X_pca[mask, :]
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=f'batch {category}')
                    
                if isinstance(select_categories, list) and circle:
                    selected_categories = select_categories
                    #draw circle for the selected categories
                    if category in selected_categories:
                        center = X_cat.mean(axis=0)
                        radius = np.max(np.linalg.norm(X_cat - center, axis=1))
                        #color_with_alpha = tuple(list(color) + [0.5]) # Add alpha value of 0.5
                        circle = plt.Circle(center, radius, fill=True, color=color, alpha=0.1)
                        ax.add_artist(circle)
                
                elif select_categories is None and circle:
                    center = X_cat.mean(axis=0)
                    radius = np.max(np.linalg.norm(X_cat - center, axis=1))
                    #color_with_alpha = tuple(list(color) + [0.5]) # Add alpha value of 0.5
                    circle = plt.Circle(center, radius, fill=True, color=color, alpha=0.1)
                    ax.add_artist(circle)
                
                else:
                    pass
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1])
        
        
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        
        # Set any additional plot options based on kwargs
        for key, value in kwargs.items():
            if key == 'xlim':
                ax.set_xlim(value)
            elif key == 'ylim':
                ax.set_ylim(value)
            elif key == 'title':
                ax.set_title(value)
            elif key == 'xlabel':
                ax.set_xlabel(value)
            elif key == 'ylabel':
                ax.set_ylabel(value)
        #ax.set_xlim([-150, 150]) # set the x-axis limits
        #ax.set_ylim([-150, 150]) # set the y-axis limits
        
        
        ax.legend()
        plt.show()
        
        
        return X_pca
    
    
    
def category_prep(categorical_data, nan=False, color_list=None):
    """
    Given an array-like object of categorical data, returns a dictionary mapping unique categories to colors,
    as well as a new array of the input data converted to strings.

    Args:
        categorical_data (array-like): The input categorical data. Must be an instance of pandas.Categorical,
            pandas.Series, numpy.ndarray, or list.
        nan (bool, optional): If True, retains NaN values in the unique categories set. Default is False.
        color_list (list of str, optional): A list of color names to use for the category-color mapping. If not
            provided, defaults to ['red', 'blue', 'green', 'purple', 'orange'].

    Returns:
        A tuple of two objects:
        - A dictionary mapping unique categories to colors from the provided color_list.
        - A new array of the input data converted to strings.

    Examples:
        >>> category_prep([1, 2, 2, 3, np.nan])
        ({'1': 'red', '2': 'blue', '3': 'green'}, array(['1', '2', '2', '3', 'nan'], dtype='<U3'))
    """
    
    if color_list is None:
        color_list = ['red', 'blue', 'green', 'purple', 'orange']
    
    elif color_list is not None and len(color_list) < len(set(np.array(categorical_data, dtype=str))):
        print('list of colors must be longer than the number of categories in the categorical data')
        return 

    if isinstance(categorical_data, (pd.Categorical, pd.Series, np.ndarray, list)):
        new = np.array(categorical_data, dtype=str)
        the_set = set(new)
        if not nan and 'nan' in the_set:
            the_set.remove('nan')
        return dict(zip(the_set, color_list)), new
       

def plot_pca(X, categorical_data:tuple=None, select_categories:list=None, circle=False, components=5, **kwargs):
    """
    Perform PCA on the given data and plot the results.

    Parameters:
    -----------
    X : numpy.ndarray
        The data to perform PCA on.

    categorical_data : tuple, optional
        A tuple of two elements. The first element is a dictionary that maps category names
        to colors. The second element is a numpy.ndarray that indicates the category of each
        data point in X.

    **kwargs : dict, optional
        Additional keyword arguments to customize the plot. Supported options include:

        - xlim : list or tuple
            The limits of the x-axis.
        - ylim : list or tuple
            The limits of the y-axis.
        - title : str
            The title of the plot.
        - xlabel : str
            The label of the x-axis.
        - ylabel : str
            The label of the y-axis.

    Returns:
    --------
    numpy.ndarray
        The result of performing PCA on X.
    """

    # Create an instance of the iterative imputer
    imputer = IterativeImputer(max_iter=50, random_state=1)

    # Impute the missing values in the data
    data_imputed = imputer.fit_transform(X)

     # Perform PCA on the imputed data
    pca = PCA(n_components=components)
    pca.fit(data_imputed)

     # Return the PCA transformed data
    X_pca = pca.transform(data_imputed)

    fig, ax = plt.subplots()
    
    
    
    
    #Plot the PCA results with colored data points
    if categorical_data is not None:
        colors, y = categorical_data

        for category, color in colors.items():
            mask = (y == category)
            X_cat = X_pca[mask, :]
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=f'batch {category}')
                
            if isinstance(select_categories, list) and circle:
                selected_categories = select_categories
                #draw circle for the selected categories
                if category in selected_categories:
                    center = X_cat.mean(axis=0)
                    radius = np.max(np.linalg.norm(X_cat - center, axis=1))
                    #color_with_alpha = tuple(list(color) + [0.5]) # Add alpha value of 0.5
                    circle = plt.Circle(center, radius, fill=True, color=color, alpha=0.1)
                    ax.add_artist(circle)
            
            elif select_categories is None and circle:
                center = X_cat.mean(axis=0)
                radius = np.max(np.linalg.norm(X_cat - center, axis=1))
                #color_with_alpha = tuple(list(color) + [0.5]) # Add alpha value of 0.5
                circle = plt.Circle(center, radius, fill=True, color=color, alpha=0.1)
                ax.add_artist(circle)
            
            else:
                pass
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1])
    
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    
    # Set any additional plot options based on kwargs
    for key, value in kwargs.items():
        if key == 'xlim':
            ax.set_xlim(value)
        elif key == 'ylim':
            ax.set_ylim(value)
        elif key == 'title':
            ax.set_title(value)
        elif key == 'xlabel':
            ax.set_xlabel(value)
        elif key == 'ylabel':
            ax.set_ylabel(value)
    #ax.set_xlim([-150, 150]) # set the x-axis limits
    #ax.set_ylim([-150, 150]) # set the y-axis limits
    
    
    ax.legend()
    plt.show()
    
    
    return X_pca

