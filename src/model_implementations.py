import numpy as np
import pandas as pd

import torch

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression

import pickle

def split_by_labels(df, n_split):
    """
    Splits a DataFrame into `n_split` parts with the same proportion of labels in each.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame, must have a 'Labels' column.
        n_split (int): The number of splits.

    Returns:
        list of pd.DataFrame: List of DataFrames, one for each split.
    """

    if 'Labels' not in df.columns:
        raise ValueError("The DataFrame must have a 'Labels' column.")
    
    # shuffling data before doing the even proportion
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    splits = []
    for label, group in df.groupby('Labels'):
        group_splits = np.array_split(group, n_split)
        for i, split in enumerate(group_splits):
            if len(splits) <= i:
                splits.append(split)
            else:
                splits[i] = pd.concat([splits[i], split], ignore_index=True)
    
    reconcatenated_df = pd.concat(splits[:-1], ignore_index=True)
    
    return reconcatenated_df, splits[-1]


class Double_Net(torch.nn.Module):

    def __init__(self, input_size, K, dropout_rate=0.5):
        super(Double_Net, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, K), 
            torch.nn.BatchNorm1d(K),         
            torch.nn.ReLU(),                  
            torch.nn.Dropout(dropout_rate),    
            
            torch.nn.Linear(K, K),       
            torch.nn.BatchNorm1d(K),         
            torch.nn.ReLU(),                   
            torch.nn.Dropout(dropout_rate),    
            
            torch.nn.Linear(K, K),       
            torch.nn.BatchNorm1d(K),         
            torch.nn.ReLU(),                   
            torch.nn.Dropout(dropout_rate), 

            torch.nn.Linear(K, 1),
        )
    
    def forward(self, x):
        return self.network(x)