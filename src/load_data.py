"""Module to load data."""
import numpy as np
import pandas as pd


def read(fp):
    """Read a csv file to dataframe.

    Args
    ----
    fp: string, filepath to the csv file

    Returns
    -------
    DataFrame
    """
    df = (pd.read_csv(fp, low_memory=False)
          .loc[lambda x: x['related'] == 1]
          .drop(['id', 'split', 'original', 'genre', 'related'], axis=1))

    return df


def load_dataset(fp, col):
    """Loads the training and validation dataset.

    Args
    ----
    fp: string, csv file
    col: string, column name to use in classification

    Returns
    -------
    tuple (of messages and labels)
    """
    df = read(fp)
    return df['message'].values, df[col].values


def make_training_dataset(csv1, csv2, col):
    """Merges two files in the same format to create a dataset."""
    data1 = load_dataset(csv1, col)
    data2 = load_dataset(csv2, col)
    X = np.concatenate((data1[0], data2[0]), axis=0)
    y = np.concatenate((data1[1], data2[1]), axis=0)
    return X, y
