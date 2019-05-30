import pandas as pd
import numpy as np
import os

def load_dataset(n_valid=10000, data_path='data_movie_lens_100k/'):
    ## Load the dataset
    try:
        # Try first, in case directory is wrong, one too deep
        train_df = pd.read_csv(os.path.join('..', data_path, "ratings_train.csv"))
        test_df = pd.read_csv(os.path.join('..', data_path, "ratings_test_masked.csv"))
    except IOError:
        train_df = pd.read_csv(os.path.join(data_path, "ratings_train.csv"))
        test_df = pd.read_csv(os.path.join(data_path, "ratings_test_masked.csv"))

    train_data_tuple = (
        train_df['user_id'].values[n_valid:],
        train_df['item_id'].values[n_valid:],
        train_df['rating'].values[n_valid:])
    valid_data_tuple = (
        train_df['user_id'].values[:n_valid],
        train_df['item_id'].values[:n_valid],
        train_df['rating'].values[:n_valid])
    test_data_tuple = (
        test_df['user_id'].values,
        test_df['item_id'].values,
        test_df['rating'].values)
    n_users = 1 + np.maximum(test_df['user_id'].max(), train_df['user_id'].max())
    n_items = 1 + np.maximum(test_df['item_id'].max(), train_df['item_id'].max())
    return train_data_tuple, valid_data_tuple, test_data_tuple, n_users, n_items

if __name__ == '__main__':
    load_dataset()
