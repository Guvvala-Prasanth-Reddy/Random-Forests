from tree.Forest import Forest
from tree.Node import Node
from utils.consts import *
from training.build_decision_tree import build_tree
import pandas as pd
from utils.dataframeutils import *
from validation.validate import get_tree_acc, get_forest_acc
import time
from utils.dataframeutils import handle_missing_values
import sys
import pickle

sys.setrecursionlimit( 10**8)
def dfs(tree , level ):
    if( type(tree) is Node ):
        print("feature : " , tree.feature , "at level" , level)
        for i in tree.branches:
            print( i.feature_value)
            dfs(i.tree , level+1)
    else:
        print(tree.target)


def build_random_forest(training_df: pd.DataFrame, testing_df: pd.DataFrame, num_trees=no_of_trees_in_forest) -> Forest:
    """ Constructs a random forest using the provided training data

        Parameters:
            df: a dataframe of training data
            num_trees: number of trees to include in the forest

        Returns:
            a Forest object representing the random forest trained on the
                provided data
    """

    cols_to_keep_frac = 8/25
    rows_to_keep_frac = 1.0
    
    forest = Forest()
    for i in range(num_trees):

        # column bagging
        feature_col_names = list(training_df.columns)
        feature_col_names.remove(target_column)
        feature_col_names.remove('TransactionID')
        col_names_to_drop = random.sample(feature_col_names, int((1 - cols_to_keep_frac) * len(feature_col_names)))
        sampled_training_df = training_df.drop(col_names_to_drop, axis=1, inplace=False)

        # row bagging --> Stratify or use all positive samples
        is_fraud_rows = sampled_training_df.loc[sampled_training_df[target_column] == 1]
        is_not_fraud_rows = sampled_training_df.loc[sampled_training_df[target_column] == 0]
        sampled_is_not_fraud_rows = is_not_fraud_rows.sample(frac=rows_to_keep_frac, replace=True)
        sampled_training_df = pd.concat([is_fraud_rows, sampled_is_not_fraud_rows], axis=0).reset_index(drop=True)

        # determine the imbalance factor using number of positive and negative targets
        imbalance_factor = len(sampled_is_not_fraud_rows) / len(is_fraud_rows)

        t0 = time.time()
        tree = build_tree(
            sampled_training_df,
           set(),
            split_metric='entropy',
            imbalance_factor=imbalance_factor
        )

        (tree_err, tree_acc) = get_tree_acc(tree, testing_df)
        forest.add_tree(tree)

        print(f'Tree {i + 1} of {num_trees} completed with accuracy {tree_acc} in {time.time() - t0} seconds')

    return forest


if __name__ == "__main__":

    t0 = time.time()

    # read entire training dataset and handle missing values
    whole_training_data = pd.read_csv('data/train.csv')
    whole_training_data = handle_missing_values(whole_training_data)

    # divide into separate training and testing datasets
    (training_data, testing_data) = split_data(whole_training_data, 1, 1, False)

    forest = build_random_forest(training_data, testing_data, num_trees=no_of_trees_in_forest)
    
    forest_build_time = time.time() - t0
    t0 = time.time()
    print(f'Forest build in {forest_build_time} seconds')

    # get the accuracy of the forest
    (forest_err, forest_acc) = get_forest_acc(forest, testing_data)
    print(f'Forest accuracy {forest_acc} in {time.time() - t0} seconds')
    
    file = open(f'models/forest-{no_of_trees_in_forest}-trees-{forest_acc}-acc', 'wb')
    pickle.dump(forest, file)
    file.close()

