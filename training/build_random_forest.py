from tree.Forest import Forest
from tree.Node import Node
from utils.consts import *
from training.build_decision_tree import build_tree
import pandas as pd
from utils.dataframeutils import *
from validation.validate import get_tree_acc, get_forest_acc
import time
from training.build_decision_tree import handle_missing_values
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


def build_random_forest(df: pd.DataFrame, num_trees=no_of_trees_in_forest) -> Forest:
    """ Constructs a random forest using the provided training data

        Parameters:
            df: a dataframe of training data
            num_trees: number of trees to include in the forest

        Returns:
            a Forest object representing the random forest trained on the
                provided data
    """

    forest = Forest()
    for i in range(num_trees):

        # this is the bagged data used for training this tree of the forest
        (tree_train_data, tree_test_data) = split_data(df, 0.75, 0.1, True)
         
        tree = build_tree(
            tree_train_data,
            split_metric='misclassification'
        )

        tree_acc = get_tree_acc(tree, tree_test_data)
        forest.add_tree(tree)

        print(f'Tree {i + 1} of {num_trees} completed with accuracy {tree_acc}')

    return forest


if __name__ == "__main__":

    t0 = time.time()

    # read entire training dataset and handle missing values
    whole_training_data = pd.read_csv('data/train.csv')
    whole_training_data = handle_missing_values(whole_training_data)

    # divide into separate training and testing datasets
    (training_data, testing_data) = split_data(whole_training_data, 1, 1, False)

    forest = build_random_forest(training_data, num_trees=no_of_trees_in_forest)
    
    forest_build_time = time.time() - t0
    t0 = time.time()
    print(f'Forest build in {forest_build_time} seconds')

    # get the accuracy of the forest
    forest_acc = get_forest_acc(forest, testing_data)
    print(f'Forest accuracy {forest_acc} in {time.time() - t0} seconds')
    
    file = open(f'models/forest-{no_of_trees_in_forest}-trees-{forest_acc}-acc', 'wb')
    pickle.dump(forest, file)
    file.close()

