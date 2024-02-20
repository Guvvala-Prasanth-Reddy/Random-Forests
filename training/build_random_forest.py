from tree.Tree import Tree
from tree.Forest import Forest
from tree.Leaf import Leaf
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
from time import gmtime, strftime
import os

sys.setrecursionlimit( 10**8)
def dfs(tree , level ):
    if( type(tree) is Node ):
        print("feature : " , tree.feature , "at level" , level)
        for i in tree.branches:
            print( i.feature_value)
            dfs(i.tree , level+1)
    else:
        print(tree.target)

def build_random_forest(df: pd.DataFrame, num_trees=no_of_trees_in_forest):

    forest = Forest()
    for i in range(num_trees):

        # this is the bagged data used for training this tree of the forest
        (tree_data_train, _) = split_data(df, 0.75, 0.1, True)

        #forest.add_tree(build_decision_tree(tree_data_train))
        #return forest



        # k-fold cross validation
        num_k_fold = 5
        k_fold_data = train_test_data_split(tree_data_train, mode=stratified_k_fold, number_of_folds=num_k_fold)

        best_tree = None
        best_tree_acc = float('-inf')

        counter = 0
        for value in k_fold_data:

            print(f'Started k fold tree {counter} of {num_k_fold}')

            training_indices = value[0]
            testing_indicies = value[1]

            tree = build_tree(
                tree_data_train.iloc[training_indices],
                split_metric='misclassification'
            )

            tree_acc = get_tree_acc(tree, tree_data_train.iloc[testing_indicies])

            print(f'Finished k fold tree {counter} of {num_k_fold}')
            counter += 1

            if tree_acc > best_tree_acc:
                best_tree = tree
                best_tree_acc = tree_acc

        print(f'Tree {i} has accuracy {best_tree_acc}')

        forest.add_tree(best_tree)

    return forest

if __name__ == "__main__":


    t0 = time.time()

    # read entire training dataset and handle missing values
    whole_training_data = pd.read_csv('data/train.csv')
    whole_training_data = handle_missing_values(whole_training_data)

    # divide into separate training and testing datasets
    (training_data, testing_data) = split_data(whole_training_data, 1, 1, False)

    num_trees = 3
    forest = build_random_forest(training_data, num_trees=num_trees)

    # get the accuracy of the forest
    forest_acc = get_forest_acc(forest, testing_data)
    print(f'Forest accuracy: {forest_acc}')
    
    # save tree model to file
    s = strftime("%a, %d %b %Y %H:%M:%S", gmtime())
    
    file = open(f'models/forest-{s}-{num_trees}-trees-{forest_acc}-acc', 'wb')
    pickle.dump(forest, file)
    file.close()

    print(f'Forest build in {time.time() - t0} seconds')
