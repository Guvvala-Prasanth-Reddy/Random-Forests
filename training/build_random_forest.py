from tree.Tree import Tree
from tree.Forest import Forest
from tree.Leaf import Leaf
from tree.Node import Node
from utils.consts import *
from training.build_decision_tree import build_tree
import pandas as pd
from utils.dataframeutils import *
from validation.validate import get_tree_acc
import time
from training.build_decision_tree import handle_missing_values
import sys
sys.setrecursionlimit( 10**8)
def dfs(tree , level ):
    if( type(tree) is Node ):
        print("feature : " , tree.feature , "at level" , level)
        for i in tree.branches:
            print( i.feature_value)
            dfs(i.tree , level+1)
    else:
        print(tree.target)
def build_random_forest(num_trees=no_of_trees_in_forest):

    start_time = time.time()

    df = pd.read_csv("data/train.csv")
    print(f'Time to read file: {time.time() - start_time} seconds')
    start_time = time.time()

    df = handle_missing_values(df)
    print(f'Time to handle missing values: {time.time() - start_time} seconds')
    print(df)
    start_time = time.time()
    training_data = df
    print(training_data.isna())
    forest = Forest()
    for i in range(num_trees):

        # this is the bagged data used for training this tree of the forest
        (tree_data_train, _) = split_data(training_data, 0.1, 0.75)

        # k-fold cross validation
        num_k_fold = 5
        k_fold_data = train_test_data_split(tree_data_train, mode=stratified_k_fold, number_of_folds=num_k_fold)

        best_tree = None
        best_tree_acc = float('-inf')

        for value in k_fold_data:

            training_indices = value[0]
            testing_indicies = value[1]

            tree = build_tree(
                tree_data_train.iloc[training_indices]
            )
            dfs(tree,0)
            print("""another treee """)

            tree_acc = get_tree_acc(tree, tree_data_train.iloc[testing_indicies])
            if tree_acc > best_tree_acc:
                best_tree = tree
                best_tree_acc = tree_acc

        print(f'Tree {i} has accuracy {best_tree_acc}')

        forest.add_tree(best_tree)

    return forest

if __name__ == "__main__":
    build_random_forest(1)

