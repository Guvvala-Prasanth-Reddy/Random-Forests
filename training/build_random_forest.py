from tree.Tree import Tree
from tree.Forest import Forest
from utils.consts import *
from training.build_decision_tree import build_tree
import pandas as pd
from utils.dataframeutils import *
from validation.validate import get_tree_acc

def build_random_forest(num_trees=no_of_trees_in_forest):

    training_data = pd.read_csv('data/train.csv')
    training_data = training_data.sample(frac=0.1)
    forest = Forest()
    for i in range(num_trees):

        # this is the bagged data used for training this tree of the forest
        (tree_data_train, _) = split_data(training_data, 0.7, 0.75)

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

            tree_acc = get_tree_acc(tree, tree_data_train.iloc[testing_indicies])
            if tree_acc > best_tree_acc:
                best_tree = tree
                best_tree_acc = tree_acc

        print(f'Tree {i} has accuracy {best_tree_acc}')

        forest.add_tree(best_tree)

    return forest

if __name__ == "__main__":
    build_random_forest(1)

