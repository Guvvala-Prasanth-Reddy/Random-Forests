from utils.consts import categorical_features
from tree.Leaf import Leaf
from tree.Tree import Tree
import numpy as np
import statistics as st
import pandas as pd

def tree_classify(df: pd.DataFrame, tree: Tree):
    """ Returns the decision tree classification of the given instance

        Parameters:
            df: a single row Pandas dataframe
            tree: Either a Leaf or Node object representing the decision
                tree used to make the classification
    """

    # case 1: tree is a leaf
    if type(tree) is Leaf:
        return tree.target

    # case 2: tree is a node
    if tree.feature in categorical_features:
        for branch in tree.branches:
            if list(df[tree.feature])[0] == branch.feature_value:
                return tree_classify(df, branch.tree)
    else:
        split_cutoff_value = tree.branches[0].feature_value.replace('<', '')
        feature_value = list(df[tree.feature])[0]

        if feature_value < float(split_cutoff_value):
            return tree_classify(df, tree.branches[0].tree)
        else:
            return tree_classify(df, tree.branches[1].tree)
    
        

def forest_classify(df, forest):
    
    classification_results = []
    for tree in forest.trees:
        classification_result = tree_classify(df, tree)
        classification_results.append(classification_result)

    return st.mode(np.array(classification_results))

