from utils.consts import categorical_features
from tree.Leaf import Leaf

def tree_classify(df, tree):
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
            if df[tree.feature] == branch.feature_value:
                return tree_classify(df, branch.tree)
    else:
        split_cutoff_value = tree.branches[0].feature_value.replace('<', '')
        feature_value = float(df[tree.feature])

        if feature_value < split_cutoff_value:
            return tree_classify(df, tree.branches[0].tree)
        else:
            return tree_classify(df, tree.branches[1].tree)
        

def forest_classify(df, forest):
    pass

