
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
    for branch in tree.branches:
        if df[tree.feature] == branch.feature_value:
            return tree_classify(df, branch.tree)
        

def forest_classify(df, forest):
    pass

