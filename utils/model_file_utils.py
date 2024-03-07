import pickle
from tree.Tree import Tree
from tree.Forest import Forest
from tree.Leaf import Leaf
import numpy as np

def get_tree_depth(tree: Tree) -> int:
    """ Returns the max depth of the provided tree model

        Parameters:
            tree: A Tree object

        Returns:
            the max depth of the provided tree object
    """

    if type(tree) is Leaf:
        return 0
    else:
        tree_path_depths = np.zeros(len(tree.branches), dtype=np.uint16)
        for idx, branch in enumerate(tree.branches):
            tree_path_depths[idx] = 1 + get_tree_depth(branch.tree)
        return np.max(tree_path_depths)

def read_forest_model(filepath: str) -> Forest:
    """ Reads a Forest object from file and returns it.

        Parameters:
            filepath: the path to the file containing the forest model

        Returns
            the Forest object represented by the provided file
    """

    try:
        pickled_model = open(filepath, "rb")
        return pickle.load(pickled_model)
    except Exception:
        print('Error reading model from file. Exiting.')
        exit(1)

def read_tree_model(filepath: str) -> Tree:
    """ Reads a Tree object from file and returns it.

        Parameters:
            filepath: the path to the file containing the tree model

        Returns
            the Tree object represented by the provided file
    """

    try:
        pickled_model = open(filepath, "rb")
        return pickle.load(pickled_model)
    except Exception:
        print('Error reading model from file. Exiting.')
        exit(1)
