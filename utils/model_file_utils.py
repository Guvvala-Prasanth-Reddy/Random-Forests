import pickle
import numpy as np
from tree.Forest import Forest
from tree.Leaf import Leaf
from tree.Tree import Tree


def get_tree_node_count(tree: Tree) -> int:
    """ Returns the number of nodes in the tree object.

        Parameters:
            tree: a Tree object

        Returns:
            the number of nodes contained in the provided Tree
    """

    if type(tree) is Leaf:
        return 1
    else:
        branch_node_counts = np.zeros(len(tree.branches), dtype=np.uint16)
        for idx, branch in enumerate(tree.branches):
            branch_node_counts[idx] = 1 + get_tree_node_count(branch.tree)
        return np.sum(branch_node_counts)
    

def get_tree_depth(tree: Tree) -> int:
    """ Returns the max depth of the provided tree object

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
    """ Reads a Forest object from file and returns it

        Parameters:
            filepath: the path to the file containing the Forest object

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
            filepath: the path to the file containing the tree object

        Returns
            the Tree object represented by the provided file
    """

    try:
        pickled_model = open(filepath, "rb")
        return pickle.load(pickled_model)
    except Exception:
        print('Error reading model from file. Exiting.')
        exit(1)
