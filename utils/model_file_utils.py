import pickle
from tree.Tree import Tree
from tree.Forest import Forest

def read_tree_model(filepath: str) -> Forest:
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

