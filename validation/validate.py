import pandas as pd
from tree.Tree import Tree
from utils.consts import target_column
from classification.classify import tree_classify, forest_classify
from tree.Forest import Forest
import numpy as np
from sklearn.metrics import balanced_accuracy_score

def get_balanced_accuracy_efficient(true_targets: pd.DataFrame , pred_targets : pd.DataFrame ) -> float:
    "calculates balanced accuracy"
    positive_indices = true_targets.loc[true_targets[target_column] == 1].index
    negative_indices = true_targets.loc[true_targets[target_column] == 0].index

    positive_true_targets = true_targets.iloc[positive_indices]
    negative_true_targets = true_targets.iloc[negative_indices]

    positive_predicted_targets = pred_targets.iloc[positive_indices]
    negative_predicted_targets = pred_targets.iloc[negative_indices]

    true_positive_matches = (positive_true_targets == positive_predicted_targets).sum()
    false_negative_matches = (negative_true_targets == negative_predicted_targets).sum()

    true_negative_matches = (positive_true_targets != positive_predicted_targets).sum()
    false_positive_matches = ( negative_true_targets != negative_predicted_targets).sum()

    true_positive_rate = true_positive_matches / (true_positive_matches + false_negative_matches)
    true_negative_rate = true_negative_matches / ( true_negative_matches + false_positive_matches)

    return 0.5 * ( true_negative_rate + true_positive_rate)


def get_balanced_error(true_targets: pd.DataFrame, pred_targets: pd.DataFrame) -> float:

    positive_indices = true_targets.loc[true_targets[target_column] == 1].index
    negative_indices = true_targets.loc[true_targets[target_column] == 0].index

    num_false_positives = (true_targets.iloc[negative_indices] != pred_targets[negative_indices]).sum()
    num_false_negatives = (true_targets.iloc[positive_indices] != pred_targets[positive_indices]).sum()

    false_positive_rate = num_false_positives / len(true_targets.loc[true_targets[target_column] == 0])
    false_negative_rate = num_false_negatives / len(true_targets.loc[true_targets[target_column] == 1])

    return 0.5 * (false_positive_rate + false_negative_rate)



def get_tree_acc(tree: Tree, df: pd.DataFrame) -> tuple :
    """ Returns the balanced accuracy of the provided tree on the provided dataset.

        Parameters:
            tree: a tree object
            df: a dataframe with included correct target values

        Returns:
            The balanced accuracy of the tree on the provided data
    """

    true_targets = pd.DataFrame(data = list(df[target_column]) , columns = [target_column])
    predicted_targets = []

    for row_idx in df.index.values:
        predicted_target = tree_classify(df.loc[[row_idx]], tree)

        if predicted_target is None:
            predicted_target = 2

        predicted_targets.append( predicted_target )

    predicted_targets = pd.DataFrame( data = predicted_targets  , columns = [target_column])
    balanced_err = get_balanced_error(true_targets, predicted_targets)
    balanced_accuracy = get_balanced_accuracy_score( true_targets , predicted_targets)
    return (balanced_err , balanced_accuracy)

def get_forest_acc(forest: Forest, df: pd.DataFrame) -> float:
    """ Returns the balanced accuracy of the provided forest on the provided dataset.

        Parameters:
            forest: a Forest object
            df: a dataframe with included correct target values

        Returns:
            The balanced accuracy of the tree on the provided data
    """

    true_targets = pd.DataFrame(data = list(df[target_column]) , columns = [target_column])
    predicted_targets = []

    for row_idx in df.index.values:
        predicted_target = forest_classify(df.loc[[row_idx]], forest)
        predicted_targets.append( predicted_target )

    predicted_targets = pd.DataFrame( data = predicted_targets  , columns = [target_column])
    balanced_err = get_balanced_error(true_targets, predicted_targets)
    return 1 - balanced_err

