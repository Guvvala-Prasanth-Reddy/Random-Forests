import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from classification.classify import tree_classify, forest_classify
from tree.Forest import Forest
from tree.Tree import Tree
from utils.consts import target_column

def get_balanced_accuracy_efficient(true_targets: pd.DataFrame , pred_targets : pd.DataFrame ) -> float:
    """ Calculates the balanced accuracy of the provided predictions and true targets

        NOTE: This function is not used anymore as we switched to using sklearn's implementation
            for calculating balanced accuracy. We choose to keep this function in the codebase
            instead of removing it to demonstrate our understanding of how this calculation works.

        Parameters:
            true_targets: a DataFrame containing the true classes of a dataset
            pred_targets: a DataFrame containing predicted target classes of a dataset

        Returns:
            the balanced accuracy of the class predictions

    """
    
    # determine the positive and negative indices of the true targets
    positive_indices = true_targets.loc[true_targets[target_column] == 1].index
    negative_indices = true_targets.loc[true_targets[target_column] == 0].index

    # select subsets of negative and positive classes from the true class
    # and predictions dataframes
    positive_true_targets = true_targets.iloc[positive_indices]
    negative_true_targets = true_targets.iloc[negative_indices]

    positive_predicted_targets = pred_targets.iloc[positive_indices]
    negative_predicted_targets = pred_targets.iloc[negative_indices]

    # calculate the number of matches between the predictions and the true targets
    true_positive_matches = (positive_true_targets == positive_predicted_targets).sum()
    false_negative_matches = (negative_true_targets == negative_predicted_targets).sum()

    true_negative_matches = (positive_true_targets != positive_predicted_targets).sum()
    false_positive_matches = ( negative_true_targets != negative_predicted_targets).sum()

    # calculate true positive and true negative rates
    true_positive_rate = true_positive_matches / (true_positive_matches + false_negative_matches)
    true_negative_rate = true_negative_matches / ( true_negative_matches + false_positive_matches)

    return 0.5 * ( true_negative_rate + true_positive_rate)


def get_balanced_error(true_targets: pd.DataFrame, pred_targets: pd.DataFrame) -> float:
    """ Calculates the balanced error of the provided predictions using the provided
        true target classes

        Parameters:
            true_targets: a DataFrame containing the true classes of a dataset
            pred_targets: a DataFrame containing predicted target classes of a dataset

        Returns:
            the balanced accuracy of the class predictions
    """

    # determine the indices of the positive and negative instances of each true class
    positive_indices = true_targets.loc[true_targets[target_column] == 1].index
    negative_indices = true_targets.loc[true_targets[target_column] == 0].index

    # determine number of false positives and false negatives
    num_false_positives = (true_targets.iloc[negative_indices] != pred_targets.iloc[negative_indices]).sum()
    num_false_negatives = (true_targets.iloc[positive_indices] != pred_targets.iloc[positive_indices]).sum()

    # calculate false positive and false negative rates
    false_positive_rate = num_false_positives / len(true_targets.loc[true_targets[target_column] == 0])
    false_negative_rate = num_false_negatives / len(true_targets.loc[true_targets[target_column] == 1])

    return 0.5 * (false_positive_rate + false_negative_rate)


def get_tree_acc(tree: Tree, df: pd.DataFrame) -> tuple[float, float]:
    """ Returns the balanced error and balanced accuracy of the provided
        tree on the provided dataset.

        Parameters:
            tree: a Tree object
            df: a DataFrame with included correct target values

        Returns:
            a tuple of floats, representing the tree's balanced error and
                balanced accuracy respectively on the provided dataset
    """

    # separate the target class column from the provided DataFrame
    true_targets = pd.DataFrame(data = list(df[target_column]) , columns = [target_column])

    # classify each instance of the DataFrame, and handle NaN values
    predicted_targets = df.apply(tree_classify, args=(tree,), axis=1)
    predicted_targets.replace( { np.nan : 2 }, inplace=True)
    predicted_targets_df = predicted_targets.to_frame(name=target_column)

    # calculate balanced error and accuracy using a mix of sklearn and our own
    # implemented functions
    balanced_err = get_balanced_error(true_targets, predicted_targets_df)
    balanced_accuracy = balanced_accuracy_score(true_targets , predicted_targets_df)
    return (balanced_err, balanced_accuracy)


def get_forest_acc(forest: Forest, df: pd.DataFrame) -> tuple[float, float]:
    """ Returns the balanced error and balanced accuracy of the provided
        forest on the provided dataset.

        Parameters:
            forest: a Forest object
            df: a DataFrame with included correct target values

        Returns:
            a tuple of floats, representing the forest's balanced error and
                balanced accuracy respectively on the provided dataset
    """

    # separate the target class column from the provided DataFrame
    true_targets = pd.DataFrame(data = list(df[target_column]) , columns = [target_column])

    # classify each instance of the DataFrame, and handle NaN values
    predicted_targets = df.apply(forest_classify, args=(forest,), axis=1)
    predicted_targets.replace( {np.nan : 2 }, inplace=True)
    predicted_targets_df = predicted_targets.to_frame(name=target_column)

    # calculate balanced error and accuracy using a mix of sklearn and our own
    # implemented functions
    balanced_err = get_balanced_error(true_targets, predicted_targets_df)
    balanced_acc = balanced_accuracy_score(true_targets, predicted_targets_df)
    return (balanced_err, balanced_acc)

