import pandas as pd
from tree.Tree import Tree
from utils.consts import target_column
from classification.classify import tree_classify

def get_balanced_error(true_targets: pd.DataFrame, pred_targets: pd.DataFrame) -> float:
    """ Returns the balanced error 

        Parameters:
            true_targets: the actual target values of the provided instances
                in the testing data
            predicted_targets: the target values of the provided instances in
                the testing data predicted by the model

        Returns  
            the balanced error calculated from the provided predicted values and
            true values
    """
    
    true_target_name = true_targets.columns[0]
    pred_target_name = pred_targets.columns[0]

    false_negative_count = 0
    false_positive_count = 0
    for i in range(len(true_targets)):
        true_target_val = true_targets[true_target_name][i]
        pred_target_val = pred_targets[pred_target_name][i]

        if true_target_val == 1 and pred_target_val != 1:
            false_negative_count += 1
        if true_target_val == 0 and pred_target_val != 0:
            false_positive_count += 1

    false_negative_rate = false_negative_count / len(true_targets.loc[true_targets[true_target_name] == 1])
    false_positive_rate = false_positive_count / len(true_targets.loc[true_targets[true_target_name] == 0])

    return 0.5 * (false_negative_rate + false_positive_rate)


def get_tree_acc(tree: Tree, df: pd.DataFrame):
    """

        Parameters:
            tree: a tree object
            df: a dataframe with included correct target values

        Returns:
            The balanced accuracy of the tree on the provided data
    """

    true_targets = df.get(target_column)
    predicted_targets = []

    df.reset_index()

    for row_idx in range(len(df.index) - 1):
        predicted_target = tree_classify(df[row_idx], tree)
        predicted_targets.append(predicted_target)

    balanced_err = get_balanced_error(true_targets, predicted_targets)
    return 1 - balanced_err