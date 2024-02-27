import pandas as pd
from tree.Tree import Tree
from utils.consts import target_column
from classification.classify import tree_classify, forest_classify
from tree.Forest import Forest


def get_balanced_error_efficient(true_targets: pd.DataFrame, pred_targets: pd.DataFrame) -> float:
    print(true_targets)
    positive_indices = true_targets.loc[true_targets[target_column] == 1].index
    negative_indices = true_targets.loc[true_targets[target_column] == 0].index

    positive_true_targets = true_targets.iloc[positive_indices]
    negative_true_targets = true_targets.iloc[negative_indices]

    positive_predicted_targets = pred_targets.iloc[positive_indices]
    negative_predicted_targets = pred_targets.iloc[negative_indices]

    # get the number of matches between the positive and negative predictions
    num_positive_matches = (positive_true_targets == positive_predicted_targets).sum()
    num_negative_matches = (negative_true_targets == negative_predicted_targets).sum()

    false_positive_rate = abs((len(true_targets) - num_negative_matches)) / len(true_targets)
    false_negative_rate = abs((len(true_targets) - num_positive_matches)) / len(true_targets)

    return 0.5 * (false_negative_rate + false_positive_rate)


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
    
    true_target_name = list(true_targets.columns)[0]

    false_negative_count = 0
    false_positive_count = 0
    for i in range(len(true_targets)):
        true_target_val = true_targets.loc[i][target_column]
        pred_target_val = pred_targets.loc[i][target_column]
        if true_target_val == 1 and pred_target_val != 1:
            false_negative_count += 1
        if true_target_val == 0 and pred_target_val != 0:
            false_positive_count += 1

    false_negative_rate = false_negative_count / len(true_targets.loc[true_targets[true_target_name] == 1])
    false_positive_rate = false_positive_count / len(true_targets.loc[true_targets[true_target_name] == 0])

    return 0.5 * (false_negative_rate + false_positive_rate)


def get_tree_acc(tree: Tree, df: pd.DataFrame) -> float:
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
    #balanced_err = get_balanced_error(true_targets, predicted_targets)
    balanced_err = get_balanced_error_efficient(true_targets, predicted_targets)
    return 1 - balanced_err

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

