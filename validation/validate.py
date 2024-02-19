import pandas as pd
from tree.Tree import Tree
from utils.consts import target_column
from classification.classify import tree_classify, forest_classify
from tree.Forest import Forest

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
    print(true_targets[true_targets.isna().any(axis=1)])
    print(pred_targets[pred_targets.isna().any(axis=1)])
    
    true_target_name = list(true_targets.columns)[0]
    # pred_target_name = list(pred_targets.columns)[0]
    # print(true_targets , pred_targets)

    false_negative_count = 0
    false_positive_count = 0
    print(true_targets.index , pred_targets.index)
    for i in range(len(true_targets)):
        # print(i , type(i))
        true_target_val = true_targets.loc[i][target_column]
        pred_target_val = pred_targets.loc[i][target_column]
        print(true_target_val , pred_target_val)
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
            return float('-inf')

        predicted_targets.append( predicted_target )

    predicted_targets = pd.DataFrame( data = predicted_targets  , columns = [target_column])
    balanced_err = get_balanced_error(true_targets, predicted_targets)
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

