import pandas as pd

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
    print(f'  fpr: {false_positive_rate}')
    print(f'  fnr: {false_negative_rate}')

    return 0.5 * (false_negative_rate + false_positive_rate)