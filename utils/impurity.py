from typing import Callable
import pandas as pd
from numba import jit
import numpy as np
from utils.consts import *

def get_info_gain_categorical(impurity_function, df: pd.DataFrame, feature:str, imbalance_factor=1.0) -> float:
    """ Returns the information gain of the provided dataset considering a split on the provided
        categorical feature.

        Parameters:
            impurity_function: the function to use in determining the impurity of the proposed featuure split
                (entropy, gini_index, or missclassification_error)
            df: The data before splitting on the feature
            feature: the feature being considered for splitting criteria
            imbalance_factor: the imbalance between classes in the original dataset

        Returns:
            the information gain of splitting the dataset along the indicated categorical feature
    """

    info_gain = impurity_function(df)
    for feature_value in pd.unique(df[feature]):
        info_gain -= len(df.loc[df[feature] == feature_value]) / len(df) * impurity_function(df.loc[df[feature] == feature_value], imbalance_factor)
    return info_gain


def get_info_gain_continuous_cuda(df: pd.DataFrame, feature: str, split_metric: str, imbalance_factor: float=1.0) -> tuple[float, float]:
    """ Returns the information gain of the provided continuous feature using GPU
    
        Parameters:
            df: The data before splitting on the feature
            feature: the feature being considered for splitting criteria
            split_metric: the information gain criteria method as a string
            imbalance_factor: the imbalance between classes in the original dataset

        Returns:
            a tuple containing the information gain of splitting the dataset along the
                indicated continuous feature as well as the value across which the data
                should be split to maximize information gain
    """
    df_sorted = df.sort_values(by=feature, inplace=False)
    return cuda_info_gain(df_sorted[feature].to_numpy(), df_sorted[target_column].to_numpy(), split_metric, imbalance_factor=imbalance_factor)


@jit(target_backend='cuda', nopython=True)                         
def cuda_info_gain(feature_array: np.array, target_array: np.array, split_metric: str='entropy', imbalance_factor: float=1.0) -> tuple[float, float]:
    """ Returns the information gain of the provided continuous feature using GPU

        NOTE: Typically, the code in this function would be divided into smaller
            component functions. However, the JIT decorator was having trouble with
            function calls within this function so it was maintained as a monolith.
    
        Parameters:
            feature_array: the array of feature values
            target_array: the array of target classes
            split_metric: the information gain criteria method as a string
            imbalance_factor: the imbalance between classes in the original dataset

        Returns:
            a tuple containing the information gain of splitting the dataset along the
                indicated continuous feature as well as the value across which the data
                should be split to maximize information gain
    """

    # caching these values so they are only computed once
    feature_array_len = len(feature_array)
    num_unique_targets = len(np.unique(target_array))
    unique_targets = np.unique(target_array)
    min_feature_val = np.min(feature_array)
    max_feature_val = np.max(feature_array)

    max_info_gain = 0
    max_info_gain_cutoff = 0

    # determine impurity of whole feature column
    target_proportions = np.zeros(num_unique_targets)
    for (idx, target) in enumerate(unique_targets):
        target_proportions[idx] = len(feature_array[feature_array == target]) / feature_array_len
    feature_array_impurity = 0
    if split_metric == 'entropy':
        feature_array_impurity = -1 * np.sum(np.multiply(target_proportions, np.log2(target_proportions)))
    elif split_metric == 'gini':
        feature_array_impurity = 1 - np.sum(np.square(target_proportions))
    else:
        feature_array_impurity = 1 - np.max(target_proportions)

    target_positive_indices = np.asarray(target_array == 1).nonzero()[0]

    # for each possible split index (those where adjacent feature values correspond to
    # differing classes), compute the information gain from performing a split there and
    # maintain a record of the highest information gain
    for positive_index in target_positive_indices:
        for adjacent_index in [positive_index - 1, positive_index + 1]:
            if (adjacent_index > 0 and adjacent_index < len(feature_array) - 1) and (target_array[adjacent_index] != target_array[positive_index]):
                split_value = 0.5 * (feature_array[positive_index] + feature_array[adjacent_index])

                # do this check to avoid a split where one branch is empty
                if split_value == min_feature_val or split_value == max_feature_val:
                    continue    
                    
                greater_than_split_indices = np.asarray(feature_array >= split_value).nonzero()[0]
                less_than_split_indices = np.asarray(feature_array < split_value).nonzero()[0]
    
                target_array_less_than = target_array[less_than_split_indices]
                target_array_greater_than = target_array[greater_than_split_indices]
    
                # determine impurity of less than feature set
                less_than_feature_proportions = np.zeros(num_unique_targets)
                for (idx, target) in enumerate(unique_targets):
                    applied_imbalance_factor = 1.0 if target == 0 else imbalance_factor
                    less_than_feature_proportions[idx] = applied_imbalance_factor * len(target_array_less_than[target_array_less_than == target]) / feature_array_len
                less_than_split_impurity = 0
                if split_metric == 'entropy':
                    less_than_split_impurity = -1 * np.sum(np.multiply(less_than_feature_proportions, np.log2(less_than_feature_proportions)))
                elif split_metric == 'gini':
                    less_than_split_impurity = 1 - np.sum(np.square(less_than_feature_proportions))
                else:
                    less_than_split_impurity = 1 - np.max(less_than_feature_proportions)
    
                # determine impurity of greater than feature set
                greater_than_feature_proportions = np.zeros(num_unique_targets)
                for (idx, target) in enumerate(unique_targets):
                    applied_imbalance_factor = 1.0 if target == 0 else imbalance_factor
                    greater_than_feature_proportions[idx] = applied_imbalance_factor * len(target_array_greater_than[target_array_greater_than == target]) / feature_array_len
                greater_than_split_impurity = 0
                if split_metric == 'entropy':
                    greater_than_split_impurity = -1 * np.sum(np.multiply(greater_than_feature_proportions, np.log2(greater_than_feature_proportions)))
                elif split_metric == 'gini':
                    greater_than_split_impurity = 1 - np.sum(np.square(greater_than_feature_proportions))
                else:
                    greater_than_split_impurity = 1 - np.max(greater_than_feature_proportions)

                # if a particular split is empty (all data in one branch), the impurity of the empty
                # split will be NaN due to log2(0)
                if np.isnan(less_than_split_impurity):
                    less_than_split_impurity = 0
                if np.isnan(greater_than_split_impurity):
                    greater_than_split_impurity = 0                
                
                info_gain = feature_array_impurity
                info_gain -= len(greater_than_split_indices) / feature_array_len * greater_than_split_impurity
                info_gain -= len(less_than_split_indices) / feature_array_len * less_than_split_impurity

                # only keep info  gain if it higher than any information gain previously seen for this feature 
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    max_info_gain_cutoff = split_value

    return (max_info_gain, max_info_gain_cutoff)


def get_info_gain_continuous(impurity_function: Callable[[pd.DataFrame], float] , df: pd.DataFrame , feature:str) -> tuple[float, float]:
    """ Returns the information gain of the provided dataset considering a split on the provided
        continuous feature.

    Parameters:
        impurity_function: impurity function eg : entropy , gini_index , missclassification_error
        df: The data before splitting on feature
        feature: the feature i.e being considered for splitting criteria

    Returns:
        (float, float): A tuple of floats, representing the maximum information gain possible splitting
            on the provided feature and the value upon which to split the feature to gain that information
            gain respectively
    """

    # performing inplace sort causes pandas to show a warning during runtime
    df = df.sort_values(by=feature, inplace=False)

    feature_max_info_gain = 0
    feature_max_info_gain_split = 0

    # caching these values outside of the loop so they only need to be
    # computed once
    df_impurity = impurity_function(df)
    len_df = len(df)

    for row_idx in range(len(df.index) - 1):

        # after the sort above, rows are not ordered by indices 0, 1, 2, ...
        # need to traverse df.index sequentially to compare adjacent rows
        nth_row_index = df.index[row_idx]
        n_plus_oneth_row_index = df.index[row_idx + 1]

        # arranging if/else this way should increase performance from branch prediction
        if df.loc[nth_row_index, target_column] == df.get(target_column)[n_plus_oneth_row_index]:
            continue
        else:
            split_value = (df.get(feature)[nth_row_index] + df.get(feature)[n_plus_oneth_row_index]) / 2

            if split_value == min(df.get(feature)) or split_value == max(df.get(feature)):
                continue

            info_gain = df_impurity
            info_gain -= len(df.loc[df[feature] < split_value]) / len_df * impurity_function(df.loc[df[feature] < split_value])
            info_gain -= len(df.loc[df[feature] >= split_value]) / len_df * impurity_function(df.loc[df[feature] >= split_value])

            # arranging if/else this way should increase performance from branch prediction
            if info_gain <= feature_max_info_gain:
                continue
            else:
                feature_max_info_gain = info_gain
                feature_max_info_gain_split = split_value
                
    return (feature_max_info_gain, feature_max_info_gain_split)


def get_list_of_probabilities_classification(df: pd.DataFrame, imbalance_factor: float=1.0) -> np.array:
    """ Returns a series of probabilites of occurences of different values in the feature column

        Parameters:
            df: The data before splitting on feature
            imbalance_factor: the imbalance between classes in the original dataset

        Returns:
            an array containing the proportion of each class in relation to the total
                number of instances, in no specified order
    """

    unique_targets = pd.unique(df[target_column])
    target_proportions = np.zeros(len(unique_targets))
    for (idx, target) in enumerate(unique_targets):
        if target == 0:
            target_proportions[idx] = len(df.loc[df[target_column] == target]) / len(df)
        else:
            target_proportions[idx] = imbalance_factor * len(df.loc[df[target_column] == target]) / len(df)
    return target_proportions
    

def get_entropy_score(df: pd.DataFrame, imbalance_factor: float=1.0) -> float:
    """ Returns the entropy calculation of dataframe by considering feature column as the target

        Parameters:
            df: The data before splitting on feature
            imbalance_factor: the imbalance between classes in the original dataset

        Returns:
            the impurity of the dataset using the entropy metric
    """

    # split this into two lines so that the function call to get_list_of_probabilities_classification()
    # only occurs once (small optimization)
    classification_prob_list = get_list_of_probabilities_classification(df, imbalance_factor)
    return -1 * np.sum(np.multiply(classification_prob_list, np.log2(classification_prob_list)))


def get_gini_score(df: pd.DataFrame, imbalance_factor: float=1.0) -> float:
    """ Returns the gini_index calculation of dataframe by considering feature column as the target

        Parameters:
            df: The data before splitting on feature
            imbalance_factor: the imbalance between classes in the original dataset

        Returns:
            the impurity of the dataset using the Gini index metric
    """
    
    return 1 - np.sum(np.square(get_list_of_probabilities_classification(df, imbalance_factor)))

    
def get_misclassification_score(df: pd.DataFrame, imbalance_factor: float=1.0) -> float:
    """ Returns the gini_index calculation of dataframe by considering feature column as the target

        Parameters:
            df: The data before splitting on feature
            imbalance_factor: the imbalance between classes in the original dataset

        Returns:
            the impurity of the dataset using the misclassification error metric
    """
    return 1 - np.max(get_list_of_probabilities_classification(df, imbalance_factor))


def get_chi_squared_value_categorical(df: pd.DataFrame, feature: str, imbalance_factor: float=1.0) -> float:
    """ Computes the chi squared value of the provided dataset to be used in comparison
        to a value from the chi squared table

        Refer to class notes from 01/30 for presentation of this algorithm

        Parameters:
            df: A Pandas dataframe
            feature: The categorical feature of the dataframe on which the split is being made
            imbalance_factor: the imbalance between classes in the original dataset

        Returns:
            float: The chi square value of the dataset on the indicated feature split
    """

    chi_squared_value = 0
    num_total_instances = len(df)
    for feature_value in pd.unique(df.get(feature)):

        # calculate the number of instances with specific feature value
        num_feature_value_split = len(df.loc[df[feature] == feature_value])

        for target in pd.unique(df.get(target_column)):

            # calculate the number of instances with specific feature value and
            # specific target value
            num_feature_value_split_with_target = len(df.loc[df[target_column] == target])

            # weight class counts due to imbalanced dataset
            if target == 1:
                num_feature_value_split_with_target *= imbalance_factor

            expectation = num_feature_value_split * num_feature_value_split_with_target / num_total_instances
            observation = len(df.loc[(df[feature] == feature_value) & (df[target_column] == target)])

            chi_squared_value += ((observation - expectation) ** 2) / expectation

    return chi_squared_value


def get_chi_squared_value_continuous(df: pd.DataFrame, feature: str, cutoff_value: float, imbalance_factor: float=1.0) -> float:
    """ Computes the chi squared value of the provided dataset to be used in comparison
        to a value from the chi squared table

        Refer to class notes from 01/30 for presentation of this algorithm

        Parameters:
            df: A Pandas dataframe
            feature: The continuous feature of the dataframe on which the split is being made
            cutoff_value (float): The cutoff point where the values of the provided
                feature are split into different branches
            imbalance_factor: the factor by which to artificially boost counts of the unrepresented
                class due to imbalanced dataset

        Returns:
            float: The chi squared value of the dataset on the indicated feature split
    """

    chi_squared_value = 0
    num_total_instances = len(df)

    for target in pd.unique(df.get(target_column)):

        applied_imbalance_factor = imbalance_factor if target == 1 else 1.0 

        expectation = len(df.loc[df[feature] < cutoff_value]) * len(df.loc[df[target_column] == target]) / num_total_instances
        observation = len(df.loc[(df[feature] < cutoff_value) & (df[target_column] == target)])

        expectation *= applied_imbalance_factor
        observation *= applied_imbalance_factor

        chi_squared_value += ((observation - expectation) ** 2) / expectation

        expectation = len(df.loc[df[feature] >= cutoff_value]) * len(df.loc[df[target_column] == target]) / num_total_instances
        observation = len(df.loc[(df[feature] >= cutoff_value) & (df[target_column] == target)])

        expectation *= applied_imbalance_factor
        observation *= applied_imbalance_factor

        chi_squared_value += ((observation - expectation) ** 2) / expectation

    return chi_squared_value
