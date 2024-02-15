import pandas as pd
import numpy as np
from utils.consts import *


def get_info_gain_categorical(impurity_function , df: pd.DataFrame , feature:str) -> float:
    """ Returns the information gain of the provided dataset considering a split on the provided
        categorical feature.

        Parameters:
            impurity_function : impurity function eg : entropy , gini_index , missclassification_error
            df : The data before splitting on feature
            feature : the feature i.e being considered for splitting criteria
    """

    info_gain = impurity_function(df)
    for feature_value in pd.unique(df[feature]):
        info_gain -= len(df.loc[df[feature] == feature_value]) / len(df) * impurity_function(df.loc[df[feature] == feature_value])
    return info_gain


def get_info_gain_continuous(impurity_function , df: pd.DataFrame , feature:str) -> float:
    """ Returns the information gain of the provided dataset considering a split on the provided
        continuous feature.

    Parameters:
        impurity_function : impurity function eg : entropy , gini_index , missclassification_error
        df : The data before splitting on feature
        feature : the feature i.e being considered for splitting criteria

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
    # computed oncec
    df_impurity = impurity_function(df)
    len_df = len(df)

    for row_idx in range(len(df.index) - 1):

        # df is indexed by TransactionID, not from 0, 1, 2, ...
        # so in order to traverse the rows sequentially by index, we
        # must sequentially get the values of df.index
        nth_row_index = df.index[row_idx]
        n_plus_oneth_row_index = df.index[row_idx + 1]

        # arranging if/else this way should increase performance from branch prediction
        if df.get(target_column)[nth_row_index] == df.get(target_column)[n_plus_oneth_row_index]:
            continue
        else:
            split_value = (df.get(feature)[nth_row_index] + df.get(feature)[n_plus_oneth_row_index]) / 2

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


def get_list_of_probabilities_classification(df: pd.DataFrame) -> np.array:
    """ Returns a series of probabilites of occurences of different values in the feature column

        Parameters:
            df : The data before splitting on feature

        Returns:
            (np.array): An array containing the proportion of each class in relation to the total
                number of instances, in no specified order
    """

    unique_targets = pd.unique(df[target_column])
    target_proportions = np.zeros(len(unique_targets))
    for (idx, target) in enumerate(unique_targets):
        target_proportions[idx] = len(df.loc[df[target_column] == target]) / len(df)
    return target_proportions
    

def get_entropy_score(df: pd.DataFrame) -> float:
    """ Returns the entropy calculation of dataframe by considering feature column as the target

        Parameters:
            df : The data before splitting on feature
    """

    # split this into two lines so that the function call to get_list_of_probabilities_classification()
    # only occurs once (small optimization)
    classification_prob_list = get_list_of_probabilities_classification(df)
    return -1 * np.sum(np.multiply(classification_prob_list, np.log2(classification_prob_list)))


def get_gini_score(df: pd.DataFrame) -> float:
    """ Returns the gini_index calculation of dataframe by considering feature column as the target

        Parameters:
            df : The data before splitting on feature
    """
    return 1 - np.sum(np.square(get_list_of_probabilities_classification(df)))

    
def get_misclassification_score(df: pd.DataFrame) -> float:
    """ Returns the gini_index calculation of dataframe by considering feature column as the target

        Parameters:
            df : The data before splitting on feature
            feature : the feature i.e being considered for splitting criteria ( mainly target )
    """
    return 1 - np.max(get_list_of_probabilities_classification(df))

def get_chi_squared_value_categorical(df: pd.DataFrame, feature: str) -> float:
    """ Computes the chi squared value of the provided dataset to be used in comparison
        to a value from the chi squared table

        Refer to class notes from 01/30 for presentation of this algorithm

        Parameters:
            df: A Pandas dataframe
            feature: The categorical feature of the dataframe on which the split is being made

        Returns:
            float: The chi squared value of the dataset on the indicated feature split
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

            expectation = num_feature_value_split * num_feature_value_split_with_target / num_total_instances
            observation = len(df.loc[(df[feature] == feature_value) & (df[target_column] == target)])

            chi_squared_value += ((observation - expectation) ** 2) / expectation

    return chi_squared_value

def get_chi_squared_value_continuous(df: pd.DataFrame, feature: str, cutoff_value: float) -> float:
    """ Computes the chi squared value of the provided dataset to be used in comparisonß
        to a value from the chi squared table

        Refer to class notes from 01/30 for presentation of this algorithm

        Parameters:
            df: A Pandas dataframe
            feature: The continuous feature of the dataframe on which the split is being made
            cutoff_value (float): The cutoff point where the values of the provided
                feature are split into different branches

        Returns:
            float: The chi squared value of the dataset on the indicated feature split
    """

    chi_squared_value = 0
    num_total_instances = len(df)
    for target in pd.unique(df.get(target_column)):

        expectation = len(df.loc[df[feature] < cutoff_value]) * len(df.loc[df[target_column] == target]) / num_total_instances
        observation = len(df.loc[(df[feature] < cutoff_value) & (df[target_column] == target)])

        chi_squared_value += ((observation - expectation) ** 2) / expectation

        expectation = len(df.loc[df[feature] >= cutoff_value]) * len(df.loc[df[target_column] == target]) / num_total_instances
        observation = len(df.loc[(df[feature] >= cutoff_value) & (df[target_column] == target)])

        chi_squared_value += ((observation - expectation) ** 2) / expectation

    return chi_squared_value
