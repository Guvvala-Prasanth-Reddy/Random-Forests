import pandas as pd
import math
import numpy as np
from utils.consts import *


def get_info_gain( error_function , df: pd.DataFrame , feature:str , target:str , entire_data_set : pd.DataFrame):
    """ Returns the information gain when the data in the dataframe df is split using the different values in the feature column

        Parameters:
            error_function : impurity function eg : entropy , gini_index , missclassification_error
            df : The data before splitting on feature
            feature : the feature i.e being considered for splitting criteria
    """
    if( feature in categorical_columns):
        impurity_data_set = error_function(entire_data_set , target) 
        impurity_split_summation = impurity_data_set
        for i in pd.unique(df.get(feature)):
            filtered_df = df.loc[df[feature] == i]
            impurity_split_summation -= filtered_df.shape[0] * error_function( df = filtered_df , feature= target) 
        return impurity_split_summation
    else:
        #TODO: implememnt the information gain for the regression values
        pass



def get_list_of_probabilities_classification( df , feature ):
    """ Returns a series of probabilites of occurences of different values in the feature column

        Parameters:
            df : The data before splitting on feature
            feature : the feature i.e being considered for splitting criteria
    """
    record_size = len(df.get(feature))
    feature_values = pd.unique(df.get(feature))
    return np.array( lambda x : df.get(feature).count(x)/record_size  ( df.get(feature_values)) )

def get_entropy_score(df : pd.DataFrame  , feature : str  ):
    """ Returns the entropy calculation of dataframe by considering feature column as the target

        Parameters:
            df : The data before splitting on feature
            feature : the feature i.e being considered for splitting criteria ( mainly target )
    """
    return np.sum( np.multiply( get_list_of_probabilities_classification(df , feature)  ,  np.log2( get_list_of_probabilities_classification(df,feature) )))


def get_gini_score(df : pd.DataFrame  , feature : str):
    """ Returns the gini_index calculation of dataframe by considering feature column as the target

        Parameters:
            df : The data before splitting on feature
            feature : the feature i.e being considered for splitting criteria ( mainly target )
    """
    return ( 1 - np.square(get_list_of_probabilities_classification(df,feature)))

    
def get_misclassification_score(df : pd.DataFrame  , feature : str):
    """ Returns the gini_index calculation of dataframe by considering feature column as the target

        Parameters:
            df : The data before splitting on feature
            feature : the feature i.e being considered for splitting criteria ( mainly target )
    """
    return 1 - max( get_list_of_probabilities_classification(df , feature))

def get_chi_squared_value(df: pd.DataFrame, feature: str) -> float:
    """ Computes the chi squared value of the provided dataset to be used in comparison
        to a value from the chi squared table

        Refer to class notes from 01/30 for presentation of this algorithm

        Parameters:
            df: A Pandas dataframe
            feature: The column of the dataframe on which the split is being made

        Returns:
            float: The chi squared value of the dataset on the indicated feature split
    """

    chi_squared_value = 0
    num_total_instances = len(df)
    for feature_value in pd.unique(df.get(feature)):

        # calculate the number of instances with specific feature value
        num_feature_value_split = len(df.loc[df[feature] == feature_value])

        for target in pd.unique(df.get('isFraud')):

            # calculate the number of instances with specific feature value and
            # specific target value
            num_feature_value_split_with_target = len(df.loc[df['isFraud'] == target])

            expectation = num_feature_value_split * num_feature_value_split_with_target / num_total_instances
            observation = len(df.loc[df[feature] == feature_value &
                                                   df['isFraud'] == target])

            chi_squared_value += ((expectation - observation) ** 2) / observation

    return chi_squared_value
