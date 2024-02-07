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

