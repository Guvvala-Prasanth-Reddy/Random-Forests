import pandas as pd
from scipy.stats import chi2
from utils.consts import *
from tree.Leaf import Leaf
from tree.Branch import Branch
from tree.Node import Node
from utils.impurity import *

MISSING_VALUE_TERMS = ['notFound', float('NaN'), 'NaN']

def handle_missing_values(df):
    """ Cleans data frame of missing values.

    Parameters:
        df: a Pandas dataframe

    Returns:
        The provided dataframe without missing values
    """

    print(f'Number of rows in dataframe before handling missing values: {len(df)}')

    # remove instances for which the target is not known
    for i in range(len(MISSING_VALUE_TERMS)):
        df = df.loc[df['isFraud'] != MISSING_VALUE_TERMS[i]]

    # for missing feature values, replace by the average (for
    # continuous features) or most common value (for categorical
    # features) of the instances sharing the same target
        
    # step 1: determine most common value for all categorical features,
    # per target
    most_common_cat_features_pos = dict()
    most_common_cat_features_neg = dict()
    for categorical_feature in categorical_features:
        most_common_cat_feature_pos = df.loc[df['isFraud'] == 1].get(categorical_feature).mode()
        most_common_cat_feature_neg = df.loc[df['isFraud'] == 0].get(categorical_feature).mode()

        most_common_cat_features_pos[categorical_feature] = most_common_cat_feature_pos
        most_common_cat_features_neg[categorical_feature] = most_common_cat_feature_neg

    

    print(f'Number of rows in dataframe after handling missing values: {len(df)}')

    return df



def build_tree(df, split_metric='entropy'):
    """ Returns a decision tree object built upon the provided data.

        Parameters:
            df: a Pandas dataframe
            split_metric (str): indicates which split metric to use
                with information gain. Options are the following:
                    'entropy': uses entropy
                    'gini': uses Gini index
                    'misclassification': uses misclassification error

        Returns:
            A decision tree object

    """

    # check whether all targets are the same in the provided dataset
    if len(pd.unique(df['isFraud'])) == 1:
        print('All the targets are the same')
        return Leaf(df['isFraud'][0])

    # assign a score to each possible data split
    feature_scores = dict()
    for feature in df.columns:
        if feature != 'isFraud' and feature != 'transactionID':
            if split_metric == 'entropy':
                feature_scores[feature] = get_info_gain(get_entropy_score, df, feature)
            elif split_metric == 'gini':
                feature_scores[feature] = get_info_gain(get_gini_score, df, feature)
            else:
                feature_scores[feature] = get_info_gain(get_misclassification_score, df, feature)

    # determine which feature has the max information gain
    max_score_feature = ''
    max_feature_score = float('-inf')
    for feature in feature_scores.keys:
        if feature_scores[feature] > max_feature_score:
            max_feature_score = feature_scores[feature]
            max_score_feature = feature

    # check if split is recommended by chi squared test
    split_chi_squared_metric = get_chi_squared_metric(df, max_score_feature)
    degrees_of_freedom = (len(pd.unique(df[max_score_feature])) - 1) * (len(pd.unique(df['isFraud'])) - 1)
    chi_squared_table_value = chi2.ppf(0.05, degrees_of_freedom)
    if chi_squared_table_value > split_chi_squared_metric:
        return Leaf(df['isFraud'][0])
    
    # create branches from the node for all attributes of the selected feature
    node = Node(max_score_feature)
    for feature_value in pd.unique(df[max_score_feature]):
        branch = Branch(feature_value,
                        build_tree(df[max_score_feature] == feature_value, split_metric))
        node.add_branch(branch)
    
    return node


if __name__ == "__main__":
    """
    Main method for testing
    """

    df = pd.read_csv("../data/train.csv")
    handle_missing_values(df)
    #build_tree(df)
