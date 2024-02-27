import pandas as pd
from scipy.stats import chi2   
from utils.consts import *
from tree.Leaf import Leaf
from tree.Branch import Branch
from tree.Node import Node
from utils.impurity import *
import time
import statistics as st
import pickle
from validation.validate import get_tree_acc
from utils.dataframeutils import split_data

MISSING_VALUE_TERMS = ['notFound', float('NaN'), 'NaN']

def handle_missing_values(df):
    """ Cleans data frame of missing values.

    Parameters:
        df: a Pandas dataframe

    Returns:
        The provided dataframe without missing values
    """

    # remove instances for which the target is not known
    for i in range(len(MISSING_VALUE_TERMS)):
        df = df.loc[df[target_column] != MISSING_VALUE_TERMS[i]]

    # for missing feature values, replace by the average (for
    # continuous features) or most common value (for categorical
    # features) of the instances sharing the same target
    df.replace('NotFound', float('nan'), inplace=True)

    feature_missing_value_replacement_dict = dict()
    for target_val in pd.unique(df[target_column]):
        feature_missing_value_replacement_dict[target_val] = dict()
        for feature in df.columns:
            if feature in categorical_features:
                feature_missing_value_replacement_dict[target_val][feature] = df.loc[df[target_column] == target_val].get(feature).mode()[0]
            else:
                feature_missing_value_replacement_dict[target_val][feature] = df.loc[df[target_column] == target_val].get(feature).mean()

    for target in pd.unique(df[target_column]):
        df1 = df.loc[df[target_column] == target].fillna(value = feature_missing_value_replacement_dict[target]).copy()
        df.loc[df1.index] = df1

    return df


def build_tree(df, seen_features=set(), split_metric='entropy', level=0):
    """ Returns a decision tree object built upon the provided data.

        Parameters:
            df: a Pandas dataframe
            seen_features: a set of features which have already been
                used to split data at some point in this path
            split_metric (str): indicates which split metric to use
                with information gain. Options are the following:
                    'entropy': uses entropy
                    'gini': uses Gini index
                    'misclassification': uses misclassification error

        Returns:
            A decision tree object

    """

    # check whether all targets are the same in the provided dataset
    if len(pd.unique(df[target_column])) == 1:
        return Leaf(list(df[target_column])[0])

    # record the categorical feature with the highest information gain, as well as its
    # corresponding information gain value
    max_score_feature = ''
    max_feature_score = float('-inf')
    for categorical_feature in df.columns:
        if (not (categorical_feature in seen_features)) and (categorical_feature in categorical_features):
            feature_info_gain = 0
            if split_metric == 'entropy':
                feature_info_gain = get_info_gain_categorical(get_entropy_score, df, categorical_feature)
            elif split_metric == 'gini':
                feature_info_gain = get_info_gain_categorical(get_gini_score, df, categorical_feature)
            else:
                feature_info_gain = get_info_gain_categorical(get_misclassification_score, df, categorical_feature)

            if feature_info_gain > max_feature_score:
                max_feature_score = feature_info_gain
                max_score_feature = categorical_feature

    # if any continuous features have a higher information gain than the categorical feature
    # with the highest information gain, record this feature (and its split cutoff) instead
    continuous_cutoff_value = 0
    for continuous_feature in df.columns:
        if (not (continuous_feature in categorical_features)) and (
           continuous_feature != target_column and continuous_feature != 'TransactionID'
           ):

            (feature_info_gain, feature_cutoff_value) = get_info_gain_continuous_cuda(df, continuous_feature, split_metric)
            
            if feature_info_gain > max_feature_score:
                max_feature_score = feature_info_gain
                max_score_feature = continuous_feature
                continuous_cutoff_value = feature_cutoff_value

    if max_score_feature in categorical_features:
        seen_features.add(max_score_feature)

    if max_feature_score == 0 or max_score_feature == '':
        return Leaf(st.mode(np.array(df[target_column])))
    
    # check if split is recommended by chi squared test
    split_chi_squared_value = 0
    degrees_of_freedom = 0
    if max_score_feature in categorical_features:
        split_chi_squared_value = get_chi_squared_value_categorical(df, max_score_feature)
        degrees_of_freedom = (len(pd.unique(df[max_score_feature])) - 1) * (len(pd.unique(df[target_column])) - 1)
    else:
        split_chi_squared_value = get_chi_squared_value_continuous(df, max_score_feature, continuous_cutoff_value)
        degrees_of_freedom = len(pd.unique(df[target_column])) - 1
    chi_squared_table_value = chi2.ppf(confidence_interval, degrees_of_freedom)

    if chi_squared_table_value > split_chi_squared_value:
        return Leaf(st.mode(np.array(df[target_column])))
    
    # create branches from the node for all attributes of the selected feature
    node = Node(max_score_feature)
    if max_score_feature in categorical_features:
        for feature_value in pd.unique(df[max_score_feature]):
            branch = Branch(feature_value,
                            build_tree(df.loc[df[max_score_feature] == feature_value],
                                       seen_features=seen_features, 
                                       split_metric=split_metric, 
                                       level=level+1))
            node.add_branch(branch)
    else:
        less_than_branch = Branch('<' + str(continuous_cutoff_value),
                                  build_tree(df.loc[df[max_score_feature] < continuous_cutoff_value], 
                                             seen_features=seen_features,
                                             split_metric=split_metric,
                                             level=level+1))
        greater_than_branch = Branch('>=' + str(continuous_cutoff_value),
                                     build_tree(df.loc[df[max_score_feature] >= continuous_cutoff_value], 
                                                seen_features=seen_features,
                                                split_metric=split_metric,
                                                level=level+1))
        node.add_branch(less_than_branch)
        node.add_branch(greater_than_branch)
    
    return node


if __name__ == "__main__":
    """
    Main method for testing
    """

    start_time = time.time()

    # read entire training dataset and handle missing values
    whole_training_data = pd.read_csv('data/train.csv')
    whole_training_data = handle_missing_values(whole_training_data)

    # divide into separate training and testing datasets
    (training_data, testing_data) = split_data(whole_training_data, 1, 1, False)

    is_fraud_rows = training_data.loc[training_data[target_column] == 1]
    is_not_fraud_rows = training_data.loc[training_data[target_column] == 0]
    sampled_is_not_fraud_rows = is_not_fraud_rows.sample(frac=1.0)
    sampled_training_data = pd.concat([is_fraud_rows, sampled_is_not_fraud_rows], axis=0).reset_index(drop=True)

    tree = build_tree(sampled_training_data)
    tree_build_time = time.time() - start_time
    print(f'Time to build tree ({len(sampled_training_data)} rows): {time.time() - start_time} seconds')
    start_time = time.time()

    tree_acc = get_tree_acc(tree, testing_data)
    print(f'Tree accuracy (generated in {time.time() - start_time} seconds): {tree_acc}')

    # save tree model to file
    file = open(f'tree-{tree_acc}-acc-{int(tree_build_time)}-seconds-{len(sampled_training_data)}-of-{len(whole_training_data)}-rows', 'wb')
    pickle.dump(tree, file)
    file.close()
