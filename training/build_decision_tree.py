import pickle
import time
import pandas as pd
from scipy.stats import chi2   
from tree.Branch import Branch
from tree.Leaf import Leaf
from tree.Node import Node
from tree.Tree import Tree
from utils.consts import *
from utils.impurity import *
from utils.dataframeutils import split_data, handle_missing_values_train
from validation.validate import get_tree_acc


def build_tree(df: pd.DataFrame, seen_features: set[str], split_metric: str='entropy', imbalance_factor: float=1.0, level: int=0) -> Tree:
    """ Returns a decision tree object built upon the provided data.

        Parameters:
            df: a pandas dataframe
            seen_features: a set of categorical features which have already been used to split
                data at some point in this path
            split_metric: indicates which information gain split metric to use. Options are the
                following:
                    'entropy': uses entropy
                    'gini': uses Gini index
                    'misclassification': uses misclassification error
            imbalance_factor: the boost by which to increase the minority class due to imbalanced
                data
            level: the depth below the root at which this iteration resides

        Returns:
            A decision tree object

    """

    # if all class values are the same, return a leaf with that class
    if len(pd.unique(df[target_column])) == 1:
        return Leaf(list(df[target_column])[0])

    # limit tree depth to predefined value, if value exceeded, return a leaf with weighted
    # mean/mode of dataset
    if level == max_depth and level > 0:
        if len(df.loc[df[target_column] == 0]) > imbalance_factor * len(df.loc[df[target_column] == 1]):
            return Leaf(0)
        else:
            return Leaf(1)

    # record the categorical feature with the highest information gain, as well as its
    # corresponding information gain value
    max_score_feature = ''
    max_feature_score = float('-inf')
    for categorical_feature in df.columns:
        if (not (categorical_feature in seen_features)) and (categorical_feature in categorical_features):
            feature_info_gain = 0
            if split_metric == 'entropy':
                feature_info_gain = get_info_gain_categorical(get_entropy_score, df, categorical_feature, imbalance_factor=imbalance_factor)
            elif split_metric == 'gini':
                feature_info_gain = get_info_gain_categorical(get_gini_score, df, categorical_feature, imbalance_factor=imbalance_factor)
            else:
                feature_info_gain = get_info_gain_categorical(get_misclassification_score, df, categorical_feature, imbalance_factor=imbalance_factor)

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

            (feature_info_gain, feature_cutoff_value) = get_info_gain_continuous_cuda(df, continuous_feature, split_metric, imbalance_factor=imbalance_factor)
            if feature_info_gain > max_feature_score:
                max_feature_score = feature_info_gain
                max_score_feature = continuous_feature
                continuous_cutoff_value = feature_cutoff_value

    # update set of seen features if a categorical feature is selected to split upon
    if max_score_feature in categorical_features:
        seen_features.add(max_score_feature)

    if max_feature_score == 0 or max_score_feature == '':
        if len(df.loc[df[target_column] == 0]) > imbalance_factor * len(df.loc[df[target_column] == 1]):
            return Leaf(0)
        else:
            return Leaf(1)
            
    # check if split is recommended by chi square test
    split_chi_squared_value = 0
    degrees_of_freedom = 0
    if max_score_feature in categorical_features:
        split_chi_squared_value = get_chi_squared_value_categorical(df, max_score_feature, imbalance_factor=imbalance_factor)
        degrees_of_freedom = (len(pd.unique(df[max_score_feature])) - 1) * (len(pd.unique(df[target_column])) - 1)
    else:
        split_chi_squared_value = get_chi_squared_value_continuous(df, max_score_feature, continuous_cutoff_value, imbalance_factor=imbalance_factor)
        degrees_of_freedom = len(pd.unique(df[target_column])) - 1
    chi_squared_table_value = chi2.ppf(chi_square_alpha, degrees_of_freedom)

    # if chi square test doesn't recommend a split, return a leaf
    if chi_squared_table_value > split_chi_squared_value:
        if len(df.loc[df[target_column] == 0]) > imbalance_factor * len(df.loc[df[target_column] == 1]):
            return Leaf(0)
        else:
            return Leaf(1)
    
    # create branches from the node for all attributes of the selected feature
    node = Node(max_score_feature)
    if max_score_feature in categorical_features:
        for feature_value in pd.unique(df[max_score_feature]):
            branch = Branch(feature_value,
                            build_tree(df.loc[df[max_score_feature] == feature_value],
                                       seen_features=seen_features, 
                                       split_metric=split_metric,
                                       imbalance_factor=imbalance_factor, 
                                       level=level+1))
            node.add_branch(branch)
    else:
        less_than_branch = Branch('<' + str(continuous_cutoff_value),
                                  build_tree(df.loc[df[max_score_feature] < continuous_cutoff_value], 
                                             seen_features=seen_features,
                                             split_metric=split_metric,
                                             imbalance_factor=imbalance_factor,
                                             level=level+1))
        greater_than_branch = Branch('>=' + str(continuous_cutoff_value),
                                     build_tree(df.loc[df[max_score_feature] >= continuous_cutoff_value], 
                                                seen_features=seen_features,
                                                split_metric=split_metric,
                                                imbalance_factor=imbalance_factor,
                                                level=level+1))
        node.add_branch(less_than_branch)
        node.add_branch(greater_than_branch)
    
    return node


if __name__ == "__main__":
    """ Main method for generating a single decision tree """

    # read entire training dataset and handle missing values
    whole_training_data = pd.read_csv(training_data_path)
    whole_training_data = handle_missing_values_train(whole_training_data)

    # divide into separate training and testing datasets
    (training_data, testing_data) = split_data(whole_training_data, 1, 1, False)

    # sample rows as desired
    is_fraud_rows = training_data.loc[training_data[target_column] == 1]
    is_not_fraud_rows = training_data.loc[training_data[target_column] == 0]
    sampled_is_not_fraud_rows = is_not_fraud_rows.sample(frac=1.0)
    sampled_training_data = pd.concat([is_fraud_rows, sampled_is_not_fraud_rows], axis=0).reset_index(drop=True)

    t0 = time.time()
    tree = build_tree(sampled_training_data, imbalance_factor=len(is_not_fraud_rows) / len(is_fraud_rows))
    print(f'Time to build tree ({len(sampled_training_data)} rows): {time.time() - t0} seconds')

    t0 = time.time()
    (tree_err, tree_acc) = get_tree_acc(tree, testing_data)
    print(f'Tree accuracy (generated in {time.time() - t0} seconds): {tree_acc}')

    # save tree model to file
    file = open(f'tree-{tree_acc}-acc', 'wb')
    pickle.dump(tree, file)
    file.close()
