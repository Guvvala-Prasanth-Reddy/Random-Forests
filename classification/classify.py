import numpy as np
import pandas as pd
import statistics as st
from tree.Leaf import Leaf
from tree.Tree import Tree
from utils.consts import *
from utils.model_file_utils import *
from utils.dataframeutils import handle_missing_values_test


def tree_classify(df: pd.DataFrame, tree: Tree) -> str:
    """ Returns the decision tree classification of the given instance

        Parameters:
            df: a single row pandas DataFrame without a class value
            tree: Either a Leaf or Node object representing the decision
                tree used to make the classification

        Returns:
            the class label predicted by the Tree
    """

    # case 1: tree is a leaf
    if type(tree) is Leaf:
        return tree.target

    # case 2: tree is a node and feature is categorical
    if tree.feature in categorical_features:
        for branch in tree.branches:
            if df[tree.feature] == branch.feature_value:
                return tree_classify(df, branch.tree)
    # case 3: tree is a node and feature is continuous
    else:
        split_cutoff_value = tree.branches[0].feature_value.replace('<', '')
        feature_value = df[tree.feature]

        if feature_value < float(split_cutoff_value):
            return tree_classify(df, tree.branches[0].tree)
        else:
            return tree_classify(df, tree.branches[1].tree)
    
    
def forest_classify(df: pd.DataFrame, forest: Forest) -> str:
    """ Classifies the given 1 row DataFrame using majority voting with the
        provided Forest model

        Parameters:
            df: a single row pandas DataFrame without a class value
            forest: a Forest model object

        Returns:
            the class label predicted by the Forest
    """
    
    classification_results = np.full(len(forest.trees), '', dtype='str')
    for (idx, tree) in enumerate(forest.trees):
        classification_results[idx] = tree_classify(df, tree)

    return st.mode(np.array(classification_results))


def generate_predictions_file(model_file_path: str, output_file: str='output.csv'):
    """ Generates a CSV file of class predictions using a Forest model

        Parameters:
            model_file_path: the path to the model file used to generate the
                predictions file
            output_file: the name of the output predictions file, defaults to
                'output.csv'

    """

    # read forest model from file
    forest_model = read_forest_model(model_file_path)

    # read testing data and hanlde missing values
    testing_df = pd.read_csv('data/test.csv')
    testing_df = handle_missing_values_test(testing_df)

    # generate predictions and concatenate with transaction ids
    predictions = testing_df.apply(forest_classify, args=(forest_model,), axis=1)
    predictions_with_ids = pd.concat([testing_df.get('TransactionID'), predictions], axis=1).reset_index(drop=True)

    # fill in any missing predictions with specified value
    predictions_with_ids.rename(columns={0: target_column}, inplace=True)
    predictions_with_ids.fillna(0, inplace=True)
    predictions_with_ids[target_column] = predictions_with_ids[target_column].astype(int)
    
    predictions_with_ids.to_csv(output_file, index=False)

# use this main function to generate CSV file by calling this file, just replace forest_model_name
if __name__ == '__main__':
    generate_predictions_file('models/forest-10-trees-0.7966742360728793-acc-alpha-0.15', output_file='output.csv')    
